"""
端到端模型噪声鲁棒性评估
负责人: Wenjuan Huang

与 noise_robustness_eval.py 的区别:
  - 旧版: 噪声嵌入缓存 -> 冻结分类头预测 (波形 -> YAMNet嵌入 -> 分类头)
  - 本版: 端到端模型直接对叠噪波形预测 (波形 -> 完整 e2e 模型)

噪声叠加方式与旧版完全一致 (高斯白噪声, 相同 SNR 公式, 相同种子,
相同 rng 推进方式), 保证三个模型的噪声衰减曲线可比。

模型加载: 用 build_e2e_model 重建架构 + load_weights (不依赖 model.save
保存的 .keras, 避免 hub.KerasLayer 加载时需联网的问题)。

Kaggle 运行方式: 在 cell1 训练完成后, 作为 cell2 运行:
  %run -i src/noise_eval_e2e.py
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

try:
    from yamnet_bird_pipeline import (
        Config, load_csv_splits, preflight_report, load_waveform,
    )
    from noise_robustness_eval import add_gaussian_noise, SNR_TIERS
    from unified_evaluation import plot_noise_decay
    from yamnet_finetune_e2e import FinetuneConfig as fcfg, build_e2e_model
except (ModuleNotFoundError, ImportError):
    pass


def _load_e2e_model(fold_dir, num_classes):
    """
    重建端到端模型并加载权重。
    不使用 tf.keras.models.load_model (避免 hub.KerasLayer 加载问题)。
    """
    weights_path = fold_dir / "best_weights.weights.h5"
    if not weights_path.exists():
        # 回退到旧格式
        keras_path = fold_dir / "yamnet_e2e_model.keras"
        if keras_path.exists():
            return tf.keras.models.load_model(str(keras_path))
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    model, _, _ = build_e2e_model(num_classes)
    model.load_weights(str(weights_path))
    return model


def _batched_predict(model, X, batch_size=16):
    """分批预测, 避免 GPU OOM (端到端模型前向占显存大)。"""
    preds_all = []
    n = len(X)
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size]
        preds = model.predict(batch, verbose=0)
        preds_all.append(np.argmax(preds, axis=1))
    return np.concatenate(preds_all)


def main_cv_e2e_noise(n_folds=5):
    """
    端到端模型 5 折噪声评估。

    流程: 对 ml_test.csv 每条音频叠噪 -> 直接喂入 e2e 模型预测。
    噪声 rng 与旧版一致 (SEED=42, 跨样本跨档推进), 保证对比公平。
    """
    cfg = Config()
    e2e_dir = cfg.OUT_DIR / "e2e"

    # 加载测试集
    df_train, df_val, df_test, missing = load_csv_splits(cfg)
    df_train, df_val, df_test = preflight_report(missing, df_train, df_val, df_test)
    if len(df_test) == 0:
        raise RuntimeError("测试集为空")

    # 读取标签映射
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    label2idx = label_map["label2idx"]
    num_classes = len(label2idx)

    # 预计算所有测试集波形
    print(f"[噪声评估] 预加载 {len(df_test)} 条测试波形 ...")
    clean_wfs = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        clean_wfs.append(load_waveform(row["filepath"]))
        if (i+1) % 100 == 0 or (i+1) == len(df_test):
            print(f"  {i+1}/{len(df_test)}")

    # 获取真实标签
    y_true = np.array([label2idx[str(l)] for l in df_test["primary_label"]], dtype=np.int64)

    # 噪声种子: 与旧版 noise_robustness_eval.py 完全一致
    # rng 只初始化一次, 跨样本跨 SNR 档推进 (不在每个 SNR 档重置)
    rng = np.random.default_rng(cfg.SEED)

    # 预计算各 SNR 档的叠噪波形 (与旧版相同的 rng 推进顺序)
    print(f"[噪声评估] 预计算噪声波形 ...")
    snr_vals = [("clean", None), ("5dB", 5.0), ("0dB", 0.0), ("-5dB", -5.0)]
    noisy_wfs_by_snr = {}
    for snr_key, snr_val in snr_vals:
        if snr_val is None:
            noisy_wfs_by_snr[snr_key] = np.stack(clean_wfs).astype(np.float32)
        else:
            wfs = []
            for wf in clean_wfs:
                wfs.append(add_gaussian_noise(wf, snr_val, rng))
            noisy_wfs_by_snr[snr_key] = np.stack(wfs).astype(np.float32)
        print(f"  {snr_key}: shape={noisy_wfs_by_snr[snr_key].shape}")

    rows = []
    for fold in range(1, n_folds + 1):
        fold_dir = e2e_dir / f"fold{fold}"
        weights_path = fold_dir / "best_weights.weights.h5"
        keras_path = fold_dir / "yamnet_e2e_model.keras"
        if not weights_path.exists() and not keras_path.exists():
            print(f"[跳过] fold{fold} 模型不存在: {fold_dir}")
            continue

        model = _load_e2e_model(fold_dir, num_classes)
        print(f"\n[噪声评估] fold{fold} ...")

        acc_by_snr = {}
        preds_by_snr = {}

        for snr_key in ["clean", "5dB", "0dB", "-5dB"]:
            X = noisy_wfs_by_snr[snr_key]
            preds = _batched_predict(model, X, batch_size=16)
            acc = float(np.mean(preds == y_true))
            acc_by_snr[snr_key] = acc
            preds_by_snr[snr_key] = preds
            print(f"  {snr_key}: {acc:.4f}")

        # 保存结果
        np.savez(fold_dir / "noise_results.npz",
                 snr_tiers=np.array(SNR_TIERS),
                 acc=np.array([acc_by_snr[s] for s in SNR_TIERS]),
                 y_true=y_true,
                 preds_clean=preds_by_snr["clean"],
                 preds_5dB=preds_by_snr["5dB"],
                 preds_0dB=preds_by_snr["0dB"],
                 preds_n5dB=preds_by_snr["-5dB"])

        rows.append({"fold": fold, **{f"acc_{s}": acc_by_snr[s] for s in SNR_TIERS}})

    # 汇总
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(e2e_dir / "cv_noise_per_fold.csv", index=False)

    # 合并到 cv_summary
    summary_path = e2e_dir / "cv_summary.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() \
        else pd.DataFrame(columns=["metric", "mean", "std"])
    noise_metrics = [f"acc_{s}" for s in SNR_TIERS]
    new_rows = pd.DataFrame([{
        "metric": m,
        "mean": float(per_fold[m].mean()),
        "std": float(per_fold[m].std(ddof=0)),
    } for m in noise_metrics])
    pd.concat([existing, new_rows], ignore_index=True).to_csv(summary_path, index=False)

    print(f"\n[CV 噪声] e2e 汇总 (mean ± std):")
    for m in noise_metrics:
        print(f"  {m}: {per_fold[m].mean():.4f} ± {per_fold[m].std(ddof=0):.4f}")
    print(f"\n对比旧版 (冻结): clean=1.91%, 5dB=0.42%, 0dB=0.37%, -5dB=0.12%")

    # 衰减曲线
    mean_accs = {k: float(per_fold[f"acc_{k}"].mean()) for k in SNR_TIERS}
    plot_noise_decay({"YAMNet_e2e": mean_accs})


if __name__ == "__main__":
    main_cv_e2e_noise()
