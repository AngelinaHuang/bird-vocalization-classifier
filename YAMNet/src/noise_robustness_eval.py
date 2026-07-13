"""
噪声鲁棒性测试
负责人: Wenjuan Huang

对测试集干净音频叠加不同强度的噪声, 评估训练好的模型在噪声下的识别率,
比较各模型的抗噪能力。噪声越强而准确率下降越少的模型, 鲁棒性越好。

噪声强度以信噪比 SNR (dB) 衡量:
  5 dB   噪声较弱
  0 dB   噪声与信号等强
  -5 dB  噪声强于信号
另设 clean 档 (不叠噪声) 作为基线, 对应训练阶段报告的准确率。

输出各 SNR 下的准确率, 交由 unified_evaluation.plot_noise_decay 绘制衰减曲线。

实现要点:
  - 噪声叠加在原始波形上, 需重新经 YAMNet 提取嵌入; clean 档复用训练缓存,
    三个噪声档重新计算。
  - 采用高斯白噪声作为可控、可复现的基准; 若需更真实的噪声, 替换 add_gaussian_noise
    即可, 接口不变。
  - 测试集直接使用 ml_test.csv, 与干净基线样本完全一致。
  - 5 折模式 (main_cv): 噪声嵌入只与波形+种子有关、与模型无关, 故算一次缓存
    (noise_embeddings.npz), 5 折模型各自在该缓存上预测, 避免重复 YAMNet。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# 复用管道函数。兼容两种运行方式: 直接以 .py 运行 (src 在 sys.path) 时正常 import;
# 在 notebook 中按顺序执行过 yamnet_bird_pipeline.py 后再运行本文件时, 这些名字
# 已存在于全局命名空间, 此时 import 失败, 走 except 直接复用已定义的名称。
try:
    from yamnet_bird_pipeline import (
        Config, load_csv_splits, preflight_report,
        build_embeddings_for_splits, load_waveform, load_yamnet, extract_embedding,
    )
    from unified_evaluation import plot_noise_decay
except ModuleNotFoundError:
    pass  # 名称已在 notebook 全局命名空间中, 后续直接引用


SNR_TIERS = ["clean", "5dB", "0dB", "-5dB"]   # 绘图顺序

# 噪声嵌入缓存的各档数组名 (与 build_noise_embeddings 写入的 key 对应)
NOISE_EMB_KEYS = [("clean", "X_clean"), ("5dB", "X_5dB"),
                  ("0dB", "X_0dB"), ("-5dB", "X_n5dB")]

cfg = Config()


# ============================================================
# 1. 按指定 SNR 叠加高斯白噪声
# ============================================================
def add_gaussian_noise(waveform: np.ndarray, snr_db: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    waveform: 干净波形 (float32, 归一化到 [-1,1])
    snr_db:   目标信噪比 (dB), 越小噪声越强
    rng:      随机数生成器, 传入以保证可复现
    返回叠噪后重新归一化到 [-1,1] 的波形。
    """
    signal_power = np.mean(waveform ** 2) + 1e-12           # 信号平均功率
    noise_power = signal_power / (10 ** (snr_db / 10))      # 由 SNR 反推噪声功率
    noise = rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    noisy = waveform + noise
    # 叠噪后幅值可能越界, 按峰值重新归一化, 与 load_waveform 一致
    peak = np.max(np.abs(noisy)) + 1e-9
    return (noisy / peak).astype(np.float32)


# ============================================================
# 2. 5 折模式: 算一次噪声嵌入, 各折模型复用预测
# ============================================================
def build_noise_embeddings(df_train, df_val, df_test, cfg_obj=cfg):
    """
    计算测试集 3 档噪声嵌入并缓存 (5 折共享)。clean 档直接复用训练缓存嵌入,
    三个噪声档叠噪后重新经 YAMNet。fold 无关, 算一次即可被各折模型复用。

    df_train/df_val 仅用于命中同一 embedding 缓存以取出 X_test (可用任意折)。
    返回 dict: X_clean, X_5dB, X_0dB, X_n5dB, y_true, test_filenames。
    """
    cache_path = cfg_obj.NOISE_EMBED_CACHE
    if cache_path.exists():
        print(f"[噪声嵌入] 读已有缓存: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    label_map = json.loads(cfg_obj.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    label2idx = label_map["label2idx"]

    # clean 档: 命中训练缓存, 不重算 YAMNet
    _, _, _, _, X_test, y_test, test_filenames = \
        build_embeddings_for_splits(df_train, df_val, df_test, label2idx)
    print(f"[噪声嵌入] 测试集 {len(df_test)} 条, clean 档复用缓存")

    rng = np.random.default_rng(cfg_obj.SEED)          # 固定种子, 5 折噪声一致
    yamnet = load_yamnet()
    buckets = {"-5dB": [], "0dB": [], "5dB": []}
    snr_vals = [(-5.0, "-5dB"), (0.0, "0dB"), (5.0, "5dB")]
    n = len(df_test)
    for i, (_, row) in enumerate(df_test.iterrows()):
        clean_wf = load_waveform(row["filepath"])
        for snr_val, key in snr_vals:
            noisy_wf = add_gaussian_noise(clean_wf, snr_val, rng)
            buckets[key].append(extract_embedding(yamnet, noisy_wf))
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [噪声嵌入] 已处理 {i+1}/{n}")

    data = {
        "X_clean": X_test,
        "X_5dB": np.stack(buckets["5dB"]).astype(np.float32),
        "X_0dB": np.stack(buckets["0dB"]).astype(np.float32),
        "X_n5dB": np.stack(buckets["-5dB"]).astype(np.float32),
        "y_true": y_test,
        "test_filenames": np.array(test_filenames),
    }
    np.savez(cache_path, **data)
    print(f"[噪声嵌入] 已缓存: {cache_path}")
    return data


def eval_noise_for_model(model_path, noise_emb):
    """对单个模型在 4 档嵌入上预测, 返回 (acc_by_snr, preds_by_snr)。"""
    model = tf.keras.models.load_model(model_path)
    acc_by_snr = {}
    preds_by_snr = {}
    y_true = noise_emb["y_true"]
    for snr_key, arr_key in NOISE_EMB_KEYS:
        preds = np.argmax(model.predict(noise_emb[arr_key], verbose=0), axis=1)
        preds_by_snr[snr_key] = preds
        acc_by_snr[snr_key] = float(np.mean(preds == y_true))
    return acc_by_snr, preds_by_snr


def main_cv(n_folds: int = 5):
    """
    5 折噪声评估: 算一次噪声嵌入, 各折模型复用预测, 汇总 mean±std 并出衰减曲线。

    需先在 cell1 运行 main_cv_all_folds() 生成各折模型 (fold{N}/yamnet_bird_model.keras)。
    """
    # 取 df_test (test 折无关; 用 fold1 默认 CSV 命中同一缓存)
    df_train, df_val, df_test, missing = load_csv_splits(cfg)
    df_train, df_val, df_test = preflight_report(missing, df_train, df_val, df_test)
    if len(df_test) == 0:
        raise RuntimeError("测试集为空, 无法进行噪声实验。")

    noise_emb = build_noise_embeddings(df_train, df_val, df_test, cfg)

    rows = []
    for fold in range(1, n_folds + 1):
        model_path = cfg.fold_dir(fold) / "yamnet_bird_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(
                f"fold{fold} 模型不存在: {model_path}; 请先在 cell1 跑 main_cv_all_folds。")
        acc_by_snr, preds_by_snr = eval_noise_for_model(model_path, noise_emb)
        print(f"[噪声] fold{fold}: " + "  ".join(f"{k}={v:.4f}" for k, v in acc_by_snr.items()))

        np.savez(cfg.fold_dir(fold) / "noise_results.npz",
                 snr_tiers=np.array(SNR_TIERS),
                 acc=np.array([acc_by_snr[s] for s in SNR_TIERS]),
                 y_true=noise_emb["y_true"],
                 preds_clean=preds_by_snr["clean"],
                 preds_5dB=preds_by_snr["5dB"],
                 preds_0dB=preds_by_snr["0dB"],
                 preds_n5dB=preds_by_snr["-5dB"])
        rows.append({"fold": fold,
                     **{f"acc_{k}": acc_by_snr[k] for k in SNR_TIERS}})

    # 汇总 mean±std
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(cfg.OUT_DIR / "cv_noise_per_fold.csv", index=False)

    noise_metrics = [f"acc_{k}" for k in SNR_TIERS]
    summary_path = cfg.OUT_DIR / "cv_summary.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() \
        else pd.DataFrame(columns=["metric", "mean", "std"])
    new_rows = pd.DataFrame([{
        "metric": m,
        "mean": float(per_fold[m].mean()),
        "std": float(per_fold[m].std(ddof=0)),
    } for m in noise_metrics])
    pd.concat([existing, new_rows], ignore_index=True).to_csv(summary_path, index=False)

    print(f"\n[CV 噪声] 汇总 (mean±std):")
    for m in noise_metrics:
        print(f"  {m}: {per_fold[m].mean():.4f} ± {per_fold[m].std(ddof=0):.4f}")

    # 出图: 用各档均值画衰减曲线
    col_for = {k: f"acc_{k}" for k in SNR_TIERS}
    mean_accs = {k: float(per_fold[col_for[k]].mean()) for k in SNR_TIERS}
    plot_noise_decay({"YAMNet_mean": mean_accs})
    print(f"[CV 噪声] 汇总已存: {summary_path}; 衰减曲线已出。")


# ============================================================
# 3. 单折模式: 直接读根目录模型评估 (兼容旧跑法)
# ============================================================
def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.SEED)          # 固定种子, 噪声可复现

    # 读取 ml_test.csv (与干净基线同一批样本); 同时读取 train/val 以命中同一缓存
    df_train, df_val, df_test, missing = load_csv_splits(cfg)
    df_train, df_val, df_test = preflight_report(missing, df_train, df_val, df_test)
    if len(df_test) == 0:
        raise RuntimeError("测试集为空, 无法进行噪声实验。")

    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    label2idx = label_map["label2idx"]

    # 命中训练时的缓存, clean 档直接复用
    X_tr, y_tr, X_va, y_va, X_test, y_test, test_filenames = \
        build_embeddings_for_splits(df_train, df_val, df_test, label2idx)
    print(f"[噪声测试] 测试集 {len(df_test)} 条 (与干净基线同一批样本)")

    # 加载训练好的模型与 YAMNet (噪声档需重新提取嵌入)
    model = tf.keras.models.load_model(cfg.MODEL_PATH)
    print(f"[噪声测试] 已加载模型: {cfg.MODEL_PATH}")
    yamnet = load_yamnet()

    # 对每条测试音频分别在 clean 与三个噪声档下预测; clean 复用缓存嵌入
    pred_by_snr = {snr: [] for snr in SNR_TIERS}
    y_true = y_test

    # df_test 行顺序与 test_filenames / X_test 一致 (由 build_embeddings_for_splits 保证)
    for i, (_, row) in enumerate(df_test.iterrows()):
        filepath = row["filepath"]

        # clean: 复用缓存嵌入
        clean_emb = X_test[i]
        pred_by_snr["clean"].append(
            int(np.argmax(model.predict(clean_emb[None, :], verbose=0), axis=1)[0]))

        # 三档噪声: 叠噪 -> 重新经 YAMNet -> 预测
        clean_wf = load_waveform(filepath)
        for snr_val, snr_key in [(-5.0, "-5dB"), (0.0, "0dB"), (5.0, "5dB")]:
            noisy_wf = add_gaussian_noise(clean_wf, snr_val, rng)
            noisy_emb = extract_embedding(yamnet, noisy_wf)
            pred_by_snr[snr_key].append(
                int(np.argmax(model.predict(noisy_emb[None, :], verbose=0), axis=1)[0]))

        if (i + 1) % 10 == 0 or (i + 1) == len(df_test):
            print(f"  已处理 {i+1}/{len(df_test)}")

    # 统计各 SNR 档准确率
    acc_by_snr = {}
    for snr in SNR_TIERS:
        acc = float(np.mean(np.array(pred_by_snr[snr]) == y_true))
        acc_by_snr[snr] = acc
        print(f"  [{snr}] accuracy = {acc:.4f}")

    # 保存结果并绘制衰减曲线
    out_npz = cfg.OUT_DIR / "noise_results.npz"
    np.savez(out_npz,
             snr_tiers=np.array(SNR_TIERS),
             acc=np.array([acc_by_snr[s] for s in SNR_TIERS]),
             y_true=y_true,
             preds_clean=np.array(pred_by_snr["clean"]),
             preds_5dB=np.array(pred_by_snr["5dB"]),
             preds_0dB=np.array(pred_by_snr["0dB"]),
             preds_n5dB=np.array(pred_by_snr["-5dB"]))
    print(f"[噪声测试] 结果已存: {out_npz}")

    plot_noise_decay({"YAMNet": acc_by_snr})
    print("[噪声测试] 完成。下一步: 将 LightGBM / FastAI 的同款结果填入 plot_noise_decay 一并对比。")


if __name__ == "__main__":
    main_cv()
