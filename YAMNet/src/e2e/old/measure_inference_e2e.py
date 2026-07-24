"""
端到端模型推理速度与显存测量
负责人: Wenjuan Huang

与 measure_inference.py 的区别:
  - 旧版: 分别测 YAMNet 编码器 + 分类头, 然后相加
  - 本版: 直接测端到端 (波形 -> 完整 e2e 模型预测), 因为 YAMNet 顶层已解冻

模型加载: 用 build_e2e_model 重建架构 + load_weights (与 noise_eval_e2e 一致)。

Kaggle 运行: cell3 (在训练 cell1 + 噪声评估 cell2 之后)
  %run -i src/measure_inference_e2e.py
"""
import csv
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

try:
    from yamnet_bird_pipeline import Config, load_csv_splits, preflight_report, load_waveform
    from yamnet_finetune_e2e import FinetuneConfig as fcfg, build_e2e_model
except (ModuleNotFoundError, ImportError):
    pass

cfg = Config()
E2E_DIR = cfg.OUT_DIR / "e2e"


def _load_e2e_model(fold_dir, num_classes):
    """重建端到端模型并加载权重 (与 noise_eval_e2e.py 一致)。"""
    weights_path = fold_dir / "best_weights.weights.h5"
    if not weights_path.exists():
        keras_path = fold_dir / "yamnet_e2e_model.keras"
        if keras_path.exists():
            return tf.keras.models.load_model(str(keras_path))
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    model, _, _ = build_e2e_model(num_classes)
    model.load_weights(str(weights_path))
    return model


def measure_e2e_inference(n_samples=50, n_warmup=5, n_folds=5):
    """测量端到端模型推理延迟与显存。"""
    df_train, df_val, df_test, missing = load_csv_splits(cfg)
    df_train, df_val, df_test = preflight_report(missing, df_train, df_val, df_test)
    n_test = min(n_samples, len(df_test))
    print(f"[推理测量] e2e 模型, 测试集取前 {n_test} 条, {n_folds} 折")

    # 读取标签映射 (用于确定 num_classes)
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    num_classes = len(label_map["label2idx"])

    # 预加载波形
    print(f"[推理测量] 预加载 {n_test} 条波形 ...")
    waveforms = []
    for i in range(n_test):
        waveforms.append(load_waveform(df_test["filepath"].iloc[i]))

    fold_details = []
    for fold in range(1, n_folds + 1):
        fold_dir = E2E_DIR / f"fold{fold}"
        weights_path = fold_dir / "best_weights.weights.h5"
        keras_path = fold_dir / "yamnet_e2e_model.keras"
        if not weights_path.exists() and not keras_path.exists():
            print(f"[跳过] fold{fold} 模型不存在")
            continue

        model = _load_e2e_model(fold_dir, num_classes)
        print(f"\n[推理测量] fold{fold} ({n_warmup} 预热 + {n_test} 正式) ...")

        # 预热
        for _ in range(n_warmup):
            _ = np.argmax(model.predict(waveforms[0][None, :], verbose=0), axis=1)[0]

        # 正式测量 (逐条)
        times = []
        for i in range(n_test):
            t0 = time.perf_counter()
            _ = np.argmax(model.predict(waveforms[i][None, :], verbose=0), axis=1)[0]
            times.append((time.perf_counter() - t0) * 1000)

        arr = np.array(times)
        fold_details.append({
            "fold": fold,
            "mean_ms": round(float(arr.mean()), 1),
            "std_ms": round(float(arr.std()), 1),
            "min_ms": round(float(arr.min()), 1),
            "max_ms": round(float(arr.max()), 1),
        })
        print(f"  fold{fold}: {arr.mean():.1f} ± {arr.std():.1f} ms/条")

    if not fold_details:
        print("[推理测量] 无可用模型")
        return

    means = np.array([d["mean_ms"] for d in fold_details])
    print(f"\n{'='*60}")
    print(f"[e2e 推理速度] {len(fold_details)} 折汇总: {means.mean():.1f} ± {means.std():.1f} ms/条")
    print(f"[对比旧版] 86.0 ± 0.6 ms/条")
    print(f"{'='*60}")

    # 显存
    gpu_cur, gpu_peak = None, None
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info = tf.config.experimental.get_memory_info("GPU:0")
            gpu_cur = info.get("current", 0) / (1024**2)
            gpu_peak = info.get("peak", 0) / (1024**2)
            print(f"[显存] 当前: {gpu_cur:.1f} MB, 峰值: {gpu_peak:.1f} MB")
    except Exception as e:
        print(f"[显存] 无法获取: {e}")

    # 保存逐折细节
    details_csv = E2E_DIR / "inference_details.csv"
    with open(details_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "mean_ms", "std_ms", "min_ms", "max_ms"])
        w.writeheader()
        for d in fold_details:
            w.writerow(d)
    print(f"[推理测量] 逐折细节已存: {details_csv}")

    # 保存汇总
    metrics = {
        "model": "YAMNet_e2e",
        "inference_e2e_mean_ms": round(float(means.mean()), 1),
        "inference_e2e_std_ms": round(float(means.std()), 1),
        "gpu_memory_current_mb": round(gpu_cur, 1) if gpu_cur else None,
        "gpu_memory_peak_mb": round(gpu_peak, 1) if gpu_peak else None,
        "n_samples": n_test,
        "n_warmup": n_warmup,
        "n_folds": len(fold_details),
    }
    out_csv = E2E_DIR / "inference_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metrics.keys())
        w.writeheader()
        w.writerow(metrics)
    print(f"[推理测量] 汇总已存: {out_csv}")


if __name__ == "__main__":
    measure_e2e_inference()
