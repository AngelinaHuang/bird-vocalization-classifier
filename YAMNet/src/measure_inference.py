"""
推理速度与显存测量
负责人: Wenjuan Huang

在 Kaggle notebook 全流程跑完后, 作为最后一个 cell 运行本脚本,
测量 YAMNet 的端到端推理延迟与 GPU 显存占用。

测量策略:
  - YAMNet 编码器在各折间共享 (冻结参数), 测一次即可
  - 分类头各折权重不同, 逐折加载测量, 取 5 折 mean±std
  - 端到端 = YAMNet 编码器 + 分类头均值

用法:
  - 在 notebook 中按顺序执行完 yamnet_bird_pipeline.py (cell1) 和
    noise_robustness_eval.py (cell3) 后, 在 cell4 中运行:
        %run -i src/measure_inference.py
    或直接粘贴本文件内容到新 cell 运行。
  - 参数: 默认测 50 条测试样本, 可在 main() 中调整 N_SAMPLES。
  - 结果写入 /kaggle/working/yamnet/inference_metrics.csv, 可随 notebook 下载。

输出:
  - 端到端推理延迟 (ms/条): 原始波形 → YAMNet 嵌入 → 分类头预测 (5 折 mean±std)
  - YAMNet 编码器延迟 (ms/条): 共享组件, 测一次
  - 分类头延迟 (ms/条): 各折分别测, 5 折 mean±std
  - GPU 显存占用 (MB, 当前与峰值)
  - 逐折细节写入 inference_details.csv
"""

import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# 复用管道函数 (notebook 中已执行过 yamnet_bird_pipeline.py, 名称在全局命名空间;
# 若以 .py 直接运行则 import)
try:
    from yamnet_bird_pipeline import (
        Config, load_csv_splits, preflight_report,
        load_waveform,
    )
except (ModuleNotFoundError, ImportError):
    pass  # notebook 模式下名称已存在

cfg = Config()
OUT_DIR = cfg.OUT_DIR  # /kaggle/working/yamnet


def measure_inference(n_samples: int = 50, n_warmup: int = 5, n_folds: int = 5):
    """
    测量 5 折推理延迟与显存。

    n_samples: 测试样本数 (从测试集中取前 N 条)
    n_warmup:  预热次数 (不计入统计)
    n_folds:   折数 (默认 5)
    """
    # ---- 加载测试集 ----
    df_train, df_val, df_test, missing = load_csv_splits(cfg)
    df_train, df_val, df_test = preflight_report(missing, df_train, df_val, df_test)
    if len(df_test) == 0:
        raise RuntimeError("测试集为空, 无法测量。")
    n_test = min(n_samples, len(df_test))
    print(f"[推理测量] 测试集 {len(df_test)} 条, 取前 {n_test} 条测量, {n_folds} 折")

    # ---- 加载 YAMNet 编码器 (共享, 只测一次) ----
    yamnet = hub.load(cfg.YAMNET_HANDLE)
    print("[推理测量] YAMNet 编码器已加载")

    # ---- 预提取所有测试样本的波形 (避免 I/O 影响计时) ----
    print(f"[推理测量] 预加载 {n_test} 条波形 ...")
    waveforms = []
    for i in range(n_test):
        waveforms.append(load_waveform(df_test["filepath"].iloc[i]))
        if (i + 1) % 10 == 0 or (i + 1) == n_test:
            print(f"  预加载 {i+1}/{n_test}")

    # ================================================================
    # 1. 测量 YAMNet 编码器延迟 (共享组件, 各折相同)
    # ================================================================
    print(f"\n[推理测量] 测量 YAMNet 编码器 ({n_warmup} 预热 + {n_test} 正式) ...")
    for i in range(n_warmup):
        wav_tf = tf.convert_to_tensor(waveforms[0], dtype=tf.float32)
        _, _, _ = yamnet(wav_tf)

    yamnet_times = []
    for i in range(n_test):
        wav_tf = tf.convert_to_tensor(waveforms[i], dtype=tf.float32)
        t0 = time.perf_counter()
        _, embeddings, _ = yamnet(wav_tf)
        emb = tf.reduce_mean(embeddings, axis=0).numpy()
        yamnet_times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 10 == 0 or (i + 1) == n_test:
            print(f"  YAMNet 编码器 {i+1}/{n_test}")

    yamnet_arr = np.array(yamnet_times)
    print(f"  YAMNet 编码器: {yamnet_arr.mean():.1f} ± {yamnet_arr.std():.1f} ms/条")

    # ================================================================
    # 2. 逐折测量分类头延迟
    # ================================================================
    # 预计算所有嵌入 (各折分类头输入相同, 缓存避免重复 YAMNet)
    print(f"\n[推理测量] 预计算 {n_test} 条嵌入 ...")
    cached_embs = []
    for i in range(n_test):
        wav_tf = tf.convert_to_tensor(waveforms[i], dtype=tf.float32)
        _, emb_tf, _ = yamnet(wav_tf)
        cached_embs.append(tf.reduce_mean(emb_tf, axis=0).numpy())
        if (i + 1) % 10 == 0 or (i + 1) == n_test:
            print(f"  预计算嵌入 {i+1}/{n_test}")

    fold_head_means = []
    fold_head_stds = []
    fold_e2e_means = []

    for fold in range(1, n_folds + 1):
        model_path = cfg.fold_dir(fold) / "yamnet_bird_model.keras"
        if not model_path.exists():
            # 兼容旧版单折产物
            model_path = cfg.OUT_DIR / "yamnet_bird_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(
                f"fold{fold} 模型不存在: {model_path}。请先运行 yamnet_bird_pipeline.py。")

        model = tf.keras.models.load_model(model_path)
        print(f"\n[推理测量] fold{fold} 分类头测量 ({n_warmup} 预热 + {n_test} 正式) ...")

        # 预热
        for _ in range(n_warmup):
            _ = np.argmax(model.predict(cached_embs[0][None, :], verbose=0), axis=1)[0]

        # 正式测量
        head_times = []
        for i in range(n_test):
            t0 = time.perf_counter()
            _ = np.argmax(model.predict(cached_embs[i][None, :], verbose=0), axis=1)[0]
            head_times.append((time.perf_counter() - t0) * 1000)

        head_arr = np.array(head_times)
        fold_head_means.append(head_arr.mean())
        fold_head_stds.append(head_arr.std())
        fold_e2e_means.append(yamnet_arr.mean() + head_arr.mean())
        print(f"  fold{fold} 分类头: {head_arr.mean():.1f} ± {head_arr.std():.1f} ms/条")
        print(f"  fold{fold} 端到端: {yamnet_arr.mean() + head_arr.mean():.1f} ms/条")

    # ================================================================
    # 3. 汇总 5 折
    # ================================================================
    head_means_arr = np.array(fold_head_means)
    e2e_means_arr = np.array(fold_e2e_means)

    print(f"\n{'='*60}")
    print(f"[推理速度] 5 折汇总 (mean ± std):")
    print(f"  端到端 (原始音频→预测): {e2e_means_arr.mean():.1f} ± {e2e_means_arr.std():.1f} ms/条")
    print(f"    其中 YAMNet 编码器:    {yamnet_arr.mean():.1f} ± {yamnet_arr.std():.1f} ms/条 (共享)")
    print(f"    其中 分类头:          {head_means_arr.mean():.1f} ± {head_means_arr.std():.1f} ms/条 (5 折)")
    print(f"{'='*60}")

    # ================================================================
    # 4. 显存测量
    # ================================================================
    gpu_current_mb = None
    gpu_peak_mb = None
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info = tf.config.experimental.get_memory_info("GPU:0")
            gpu_current_mb = info.get("current", 0) / (1024 ** 2)
            gpu_peak_mb = info.get("peak", 0) / (1024 ** 2)
            print(f"\n[显存] GPU 当前: {gpu_current_mb:.1f} MB, 峰值: {gpu_peak_mb:.1f} MB")
        else:
            print("\n[显存] 未检测到 GPU, 使用 CPU 推理")
    except Exception as e:
        print(f"\n[显存] 无法获取: {e}")

    # ================================================================
    # 5. 保存结果
    # ================================================================
    metrics = {
        "model": "YAMNet",
        "inference_e2e_mean_ms": round(float(e2e_means_arr.mean()), 1),
        "inference_e2e_std_ms": round(float(e2e_means_arr.std()), 1),
        "yamnet_encoder_mean_ms": round(float(yamnet_arr.mean()), 1),
        "yamnet_encoder_std_ms": round(float(yamnet_arr.std()), 1),
        "head_mean_ms": round(float(head_means_arr.mean()), 1),
        "head_std_ms": round(float(head_means_arr.std()), 1),
        "gpu_memory_current_mb": round(gpu_current_mb, 1) if gpu_current_mb else None,
        "gpu_memory_peak_mb": round(gpu_peak_mb, 1) if gpu_peak_mb else None,
        "n_samples": n_test,
        "n_warmup": n_warmup,
        "n_folds": n_folds,
    }

    out_csv = OUT_DIR / "inference_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"\n[推理测量] 汇总已存: {out_csv}")

    # 逐折细节
    detail_csv = OUT_DIR / "inference_details.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "e2e_mean_ms", "head_mean_ms", "head_std_ms",
                         "yamnet_mean_ms", "yamnet_std_ms", "n_samples"])
        for fold in range(1, n_folds + 1):
            writer.writerow([
                fold,
                round(fold_e2e_means[fold - 1], 1),
                round(fold_head_means[fold - 1], 1),
                round(fold_head_stds[fold - 1], 1),
                round(float(yamnet_arr.mean()), 1),
                round(float(yamnet_arr.std()), 1),
                n_test,
            ])
    print(f"[推理测量] 逐折细节已存: {detail_csv}")

    return metrics


if __name__ == "__main__":
    measure_inference(n_samples=50)