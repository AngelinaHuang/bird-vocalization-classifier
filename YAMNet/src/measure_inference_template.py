"""
推理速度与显存测量 — 通用模板 (LightGBM / FastAI 适用)
=====================================================

使用说明:
  1. 替换 load_model() 中的模型加载逻辑
  2. 替换 predict_one() 中的单条预测逻辑 (从原始音频到预测结果)
  3. 在 Kaggle notebook 全流程跑完后, 作为最后一个 cell 运行本脚本
  4. 结果写入可下载的 CSV 文件

噪声实验规范提醒:
  噪声类型: 高斯白噪声
  SNR 公式: P_noise = mean(waveform²) / (10^(SNR/10))
  叠噪后峰值归一化到 [-1, 1]
  测试集: ml_test.csv (1196 条)
  随机种子: 固定 (如 42)
  测试档位: clean / 5dB / 0dB / -5dB
"""

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 配置 (按你的模型修改)
# ============================================================
MODEL_NAME = "LightGBM"          # 或 "FastAI"
OUT_DIR = Path("/kaggle/working")  # 输出目录
TEST_CSV = "/kaggle/input/bird-metadata/ml_test.csv"  # 测试集 CSV 路径
N_SAMPLES = 50   # 测量样本数
N_WARMUP = 5     # 预热次数


# ============================================================
# 1. 加载模型 (按你的模型修改这部分)
# ============================================================
def load_model():
    """
    返回你训练好的模型对象。
    示例:
      LightGBM:  import lightgbm; return lgb.Booster(model_file='model.txt')
      FastAI:    from fastai.vision.all import load_learner; return load_learner('export.pkl')
    """
    # TODO: 替换为你的模型加载代码
    raise NotImplementedError("请替换为你的模型加载代码")


# ============================================================
# 2. 加载一条测试音频并预处理 (按你的模型修改)
# ============================================================
def load_and_preprocess(filepath: str):
    """
    输入: 音频文件路径
    返回: 你的模型所需的输入格式 (如特征向量 / mel-spectrogram 张量等)

    注意: 预处理流程必须与训练时完全一致。

    示例:
      LightGBM:  waveform → 提取声学特征 → 返回特征向量
      FastAI:    waveform → 生成 mel-spectrogram 图像 → 返回图像张量
    """
    # TODO: 替换为你的预处理代码
    raise NotImplementedError("请替换为你的预处理代码")


# ============================================================
# 3. 单条预测 (按你的模型修改)
# ============================================================
def predict_one(model, input_data) -> int:
    """
    输入: 模型 + 预处理后的输入
    返回: 预测的类别索引 (整数)

    示例:
      LightGBM:  return int(model.predict(input_data.reshape(1, -1))[0])
      FastAI:    return int(model.predict(input_data)[0])
    """
    # TODO: 替换为你的预测代码
    raise NotImplementedError("请替换为你的预测代码")


# ============================================================
# 4. 测量主流程 (一般不需要改)
# ============================================================
def measure():
    print(f"[推理测量] 模型: {MODEL_NAME}")

    # 加载模型
    model = load_model()
    print(f"[推理测量] 模型已加载")

    # 读取测试集
    df_test = pd.read_csv(TEST_CSV)
    # 你需要根据你的数据加载方式, 给每行添加 filepath 列
    # 如果 filepath 列不存在, 需要自行实现 resolve_audio_path 逻辑
    if "filepath" not in df_test.columns:
        print("[推理测量] 警告: 测试集缺少 filepath 列, 请自行补充音频路径解析")
        print("  提示: 音频路径需要解析为 Kaggle 挂载路径, 如 /kaggle/input/birdclef-2021/...")
    n_test = min(N_SAMPLES, len(df_test))
    print(f"[推理测量] 测试集 {len(df_test)} 条, 取前 {n_test} 条测量")

    # 预热
    print(f"[推理测量] 预热 {N_WARMUP} 次 ...")
    sample_path = df_test["filepath"].iloc[0]
    sample_input = load_and_preprocess(sample_path)
    for i in range(N_WARMUP):
        predict_one(model, sample_input)
    print(f"[推理测量] 预热完成")

    # 正式测量
    print(f"[推理测量] 正式测量 {n_test} 条 ...")
    times = []
    for i in range(n_test):
        filepath = df_test["filepath"].iloc[i]
        inp = load_and_preprocess(filepath)
        t0 = time.perf_counter()
        predict_one(model, inp)
        times.append((time.perf_counter() - t0) * 1000)  # 转毫秒

        if (i + 1) % 10 == 0 or (i + 1) == n_test:
            print(f"  {i+1}/{n_test}")

    times_arr = np.array(times)
    print(f"\n{'='*50}")
    print(f"[推理速度] {MODEL_NAME} 端到端: {times_arr.mean():.1f} ± {times_arr.std():.1f} ms/条")
    print(f"{'='*50}")

    # 保存结果
    metrics = {
        "model": MODEL_NAME,
        "inference_mean_ms": round(float(times_arr.mean()), 1),
        "inference_std_ms": round(float(times_arr.std()), 1),
        "n_samples": n_test,
        "n_warmup": N_WARMUP,
    }
    out_csv = OUT_DIR / f"inference_metrics_{MODEL_NAME}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"[推理测量] 结果已存: {out_csv}")

    return metrics


if __name__ == "__main__":
    measure()