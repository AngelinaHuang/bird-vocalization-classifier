"""
统一评估脚本 —— 三个模型(LightGBM / FastAI / YAMNet)都用这一套标准来比
负责人: Wenjuan Huang

设计原则: "裁判"对三个选手一视同仁。
  - 输入统一: 每个模型最终都给出 (y_true, y_pred) + 类别名
  - 指标统一: accuracy / precision / recall / F1 (macro & weighted)
  - 出图统一: 混淆矩阵、准确率柱状图、噪声下准确率衰减曲线
  - 性能统一: 单条推理延迟、GPU 显存占用

用法见文件末尾的 demo()。
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境下也能存图
import matplotlib.pyplot as plt

# 修复中文显示乱码: Windows 优先用 SimHei / Microsoft YaHei
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 "-" 显示为方块
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

OUT_DIR = Path("../outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 分类指标
# ============================================================
def compute_classification_metrics(y_true, y_pred, class_names, model_name="model"):
    """
    返回一个 dict: accuracy, macro_p/r/f1, weighted_p/r/f1, 每类也存一份。
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    # 每类指标 (长表, 方便后续分析哪些鸟容易认错)
    per_class = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    metrics["per_class"] = per_class
    print(f"[{model_name}] accuracy={metrics['accuracy']:.4f} "
          f"macro-F1={metrics['f1_macro']:.4f} weighted-F1={metrics['f1_weighted']:.4f}")
    return metrics


# ============================================================
# 2. 混淆矩阵
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, model_name="model",
                          save_path=None, top_n_classes=None):
    """
    画混淆矩阵。类别多时(86类)整张图会很大, top_n_classes 可只画样本最多的前 N 类。
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.25),
                                    max(6, len(class_names) * 0.25)))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"混淆矩阵 - {model_name}")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    path = save_path or (OUT_DIR / f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图] 混淆矩阵已存: {path}")
    return path


# ============================================================
# 3. 多模型准确率柱状图对比
# ============================================================
def plot_accuracy_bar(all_metrics, save_path=None):
    """
    all_metrics: list[dict], 每个是 compute_classification_metrics 的返回值。
    画 accuracy / macro-F1 / weighted-F1 三组柱。
    """
    df = pd.DataFrame(all_metrics)
    metrics_to_plot = ["accuracy", "f1_macro", "f1_weighted"]
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_to_plot):
        ax.bar(x + i * width, df[m], width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("分数")
    ax.set_title("三个模型总体对比")
    ax.legend()
    plt.tight_layout()
    path = save_path or (OUT_DIR / "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图] 准确率对比已存: {path}")
    return path


# ============================================================
# 4. 噪声鲁棒性衰减曲线 (本作业的核心创新点!)
# ============================================================
def plot_noise_decay(noise_results, save_path=None):
    """
    noise_results: dict[model_name] = {"clean": acc, "5dB": acc, "0dB": acc, "-5dB": acc}
    画每个模型在不同噪声强度下的准确率折线, 谁掉得少谁抗噪。
    """
    snr_order = ["clean", "5dB", "0dB", "-5dB"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, accs in noise_results.items():
        ys = [accs[k] for k in snr_order]
        ax.plot(snr_order, ys, marker="o", label=model_name)
    ax.set_xlabel("噪声强度 (越往右越吵)")
    ax.set_ylabel("准确率")
    ax.set_title("噪声鲁棒性衰减曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_path or (OUT_DIR / "noise_robustness.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图] 噪声衰减曲线已存: {path}")
    return path


# ============================================================
# 5. 性能开销: 推理延迟 + GPU 显存
# ============================================================
def measure_latency(predict_fn, sample_input, n_warmup=5, n_runs=50):
    """
    predict_fn: 输入一个样本, 返回预测 (类别下标或概率)。
    返回平均单条推理延迟 (毫秒)。
    """
    for _ in range(n_warmup):
        predict_fn(sample_input)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(sample_input)
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000
    return avg_ms


def measure_gpu_memory_mb():
    """若用了 GPU, 返回 TensorFlow 当前进程占用的显存(MB); 没有则返回 None。"""
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except NameError:
        return None
    if not gpus:
        return None
    # 注意: 真正精确的峰值显存最好用 nvidia-smi 在训练时另开终端看。
    # 这里给一个进程级近似。
    import tensorflow as tf
    info = tf.config.experimental.get_memory_info("GPU:0")
    return info.get("current", 0) / (1024 ** 2)


# ============================================================
# 6. demo: 演示怎么用 (拿 YAMNet 的测试预测来跑一遍)
# ============================================================
def demo():
    """
    演示: 读 YAMNet 跑出来的 test_predictions.npz, 出混淆矩阵和指标。
    等三个模型都跑完, 再把它们的 metrics 拼一起调 plot_accuracy_bar。
    """
    pred_path = Path("../outputs/yamnet/test_predictions.npz")
    if not pred_path.exists():
        print("先跑 yamnet_bird_pipeline.py 生成 test_predictions.npz, 再来跑这个 demo。")
        return
    data = np.load(pred_path, allow_pickle=True)
    y_true, y_pred, classes = data["y_true"], data["y_pred"], data["classes"]

    m = compute_classification_metrics(y_true, y_pred, classes, model_name="YAMNet")
    plot_confusion_matrix(y_true, y_pred, classes, model_name="YAMNet")

    # 噪声衰减曲线示例 (真实数据等噪声测试组跑完再填)
    noise_results = {
        "YAMNet": {"clean": m["accuracy"], "5dB": 0.0, "0dB": 0.0, "-5dB": 0.0},
        # "LightGBM": {...}, "FastAI": {...},
    }
    plot_noise_decay(noise_results)


if __name__ == "__main__":
    demo()
