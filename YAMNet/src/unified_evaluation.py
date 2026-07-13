"""
统一评估脚本
负责人: Wenjuan Huang

为 LightGBM / FastAI / YAMNet 三个模型提供一致的评估口径:
  - 输入统一: 各模型最终给出 (y_true, y_pred) 与类别名
  - 指标统一: accuracy / precision / recall / F1 (macro 与 weighted)
  - 出图统一: 混淆矩阵、准确率柱状图、噪声下准确率衰减曲线
  - 性能统一: 单条推理延迟、GPU 显存占用

用法见文件末尾 demo()。
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境下存图
import matplotlib.pyplot as plt

# 中文显示: Windows 优先使用 SimHei / Microsoft YaHei
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 负号显示
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
    """计算 accuracy、macro 与 weighted 的 precision/recall/F1, 并附逐类指标。"""
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
    # 逐类指标 (长表), 便于分析易混类别
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
    绘制混淆矩阵。类别较多时整张图会很大且不可读, 可通过 top_n_classes 仅画
    真实标签中样本数最多的前 N 类; 预测落在前 N 类之外的样本计入末列"其它",
    以反映模型在这些类上的错误分布。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if top_n_classes is not None and top_n_classes < len(class_names):
        # 取真实标签样本数最多的前 N 类
        vals, counts = np.unique(y_true, return_counts=True)
        top = vals[np.argsort(-counts)[:top_n_classes]].tolist()
        row_names = [class_names[i] for i in top]
        # 仅保留真实标签落在前 N 类的样本
        mask = np.isin(y_true, top)
        yt = y_true[mask]
        yp = y_pred[mask]
        # 预测落在前 N 类之外的归为 -1, 作为末列"其它"
        yp = np.where(np.isin(yp, top), yp, -1)
        labels = top + [-1]
        col_names = row_names + ["其它"]
        annot = (top_n_classes <= 30)
    else:
        labels = list(range(len(class_names)))
        row_names = col_names = class_names
        annot = len(class_names) <= 30

    cm = confusion_matrix(yt if top_n_classes is not None else y_true,
                          yp if top_n_classes is not None else y_pred,
                          labels=labels)
    size = max(8, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(cm, annot=annot, fmt="d", cmap="Blues",
                xticklabels=col_names, yticklabels=row_names, ax=ax)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"混淆矩阵 - {model_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    path = save_path or (OUT_DIR / f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图] 混淆矩阵已存: {path}")
    return path


# ============================================================
# 3. 多模型准确率柱状图
# ============================================================
def plot_accuracy_bar(all_metrics, save_path=None):
    """all_metrics 为 compute_classification_metrics 返回值的列表; 绘制 accuracy、macro-F1、weighted-F1 三组柱状图。"""
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
# 4. 噪声鲁棒性衰减曲线
# ============================================================
def plot_noise_decay(noise_results, save_path=None):
    """
    noise_results: dict[模型名] = {"clean": acc, "5dB": acc, "0dB": acc, "-5dB": acc}
    绘制各模型在不同噪声强度下的准确率折线; 下降越缓者鲁棒性越好。
    """
    snr_order = ["clean", "5dB", "0dB", "-5dB"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, accs in noise_results.items():
        ys = [accs[k] for k in snr_order]
        ax.plot(snr_order, ys, marker="o", label=model_name)
    ax.set_xlabel("噪声强度 (向右增强)")
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
# 5. 性能开销: 推理延迟与 GPU 显存
# ============================================================
def measure_latency(predict_fn, sample_input, n_warmup=5, n_runs=50):
    """predict_fn 输入一个样本返回预测; 返回平均单条推理延迟 (毫秒)。"""
    for _ in range(n_warmup):
        predict_fn(sample_input)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(sample_input)
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000
    return avg_ms


def measure_gpu_memory_mb():
    """若使用 GPU, 返回 TensorFlow 当前进程显存占用 (MB); 否则返回 None。"""
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except NameError:
        return None
    if not gpus:
        return None
    # 进程级近似; 精确峰值显存建议训练时用 nvidia-smi 查看
    import tensorflow as tf
    info = tf.config.experimental.get_memory_info("GPU:0")
    return info.get("current", 0) / (1024 ** 2)


# ============================================================
# 6. demo: 以 YAMNet 测试预测为例
# ============================================================
def demo():
    """读取 YAMNet 生成的 test_predictions.npz, 输出混淆矩阵与指标。三个模型均完成后, 可将各自 metrics 合并调用 plot_accuracy_bar。"""
    pred_path = Path("../outputs/yamnet/test_predictions.npz")
    if not pred_path.exists():
        print("请先运行 yamnet_bird_pipeline.py 生成 test_predictions.npz。")
        return
    data = np.load(pred_path, allow_pickle=True)
    y_true, y_pred, classes = data["y_true"], data["y_pred"], data["classes"]

    m = compute_classification_metrics(y_true, y_pred, classes, model_name="YAMNet")
    # 类别较多时整张混淆矩阵不可读, 仅画样本最多的前 30 类
    plot_confusion_matrix(y_true, y_pred, classes, model_name="YAMNet", top_n_classes=30)

    # 噪声衰减曲线示例 (真实数据待噪声测试完成后填入)
    noise_results = {
        "YAMNet": {"clean": m["accuracy"], "5dB": 0.0, "0dB": 0.0, "-5dB": 0.0},
        # "LightGBM": {...}, "FastAI": {...},
    }
    plot_noise_decay(noise_results)


if __name__ == "__main__":
    demo()
