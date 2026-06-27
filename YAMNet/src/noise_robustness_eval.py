"""
噪声鲁棒性测试 —— 本作业的核心创新点
负责人: Wenjuan Huang

大白话:
  我们想知道: 如果录音里有背景噪声(风声、雨声、环境杂音), 模型还能认对鸟吗?
  做法: 拿测试集的干净音频, 人为叠加不同强度的噪声, 再让训练好的模型认一遍,
        看准确率怎么掉。噪声越强, 掉得越少的模型 = 越抗噪。

  噪声强度用 SNR (信噪比, dB) 衡量:
    - 5 dB : 噪声较弱 (鸟叫比较清楚)
    - 0 dB : 噪声和鸟叫一样响
    - -5dB: 噪声比鸟叫还响 (很吵, 模型最难)
  外加一个 clean (不叠噪声) 作为基线, 对应你训练时报告的准确率。

输出: 各 SNR 下的准确率, 喂给 unified_evaluation.plot_noise_decay 画衰减曲线。

注意:
  - 噪声是叠在"原始波形"上的, 叠完要重新过 YAMNet 提 embedding, 不能复用干净缓存。
  - 这里用高斯白噪声做可控基准 (最标准、可复现)。proposal 里若要更真实, 后续可换成
    真实风/雨噪声片段, 接口不变, 只改 add_noise 函数。
  - 样本少时这条曲线也会有抖动, 等 Jianan 全量数据出来再跑才准。现在跑主要是验证流程。
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# 复用 YAMNet 管道里写好的函数, 不重复造轮子
from yamnet_bird_pipeline import (
    Config, build_dataframe_from_folders, build_or_load_embeddings,
    load_waveform, load_yamnet, extract_embedding,
)
from unified_evaluation import plot_noise_decay


SNR_TIERS = ["clean", "5dB", "0dB", "-5dB"]   # 画图时的顺序


# ============================================================
# 1. 按指定 SNR 给波形叠高斯白噪声
# ============================================================
def add_gaussian_noise(waveform: np.ndarray, snr_db: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    waveform: 干净波形 (float32, 已归一化到 [-1,1])
    snr_db:   目标信噪比(dB)。越小噪声越强。
    rng:      随机数生成器 (传入是为了可复现)。
    返回: 叠噪后重新归一化到 [-1,1] 的波形。
    """
    signal_power = np.mean(waveform ** 2) + 1e-12           # 信号平均功率
    noise_power = signal_power / (10 ** (snr_db / 10))      # 由 SNR 反推噪声功率
    noise = rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    noisy = waveform + noise
    # 叠噪后幅值可能超出 [-1,1], 按 peak 重新归一化, 和 load_waveform 保持一致
    peak = np.max(np.abs(noisy)) + 1e-9
    return (noisy / peak).astype(np.float32)


# ============================================================
# 2. 复现训练时的测试集 (用同样的种子+分层切分, 保证和 clean 基线是同一批样本)
# ============================================================
def reproduce_test_split(df, X, y):
    """
    训练时用的是固定 seed 的分层切分, 这里照搬一遍, 拿到 test 集在 df 里的下标。
    返回: test_indices (对应 df 的行号), 这样能查到每条测试音频的文件路径。
    """
    from sklearn.model_selection import train_test_split
    cfg = Config()
    idx_all = np.arange(len(df))
    idx_tmp, idx_test, _, _ = train_test_split(
        idx_all, y, test_size=cfg.TEST_SPLIT, random_state=cfg.SEED, stratify=y)
    # 注意: 训练时切的是 X/y, 这里切的是下标数组, 但顺序由 y(stratify) 决定, 结果一致。
    return idx_test


# ============================================================
# 3. 主流程: 对测试集每个样本, 在各 SNR 下重新预测, 算准确率
# ============================================================
def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.SEED)          # 固定种子, 保证噪声可复现

    # 3.1 重建数据表 + 读缓存 embedding (只为了复现切分, 不重算 embedding)
    df = build_dataframe_from_folders(cfg.RAW_DATA_DIR)
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    label2idx = label_map["label2idx"]
    X, y = build_or_load_embeddings(df, label2idx)  # 命中缓存, 秒读

    # 3.2 复现测试集下标
    test_idx = reproduce_test_split(df, X, y)
    print(f"[噪声测试] 测试集 {len(test_idx)} 条")

    # 3.3 加载训练好的模型 + YAMNet
    model = tf.keras.models.load_model(cfg.MODEL_PATH)
    print(f"[噪声测试] 已加载模型: {cfg.MODEL_PATH}")
    yamnet = load_yamnet()

    # 3.4 对每条测试音频: 干净 + 三个噪声档, 分别预测
    #     pred_by_snr[snr] = 该档下所有测试样本的预测下标
    pred_by_snr = {snr: [] for snr in SNR_TIERS}
    y_true = y[test_idx]

    for i, di in enumerate(test_idx):
        filepath = df.iloc[di]["filepath"]
        clean_wf = load_waveform(filepath)          # 干净波形

        # clean: 直接用缓存里这条的 embedding (和训练时一致), 省一次 YAMNet
        clean_emb = X[di]
        pred_by_snr["clean"].append(
            int(np.argmax(model.predict(clean_emb[None, :], verbose=0), axis=1)[0]))

        # 三个噪声档: 叠噪 -> 重新过 YAMNet -> 预测
        for snr_val, snr_key in [(-5.0, "-5dB"), (0.0, "0dB"), (5.0, "5dB")]:
            noisy_wf = add_gaussian_noise(clean_wf, snr_val, rng)
            noisy_emb = extract_embedding(yamnet, noisy_wf)
            pred_by_snr[snr_key].append(
                int(np.argmax(model.predict(noisy_emb[None, :], verbose=0), axis=1)[0]))

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(test_idx)}")

    # 3.5 算每个 SNR 档的准确率
    acc_by_snr = {}
    for snr in SNR_TIERS:
        acc = float(np.mean(np.array(pred_by_snr[snr]) == y_true))
        acc_by_snr[snr] = acc
        print(f"  [{snr}] accuracy = {acc:.4f}")

    # 3.6 存结果 + 画衰减曲线
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
    print("[噪声测试] 完成。下一步: 把 LightGBM / FastAI 的同款结果也填进 plot_noise_decay 一起比。")


if __name__ == "__main__":
    main()
