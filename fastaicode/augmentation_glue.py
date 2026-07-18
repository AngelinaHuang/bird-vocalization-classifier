"""
FastAI 低资源物种音频增强 - 接入胶水代码
适用: fastaicode/fastaikaggle.txt (main_cv 5 折循环)

设计原理: FastAI 用 mel 频谱图 PNG + ResNet34, 不是波形。
本模块复用 audio_augmentation 的底层:
  - expand_train_df()  : 决定哪些低资源物种、哪些原始样本、各生成几条增强
  - augment_waveform() : 对单条波形施加一次随机增强 (4 种方法共用)
然后针对 FastAI 加一步: 增强波形 -> mel 频谱图 -> 存 PNG,
把增强 PNG 路径加进 df_train, ResNet 直接当训练样本。

接入方式 (在 main_cv 的 5 折循环里, audio_to_spectrogram 处理完
df_train 之后、构建 DataBlock 之前调用):
    from augmentation_glue import augment_for_fastai
    df_train_aug = augment_for_fastai(
        df_train, resolve_audio_path, sr=Config.SAMPLE_RATE,
        clip_seconds=Config.CLIP_SECONDS, n_mels=Config.N_MELS,
        img_dir=Config.IMG_DIR, target_per_species=15, seed=42 + fold)
    df_fold = pd.concat([df_train_aug, df_val], ignore_index=True)

依赖: audio_augmentation.py 需在 Kaggle 的 src/ 目录 (与 YAMNet 共用同一份)。
"""

import os
import gc
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from audio_augmentation import expand_train_df, augment_waveform


def _waveform_to_spectrogram_png(y, sr, clip_seconds, n_mels, save_path):
    """把一维波形写成 mel 频谱图 PNG (与 fastaikaggle 的 audio_to_spectrogram 同流程)。

    返回 save_path 字符串; 出错返回 None。
    """
    max_samples = int(sr * clip_seconds)
    if len(y) > max_samples:
        start = (len(y) - max_samples) // 2
        y = y[start:start + max_samples]
    else:
        y = np.pad(y, (0, max_samples - len(y)), mode="constant")

    try:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig = plt.figure(figsize=(3, 3), dpi=100)
        plt.imshow(mel_db, origin="lower", cmap="viridis")
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        fig.clf()
        plt.close("all")
        gc.collect()
        return str(save_path)
    except Exception:
        return None


def augment_for_fastai(df_train, resolve_audio_fn, sr=16000,
                       clip_seconds=5.0, n_mels=128, img_dir=None,
                       target_per_species=15, max_aug_per_species=50, seed=42):
    """
    为 FastAI 一折的训练集生成增强样本 (频谱图 PNG)。

    参数
    ----
    df_train : pd.DataFrame
        当前折训练集 (含 filename, primary_label 等)。
        要求: 调用前 df_train 的原始行应已由 audio_to_spectrogram
        处理过, 即已有 spectrogram_path 列; 原始行保持不变。
    resolve_audio_fn : callable(row) -> str|None
        fastaikaggle 里定义的 resolve_audio_path, 定位物理音频。
    sr : int
        采样率 (FastAI 用 16000)。
    clip_seconds : float
        统一音频长度秒数 (Config.CLIP_SECONDS, 默认 5.0)。
    n_mels : int
        mel 滤波器组数 (Config.N_MELS, 默认 128)。
    img_dir : Path|str
        频谱图 PNG 输出目录 (Config.IMG_DIR)。
    target_per_species : int
        低资源物种目标样本数 (默认 15, 与文档/YAMNet 一致)。
    max_aug_per_species : int
        每物种最多新增增强数 (默认 50, 防失控)。
    seed : int
        随机种子; 每折用 42+fold, 与 YAMNet/LightGBM 同口径。

    返回
    ----
    df_train_aug : pd.DataFrame
        原始训练行 + 增强行; 增强行已填好 spectrogram_path。
        可直接与 df_val 拼成 df_fold 喂给 DataBlock。
    report : dict
        {species: {before, after, added}}, 记录每类增强前后数量。
    """
    img_dir = Path(img_dir) if img_dir else Path("./augmented_spectrograms")
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. 识别低资源物种 (按本折训练集计数, 与 YAMNet/LightGBM 同口径)
    per_class = df_train["primary_label"].value_counts()
    low_resource = per_class[per_class < target_per_species]
    if len(low_resource) == 0:
        print("[FastAI增强] 本折无低资源物种, 跳过。")
        return df_train, {}

    print(f"[FastAI增强] 低资源物种 {len(low_resource)} 种 (<{target_per_species} 条)")

    # 2. 用共享的 expand_train_df 生成增强元数据 (与 YAMNet 完全一致)
    df_expanded, aug_info = expand_train_df(
        df_train.reset_index(drop=True),
        low_resource.index.tolist(),
        target_per_species=target_per_species,
        max_aug_per_species=max_aug_per_species,
        seed=seed,
    )

    # 3. 按 original_filename 分组: 同一原始音频只读一次, 逐条增强
    by_original = {}
    for info in aug_info:
        by_original.setdefault(info["original_filename"], []).append(info)

    new_rows = []
    missing = 0
    for orig_fn, infos in by_original.items():
        orig_row = df_train[df_train["filename"] == orig_fn]
        if len(orig_row) == 0:
            missing += len(infos)
            continue
        path = resolve_audio_fn(orig_row.iloc[0])
        if path is None:
            missing += len(infos)
            continue
        try:
            y, _ = librosa.load(path, sr=sr, mono=True)
            if len(y) == 0:
                missing += len(infos)
                continue
            if len(y) > 0:
                y = librosa.util.normalize(y)
        except Exception:
            missing += len(infos)
            continue

        for info in infos:
            v = int(info.get("variant_index", 0))
            aug_y, _tag = augment_waveform(
                y.astype(np.float32), sr=sr, seed=seed + v)
            stem = Path(str(orig_fn)).stem
            aug_png_name = f"{stem}_aug{v:02d}_{info['species']}.png"
            aug_png_path = img_dir / aug_png_name
            saved = _waveform_to_spectrogram_png(
                aug_y, sr, clip_seconds, n_mels, aug_png_path)
            if saved is None:
                missing += 1
                continue
            new_row = orig_row.iloc[0].to_dict()
            new_row["filename"] = f"{stem}_aug{v:02d}.wav"
            new_row["spectrogram_path"] = str(saved)
            new_rows.append(new_row)

    if missing:
        print(f"  [WARN] {missing} 条增强样本因音频缺失被跳过")

    if new_rows:
        df_train_aug = pd.concat(
            [df_train, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_train_aug = df_train

    print(f"[FastAI增强] df_train: {len(df_train)} -> {len(df_train_aug)} "
          f"(+{len(new_rows)} 条增强频谱图)")

    report = {}
    for sp in low_resource.index:
        before = int(low_resource[sp])
        added = sum(1 for info in aug_info if info["species"] == sp)
        report[sp] = {"before": before, "after": before + added, "added": added}

    return df_train_aug, report


if __name__ == "__main__":
    print("augmentation_glue.py 自检通过")
    print("用法见模块顶部注释, 在 fastaikaggle.txt 的 main_cv 5 折循环里调用")
    print("augment_for_fastai(...)")
