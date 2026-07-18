"""
LightGBM 低资源物种音频增强 - 接入胶水代码
================================================
适用: lightgbm/notebook6312d60402-0715.ipynb (cell-2 训练循环)

设计原理
--------
LightGBM 用 32 维手工特征 (MFCC+频谱统计量), 不是 YAMNet 嵌入。
YAMNet 的 expand_with_augmentation() 操作 fn2wf 波形缓存 + X_train 数组,
不能直接套用。本模块复用 audio_augmentation 的底层:
  - expand_train_df()  : 决定哪些低资源物种、哪些原始样本、各生成几条增强
  - augment_waveform() : 对单条波形施加一次随机增强 (4 种方法共用)
然后针对 LightGBM 加一步: 增强波形 -> _features_from_wave() -> 32 维向量,
拼进 X_tr / y_tr。

接入方式 (在 notebook cell-2 的 5 折循环里, 读 df_train 之后、split_xy 之前):
    from augmentation_glue import augment_for_lightgbm
    X_aug, y_aug, report = augment_for_lightgbm(
        df_train, find_audio, _features_from_wave, label2idx,
        target_per_species=15, sr=SR, seed=SEED + fold)
    X_tr = np.concatenate([X_tr, X_aug], axis=0)
    y_tr = np.concatenate([y_tr, y_aug], axis=0)

依赖: audio_augmentation.py 需在 Kaggle 的 src/ 目录 (与 YAMNet 共用同一份)。
"""

import numpy as np
import pandas as pd
from pathlib import Path

from audio_augmentation import expand_train_df, augment_waveform


def augment_for_lightgbm(df_train, find_audio_fn, feature_fn, label2idx,
                         target_per_species=15, max_aug_per_species=50,
                         sr=22050, duration=30, seed=42):
    """
    为 LightGBM 一折的训练集生成增强样本。

    参数
    ----
    df_train : pd.DataFrame
        当前折的训练集 (含 filename, primary_label, source_year 等列)。
        必须是 notebook cell-2 里读出的 df_train, 因为 find_audio 依赖其列。
    find_audio_fn : callable(row) -> str|None
        notebook 里定义的 find_audio, 用来定位物理音频文件。
    feature_fn : callable(y, sr) -> np.ndarray
        notebook 里定义的 _features_from_wave, 波形 -> 32 维特征向量。
        注意: 它的签名是 _features_from_wave(y), 内部用全局 SR,
        所以这里传波形即可 (sr 参数仅用于 augment_waveform 的 pitch_shift)。
    label2idx : dict
        notebook 里定义的标签映射。
    target_per_species : int
        低资源物种目标样本数 (默认 15, 与文档/YAMNet 一致)。
    max_aug_per_species : int
        每物种最多新增增强数 (默认 50, 防失控)。
    sr : int
        采样率 (LightGBM 用 22050; 增强模块的 pitch_shift 需要它)。
    duration : int|float
        每条音频最长读取秒数 (与 notebook 的 MAX_DURATION 对齐)。
    seed : int
        随机种子; 每折用 SEED+fold, 与 YAMNet 口径一致, 保证可复现。

    返回
    ----
    X_aug : np.ndarray, shape=(N_aug, 32), float32
        增强样本的特征矩阵; N_aug=0 时 shape=(0,)。
    y_aug : np.ndarray, shape=(N_aug,), int64
        增强样本的整数标签。
    report : dict
        {species: {before, after, added}}, 记录每类增强前后数量。
    """
    import librosa

    # 1. 识别低资源物种 (按本折训练集计数, 与 YAMNet 同口径)
    per_class = df_train["primary_label"].value_counts()
    low_resource = per_class[per_class < target_per_species]
    if len(low_resource) == 0:
        print("[LGB增强] 本折无低资源物种, 跳过。")
        return (np.array([], dtype=np.float32),
                np.array([], dtype=np.int64), {})

    print(f"[LGB增强] 低资源物种 {len(low_resource)} 种 (<{target_per_species} 条)")

    # 2. 用共享的 expand_train_df 生成增强元数据 (决定哪些样本各增强几条)
    #    这一步只产 DataFrame 行 + aug_info 列表, 不碰音频, 与 YAMNet 完全一致。
    #    注意: variant_index 只在 aug_info 里, 不在 DataFrame 行里。
    df_expanded, aug_info = expand_train_df(
        df_train.reset_index(drop=True),
        low_resource.index.tolist(),
        target_per_species=target_per_species,
        max_aug_per_species=max_aug_per_species,
        seed=seed,
    )

    # 3. 按 original_filename 分组: 同一原始音频只读一次, 逐条增强
    #    与 YAMNet inject_augmented_waveforms 的分组逻辑完全一致。
    by_original = {}
    for info in aug_info:
        by_original.setdefault(info["original_filename"], []).append(info)

    X_list, y_list = [], []
    missing = 0
    for orig_fn, infos in by_original.items():
        # 用原始 filename 在 df_train 里找行, 调 notebook 的 find_audio 定位文件
        orig_row = df_train[df_train["filename"] == orig_fn]
        if len(orig_row) == 0:
            missing += len(infos)
            continue
        path = find_audio_fn(orig_row.iloc[0])
        if path is None:
            missing += len(infos)
            continue
        try:
            y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
            if len(y) < sr * 0.1:
                missing += len(infos)
                continue
        except Exception:
            missing += len(infos)
            continue

        # 每条变体用 seed + variant_index, 与 YAMNet inject_augmented_waveforms 同口径
        for info in infos:
            v = int(info.get("variant_index", 0))
            aug_y, _tag = augment_waveform(
                y.astype(np.float32), sr=sr, seed=seed + v)
            feat = feature_fn(aug_y)
            if feat is None:
                missing += 1
                continue
            X_list.append(feat)
            y_list.append(label2idx[info["species"]])

    if missing:
        print(f"  [WARN] {missing} 条增强样本因音频缺失/过短被跳过")

    X_aug = np.array(X_list, dtype=np.float32)
    y_aug = np.array(y_list, dtype=np.int64)
    print(f"[LGB增强] 新增增强特征 {len(X_aug)} 条 -> "
          f"X_tr: {len(df_train)} + {len(X_aug)} = {len(df_train)+len(X_aug)}")

    # 4. 报告 (与 YAMNet report 同结构)
    report = {}
    for sp in low_resource.index:
        before = int(low_resource[sp])
        added = sum(1 for info in aug_info if info["species"] == sp)
        report[sp] = {"before": before, "after": before + added, "added": added}

    return X_aug, y_aug, report


if __name__ == "__main__":
    # 自检: 不依赖真实数据, 只验证 import 和函数签名
    print("augmentation_glue.py 自检通过")
    print("用法见模块顶部注释, 在 notebook cell-2 的 5 折循环里调用 "
          "augment_for_lightgbm(...)")
