"""
=============================================================================
On-the-Fly Audio Augmentation for Low-Resource Bird Species
低资源鸟类音频实时增强模块

用途: 在 YAMNet 训练流程中, 对样本数不足的物种进行实时音频增强,
      无需预先生成增强文件, 每次运行随机选择增强方法, 增加多样性。

集成方式: 在 yamnet_finetune_e2e.py 的 build_waveform_cache() 之后调用
         expand_with_augmentation(), 将增强波形注入缓存和训练数组。

依赖: librosa, numpy (Kaggle 环境预装)
=============================================================================
"""

import numpy as np
from pathlib import Path

# ── 单种增强方法 ──────────────────────────────────────────

def time_stretch(y, rate):
    """时间拉伸: 变速不变调。rate<1 减速, rate>1 加速。"""
    import librosa
    return librosa.effects.time_stretch(y=y, rate=rate)


def pitch_shift(y, sr, n_steps):
    """音高偏移: n_steps 个半音。正值升调, 负值降调。"""
    import librosa
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def add_noise(y, snr_db, seed=None):
    """叠加高斯白噪声, 指定信噪比 (dB)。"""
    rng = np.random.RandomState(seed) if seed else np.random
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.randn(len(y)).astype(np.float32) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def change_volume(y, gain):
    """音量缩放: gain<1 减小, gain>1 增大。"""
    return (y * gain).astype(np.float32)


# ── 随机增强调度 ──────────────────────────────────────────

# 增强方法及其参数选项
AUG_METHODS = {
    "time_stretch": {
        "fn": time_stretch,
        "params": [0.85, 0.90, 0.95, 1.05, 1.10],
        "tag": lambda p: f"ts{int(p*100):03d}",
    },
    "pitch_shift": {
        "fn": pitch_shift,
        "params": [-2, -1, 1, 2],
        "tag": lambda p: f"ps{p:+d}",
        "needs_sr": True,
    },
    "noise": {
        "fn": add_noise,
        "params": [5, 10, 15],
        "tag": lambda p: f"noise{p}db",
    },
    "volume": {
        "fn": change_volume,
        "params": [0.70, 0.85, 1.15, 1.30],
        "tag": lambda p: f"vol{int(p*100):03d}",
    },
}


def augment_waveform(y, sr=16000, methods=None, seed=None):
    """
    对单条波形应用一次随机增强。

    Args:
        y: 1-D numpy array, 原始波形
        sr: 采样率 (默认 16000)
        methods: 可选方法列表, None=全部
        seed: 随机种子

    Returns:
        (augmented_y, method_tag)
          augmented_y: 增强后的波形 (长度不变)
          method_tag:   方法标识符, 如 "ts085", "ps+2"
    """
    rng = np.random.RandomState(seed) if seed else np.random
    method_names = list(AUG_METHODS.keys()) if methods is None else methods
    name = rng.choice(method_names)
    info = AUG_METHODS[name]
    param = rng.choice(info["params"])

    if info.get("needs_sr"):
        aug_y = info["fn"](y, sr, param)
    else:
        aug_y = info["fn"](y, param)

    # 确保长度一致 (某些增强可能略微改变长度)
    if len(aug_y) != len(y):
        if len(aug_y) < len(y):
            aug_y = np.pad(aug_y, (0, len(y) - len(aug_y)))
        else:
            aug_y = aug_y[:len(y)]

    return aug_y.astype(np.float32), info["tag"](param)


# ── DataFrame 扩展: 生成增强样本的元数据行 ─────────────────

def expand_train_df(df_train, low_resource_species, target_per_species=15,
                    max_aug_per_species=50, seed=42):
    """
    为低资源物种生成增强样本的 DataFrame 行。

    本函数只生成元数据行 (不处理音频), 音频在后续
    inject_augmented_waveforms 中实时生成。

    Args:
        df_train: 训练集 DataFrame (含 filename, filepath, primary_label 等)
        low_resource_species: list of str, 需要增强的物种代码
        target_per_species: 目标每类最少样本数
        max_aug_per_species: 每个物种最多生成的新增增强样本数 (防止失控)
        seed: 随机种子, 用于低资源样本的选择 (保证可复现)

    Returns:
        df_expanded: 原始行 + 增强行 的 DataFrame
        aug_rows_info: list of dict, 每条增强行的元信息
            [{original_filename, species, method_tag, new_filename}, ...]
    """
    import pandas as pd

    df = df_train.copy()
    df["_is_augmented"] = False
    df["_original_filename"] = ""

    new_rows = []
    aug_info = []
    rng = np.random.RandomState(seed)

    for species in low_resource_species:
        sp_mask = df["primary_label"] == species
        sp_df = df[sp_mask]
        current_count = len(sp_df)
        needed = min(target_per_species - current_count, max_aug_per_species)
        if needed <= 0:
            continue

        # 计算每个原始样本应生成的增强变体数量
        if needed >= current_count:
            # 均匀分配: 每个样本至少分到 needed//current_count 条
            variants_per_sample = needed // current_count
            extra = needed % current_count
            sample_variants = []
            for i in range(current_count):
                n = variants_per_sample + (1 if i < extra else 0)
                n = min(n, 16)  # 单条原始录音最多生成 16 个变体
                sample_variants.append(n)
        else:
            # needed < current_count: 随机选择 needed 个样本各生成 1 条
            # 避免之前 max(1, needed//current_count) 导致的严重超发
            chosen_idx = set(rng.choice(current_count, needed, replace=False))
            sample_variants = [
                1 if i in chosen_idx else 0
                for i in range(current_count)
            ]

        for i, (_, row) in enumerate(sp_df.iterrows()):
            n_variants = sample_variants[i]
            if n_variants <= 0:
                continue

            original_fn = str(row["filename"])
            stem = Path(original_fn).stem

            for v in range(n_variants):
                # 增强方法的 tag 在音频生成时随机确定, 这里用占位符
                new_fn = f"{stem}_aug{v:02d}.wav"
                new_row = row.to_dict()
                new_row["filename"] = new_fn
                new_row["filepath"] = ""  # 无实际文件, 从缓存中读取
                new_row["_is_augmented"] = True
                new_row["_original_filename"] = original_fn
                new_rows.append(new_row)
                aug_info.append({
                    "species": species,
                    "original_filename": original_fn,
                    "new_filename": new_fn,
                    "variant_index": v,
                })

    if new_rows:
        df_expanded = pd.concat(
            [df, pd.DataFrame(new_rows)],
            ignore_index=True
        )
    else:
        df_expanded = df

    return df_expanded, aug_info


# ── 缓存注入: 将增强波形加入 fn2wf ─────────────────────────

def inject_augmented_waveforms(fn2wf, aug_info, sr=16000, seed=42):
    """
    在已有的波形缓存中注入增强波形。

    从原始波形生成增强变体, 以新 filename 为 key 存入 fn2wf。

    Args:
        fn2wf: dict, {filename: waveform_array} (会被原地修改)
        aug_info: expand_train_df() 返回的 aug_rows_info
        sr: 采样率
        seed: 随机种子
    """
    # 按原始文件分组, 同一原始文件的多条增强共享同一个 base waveform
    by_original = {}
    for info in aug_info:
        orig = info["original_filename"]
        if orig not in by_original:
            by_original[orig] = []
        by_original[orig].append(info)

    for orig_fn, infos in by_original.items():
        if orig_fn not in fn2wf:
            print(f"  [WARN] 原始波形不在缓存中: {orig_fn}, 跳过 {len(infos)} 条增强")
            continue

        base_wf = fn2wf[orig_fn].copy()

        for info in infos:
            aug_wf, tag = augment_waveform(base_wf, sr=sr, seed=seed + info["variant_index"])
            fn2wf[info["new_filename"]] = aug_wf
            info["actual_tag"] = tag  # 记录实际应用的增强方法


# ── 一站式函数: 主流程调用 ────────────────────────────────

def expand_with_augmentation(df_train, fn2wf, X_train, y_train_raw,
                              target_per_species=15, max_aug_per_species=50,
                              sr=16000, seed=42):
    """
    一站式: 识别低资源物种 → 扩展 DataFrame → 注入增强波形 → 重建训练数组。

    Args:
        df_train: 训练集 DataFrame
        fn2wf: 波形缓存 dict (会被原地修改)
        X_train: 训练波形数组 (shape=[N, samples])
        y_train_raw: 训练标签数组 (字符串)
        target_per_species: 每类目标最少样本数
        max_aug_per_species: 每物种最多新增增强样本数 (防止失控, 文档规定 50)
        sr: 采样率
        seed: 随机种子

    Returns:
        df_expanded, fn2wf, X_expanded, y_expanded, report
          report: dict {species: {before, after, added}}
    """
    import pandas as pd

    # 1. 识别低资源物种
    per_class = df_train["primary_label"].value_counts()
    low_resource = per_class[per_class < target_per_species]

    if len(low_resource) == 0:
        print("[增强] 所有物种均 >= target_per_species, 无需增强。")
        return df_train, fn2wf, X_train, y_train_raw, {}

    print(f"[增强] 低资源物种: {len(low_resource)} 种 (少于 {target_per_species} 条)")
    for sp, cnt in low_resource.items():
        added = min(target_per_species - cnt, max_aug_per_species)
        print(f"  {sp:>10s}: {cnt} -> {cnt + added} (+{added})")

    # 2. 扩展 DataFrame 元数据
    df_expanded, aug_info = expand_train_df(
        df_train, low_resource.index.tolist(), target_per_species,
        max_aug_per_species=max_aug_per_species, seed=seed
    )
    print(f"[增强] DataFrame: {len(df_train)} -> {len(df_expanded)} 行 "
          f"(+{len(aug_info)} 条增强)")

    # 3. 注入增强波形到缓存
    inject_augmented_waveforms(fn2wf, aug_info, sr=sr, seed=seed)
    print(f"[增强] 波形缓存: {len(fn2wf)} 条 (含增强)")

    # 4. 重建训练数组 (仅重建 X_train, y_train_raw)
    train_mask = df_expanded["_is_augmented"] == False
    aug_mask = df_expanded["_is_augmented"] == True

    # 原始部分直接用已有数组
    X_original = X_train
    y_original = y_train_raw

    # 增强部分从 fn2wf 中提取
    aug_fns = df_expanded[aug_mask]["filename"].tolist()
    aug_labels = df_expanded[aug_mask]["primary_label"].tolist()

    X_aug_list = []
    missing = 0
    for fn in aug_fns:
        if fn in fn2wf:
            X_aug_list.append(fn2wf[fn])
        else:
            missing += 1
    if missing:
        print(f"  [WARN] {missing} 条增强波形未找到, 已跳过")

    if X_aug_list:
        X_aug = np.stack(X_aug_list).astype(np.float32)
        y_aug = np.array(aug_labels)
        X_expanded = np.concatenate([X_original, X_aug], axis=0)
        y_expanded = np.concatenate([y_original, y_aug], axis=0)
    else:
        X_expanded = X_original
        y_expanded = y_original

    print(f"[增强] X_train: {X_train.shape} -> {X_expanded.shape}")

    # 5. 生成报告
    report = {}
    for sp in low_resource.index:
        before = int(low_resource[sp])
        after = int((df_expanded["primary_label"] == sp).sum())
        report[sp] = {"before": before, "after": after, "added": after - before}

    return df_expanded, fn2wf, X_expanded, y_expanded, report


# ── 独立使用: 命令行批量增强 (可选) ────────────────────────

if __name__ == "__main__":
    """测试: 对单条音频生成多个增强变体并保存。"""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_augmentation.py <audio_file> [output_dir]")
        sys.exit(1)

    import librosa
    import soundfile as sf
    from pathlib import Path

    audio_path = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./augmented")
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    stem = Path(audio_path).stem

    print(f"原始: {audio_path} ({len(y)/sr:.1f}s, {sr}Hz)")

    # 生成所有变体
    for method_name, info in AUG_METHODS.items():
        for param in info["params"]:
            if info.get("needs_sr"):
                aug_y = info["fn"](y, sr, param)
            else:
                aug_y = info["fn"](y, param)
            tag = info["tag"](param)
            out_path = out_dir / f"{stem}_{tag}.wav"
            sf.write(str(out_path), aug_y, sr)
            print(f"  -> {out_path}")
