"""
YAMNet 端到端微调 - 单文件自包含版 (Kaggle 一键复制即跑)
=============================================================================
负责人: Wenjuan Huang

数据集: full-kaggle-data/processed data-129834/ (传到 Kaggle 作 input dataset)
  - 02_train_full_weighted.csv   129,834 训练行 (含 sampler_weight / loss_class_weight)
  - 03_test_holdout.csv          14,416 留出测试行
  - cv_fold{1-5}_train/val.csv   5 折 CV 切分
  - class_weights.csv            每物种一行, 预计算权重 (1/sqrt(count), clip[0.25,4.00])
  - 1,126 鸟类物种

集成内容:
  1. Kaggle 挂载扫描 (定位 train_audio + CSV, 剪枝避遍历几十万 .ogg)
  2. CSV 加载 + 音频路径解析 + 波形读取/缓存
  3. 低资源物种音频增强 (time_stretch / pitch_shift / noise / volume, 仅 <15 条的物种)
  4. YAMNet 端到端模型 (解冻顶层 + 分类头)
  5. 自定义训练循环 (差分学习率 + MixUp + 类别权重, 权重读 class_weights.csv)
  6. 5 折交叉验证主流程 + 测试集预测
  7. 噪声鲁棒性评估 (clean / 5dB / 0dB / -5dB 四档, 与 LightGBM 对称)
  8. 推理速度与显存测量

依赖: numpy, pandas, librosa, tensorflow, tensorflow_hub, matplotlib
     (Kaggle 已预装; tensorflow_hub 首次加载 YAMNet 需联网约 17MB)

参考结构: lightgbm/notebook6312d60402-0715.ipynb (单 cell 自包含风格)
         数据准备: full-kaggle-data/Data Augmentation Documentation.md

运行方式 (Kaggle notebook, 单 cell):
  直接粘贴本文件全部内容运行; 或上传 .py 后用 %run -i 运行
  产物: /kaggle/working/yamnet/e2e/ 下的 fold{N}/ 子目录 + cv_summary.csv
=============================================================================
"""

import os
import re
import csv
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. 全局配置
# ============================================================
class Cfg:
    # YAMNet 输入: 16kHz 单声道 float32 [-1,1]
    SR = 16000
    CLIP_SECONDS = 5.0

    # Kaggle 输入根 + 输出目录 (Kaggle 写 /kaggle/working)
    INPUT_BASE = "/kaggle/input"
    OUT_DIR = "/kaggle/working/yamnet/e2e"
    os.makedirs(OUT_DIR, exist_ok=True)

    # CSV 文件名 (与 full-kaggle-data/processed data-129834/ 一致)
    TEST_CSV = "03_test_holdout.csv"
    TRAIN_WEIGHTED_CSV = "02_train_full_weighted.csv"  # 含 sampler_weight / loss_class_weight
    CLASS_WEIGHTS_CSV = "class_weights.csv"            # 每物种一行, 预计算权重
    N_FOLDS = 5

    @staticmethod
    def fold_csvs(fold):
        return (f"cv_fold{fold}_train.csv", f"cv_fold{fold}_val.csv")

    # 音频目录命名 (不同年份不一致)
    AUDIO_ROOT_CANDIDATES = ("train_audio", "train_short_audio")

    # YAMNet 模型 (首次运行联网下载, 约 17MB)
    YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

    # 端到端训练超参数
    BATCH_SIZE = 8
    EPOCHS = 40
    HEAD_LR = 1e-3
    TOP_LAYER_LR = 1e-5
    UNFREEZE_LAYERS = 6
    MIXUP_ALPHA = 0.2
    USE_CLASS_WEIGHTS = True
    DROPOUT = 0.3
    SEED = 42

    # 标签映射路径
    LABEL_MAP_PATH = os.path.join(OUT_DIR, "label_map.json")
    WAVEFORM_CACHE = os.path.join(OUT_DIR, "waveforms_cache.npz")


cfg = Cfg()

# ============================================================
# 1. 扫描 Kaggle 挂载: 一次 os.walk 同时拿到音频根 + CSV 路径
# ============================================================
# 参考 lightgbm/notebook6312d60402-0715.ipynb cell-2 的扫描逻辑。
# 遇到 train_audio / train_short_audio 后剪枝, 不进入几十万 .ogg, 扫描很快。
year2root = {}
csv_paths = {}


def scan_kaggle_inputs(input_base=cfg.INPUT_BASE):
    """遍历 /kaggle/input 一次, 收集音频根目录与目标 CSV 路径。

    返回 (year2root, csv_paths):
      year2root: {年份:int 或根路径字符串 -> 音频根路径}
      csv_paths: {csv 文件名 -> 完整路径}
    """
    target_csvs = {cfg.TEST_CSV, cfg.TRAIN_WEIGHTED_CSV, cfg.CLASS_WEIGHTS_CSV}
    for f in range(1, cfg.N_FOLDS + 1):
        target_csvs.update(cfg.fold_csvs(f))

    if not os.path.exists(input_base):
        print(f"[警告] {input_base} 不存在, 可能不在 Kaggle 或未挂载数据集")
        return {}, {}

    y2r, csvs = {}, {}
    for root, dirs, files in os.walk(input_base, topdown=True):
        for name in cfg.AUDIO_ROOT_CANDIDATES:
            if name not in dirs:
                continue
            cand = os.path.join(root, name)
            try:
                items = os.listdir(cand)
            except Exception:
                items = []
            has_ogg = any(x.lower().endswith(".ogg") for x in items)
            has_subdir = any(os.path.isdir(os.path.join(cand, x)) for x in items)
            if not items or not (has_ogg or has_subdir):
                continue
            m = re.search(r"(20\d{2})", root)
            key = int(m.group(1)) if m else root
            if key not in y2r:
                y2r[key] = cand
        # 剪枝: 不进入音频文件目录, 避免遍历几十万 .ogg
        dirs[:] = [d for d in dirs if d not in cfg.AUDIO_ROOT_CANDIDATES]
        for f in files:
            if f in target_csvs and f not in csvs:
                csvs[f] = os.path.join(root, f)
    return y2r, csvs


def print_mounted_inputs(input_base=cfg.INPUT_BASE, max_per_dir=8):
    """列出 /kaggle/input 下挂载的数据集及其一层子目录, 用于排查。"""
    if not os.path.exists(input_base):
        print(f"[挂载] {input_base} 不存在, 可能不在 Kaggle 环境或未挂载数据集。")
        return
    entries = sorted(d for d in os.listdir(input_base)
                     if os.path.isdir(os.path.join(input_base, d)))
    if not entries:
        print(f"[挂载] {input_base} 下无子目录, 请在 notebook 右侧 Add Input 挂载数据集。")
        return
    print(f"[挂载] {input_base} 下发现 {len(entries)} 个数据集:")
    for d in entries:
        sub = [c for c in os.listdir(os.path.join(input_base, d))
               if os.path.isdir(os.path.join(input_base, d, c))][:max_per_dir]
        print(f"  - {d}  子目录: {sub}")


# ============================================================
# 2. 共享标签映射 (与 LightGBM / 旧版 YAMNet 完全一致)
# ============================================================
def build_label_map(csv_paths_dict, label_map_path=cfg.LABEL_MAP_PATH):
    """classes = sorted(fold1 train + fold1 val + test 的 primary_label 并集)。
    顺序与 LightGBM 相同, 两模型 classes 数组对齐, 整数下标可直接横比。
    """
    tr, va = cfg.fold_csvs(1)
    label_series = pd.concat([
        pd.read_csv(csv_paths_dict[tr], usecols=["primary_label"])["primary_label"],
        pd.read_csv(csv_paths_dict[va], usecols=["primary_label"])["primary_label"],
        pd.read_csv(csv_paths_dict[cfg.TEST_CSV], usecols=["primary_label"])["primary_label"],
    ]).astype(str)
    classes = sorted(label_series.unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    with open(label_map_path, "w", encoding="utf-8") as fp:
        json.dump({"label2idx": label2idx,
                   "idx2label": {str(i): c for i, c in idx2label.items()}},
                  fp, ensure_ascii=False, indent=2)
    print(f"[标签] {len(classes)} 类, 存 {label_map_path}")
    return classes, label2idx, idx2label


# ============================================================
# 3. 音频路径解析 + 波形读取 + 波形缓存
# ============================================================
def parse_years(s):
    """解析 source_year 字段, 返回年份列表 (整数, 降序, 最新优先)。
    数据中可能用逗号或分号分隔多个年份; 无效或空值返回 []。"""
    if s is None or (isinstance(s, float) and pd.isna(s)) or not str(s).strip():
        return []
    years = []
    for part in str(s).replace(";", ",").split(","):
        try:
            years.append(int(part.strip()))
        except ValueError:
            pass
    return sorted(years, reverse=True)


def find_audio_path(row, year2root_dict, all_roots_list):
    """根据一行 CSV 元数据定位音频文件。

    不同年份 BirdCLEF 目录结构不一致, 依次尝试:
      1) root/<CSV 中的 filename>
      2) root/<primary_label>/<basename>
      3) root/<basename>
    优先 source_year 指定的目录, 未命中再遍历其他年份。
    返回第一个真实存在的路径, 全部未命中返回 None。
    """
    fname = str(row["filename"]).replace("\\", "/")
    base = fname.split("/")[-1]
    pl = str(row.get("primary_label", "")).strip()

    roots_to_try = []
    for y in parse_years(row.get("source_year", None)):
        if y in year2root_dict and year2root_dict[y] not in roots_to_try:
            roots_to_try.append(year2root_dict[y])
    for r in all_roots_list:
        if r not in roots_to_try:
            roots_to_try.append(r)

    candidates = []
    for root in roots_to_try:
        for cand in (os.path.join(root, fname),
                     os.path.join(root, pl, base) if pl else None,
                     os.path.join(root, base)):
            if cand and cand not in candidates:
                candidates.append(cand)
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_waveform(path, sr=cfg.SR, clip_seconds=cfg.CLIP_SECONDS):
    """读取音频, 重采样 16kHz, 转单声道, 固定长度 (取中段/补零), 峰值归一化。
    返回 shape=[samples] 的 float32 数组。"""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * clip_seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak).astype(np.float32)


def build_waveform_cache(df_train, df_val, df_test, year2root_dict,
                         cache_path=cfg.WAVEFORM_CACHE):
    """为三个 split 预计算波形并缓存到 npz。端到端训练每 epoch 都要读波形,
    缓存到内存避免反复磁盘 I/O。

    缓存 key: filenames (跨折复用)。
    返回 (X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw,
           test_filenames, fn2wf)。
    """
    fn2wf = {}
    if os.path.exists(cache_path):
        print(f"[波形缓存] 读已有: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        for i, fn in enumerate(d["filenames"]):
            fn2wf[str(fn)] = d["waveforms"][i]
        print(f"  命中 {len(fn2wf)} 条")

    all_roots = list(year2root_dict.values())

    def to_records(df):
        return [(str(r["filename"]),
                 find_audio_path(r, year2root_dict, all_roots),
                 str(r["primary_label"])) for _, r in df.iterrows()]

    rec_train = to_records(df_train)
    rec_val = to_records(df_val)
    rec_test = to_records(df_test)

    # 过滤未命中音频的记录 (path is None)
    rec_train = [r for r in rec_train if r[1] is not None]
    rec_val = [r for r in rec_val if r[1] is not None]
    rec_test = [r for r in rec_test if r[1] is not None]

    all_recs = rec_train + rec_val + rec_test
    seen = set()
    unique_recs = []
    for fn, fp, lab in all_recs:
        if fn not in seen:
            seen.add(fn)
            unique_recs.append((fn, fp, lab))

    need = [(fn, fp, lab) for fn, fp, lab in unique_recs if fn not in fn2wf]
    if need:
        print(f"[波形缓存] 需补算 {len(need)} 条 ...")
        for i, (fn, fp, lab) in enumerate(need):
            fn2wf[fn] = load_waveform(fp)
            if (i + 1) % 100 == 0 or (i + 1) == len(need):
                print(f"  已补算 {i+1}/{len(need)}")
        all_fn = list(fn2wf.keys())
        all_wf = np.stack([fn2wf[f] for f in all_fn]).astype(np.float32)
        np.savez(cache_path, filenames=np.array(all_fn), waveforms=all_wf)
        print(f"[波形缓存] 已回写: {cache_path}")
    else:
        print("[波形缓存] 全部命中")

    def slice_xy(records):
        X = np.stack([fn2wf[fn] for fn, _, _ in records]).astype(np.float32)
        y = np.array([lab for _, _, lab in records])
        return X, y

    X_train, y_train_raw = slice_xy(rec_train)
    X_val, y_val_raw = slice_xy(rec_val)
    X_test, y_test_raw = slice_xy(rec_test)
    test_filenames = [fn for fn, _, _ in rec_test]
    return X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw, test_filenames, fn2wf


# ============================================================
# 4. 低资源物种音频增强 (自包含, 无外部依赖)
# ============================================================
# 单种增强方法
def _time_stretch(y, rate):
    import librosa
    return librosa.effects.time_stretch(y=y, rate=rate)


def _pitch_shift(y, sr, n_steps):
    import librosa
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def _add_noise_aug(y, snr_db, seed=None):
    rng = np.random.RandomState(seed) if seed else np.random
    signal_power = np.mean(y ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.randn(len(y)).astype(np.float32) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def _change_volume(y, gain):
    return (y * gain).astype(np.float32)


# 增强方法及其参数选项
_AUG_METHODS = {
    "time_stretch": {"fn": _time_stretch, "params": [0.85, 0.90, 0.95, 1.05, 1.10],
                     "tag": lambda p: f"ts{int(p*100):03d}"},
    "pitch_shift":  {"fn": _pitch_shift, "params": [-2, -1, 1, 2],
                     "tag": lambda p: f"ps{p:+d}", "needs_sr": True},
    "noise":        {"fn": _add_noise_aug, "params": [5, 10, 15],
                     "tag": lambda p: f"noise{p}db"},
    "volume":       {"fn": _change_volume, "params": [0.70, 0.85, 1.15, 1.30],
                     "tag": lambda p: f"vol{int(p*100):03d}"},
}


def augment_waveform(y, sr=cfg.SR, methods=None, seed=None):
    """对单条波形应用一次随机增强。
    返回 (augmented_y, method_tag)。"""
    rng = np.random.RandomState(seed) if seed else np.random
    method_names = list(_AUG_METHODS.keys()) if methods is None else methods
    name = rng.choice(method_names)
    info = _AUG_METHODS[name]
    param = rng.choice(info["params"])
    if info.get("needs_sr"):
        aug_y = info["fn"](y, sr, param)
    else:
        aug_y = info["fn"](y, param)
    # 确保长度一致
    if len(aug_y) != len(y):
        if len(aug_y) < len(y):
            aug_y = np.pad(aug_y, (0, len(y) - len(aug_y)))
        else:
            aug_y = aug_y[:len(y)]
    return aug_y.astype(np.float32), info["tag"](param)


def expand_with_augmentation(df_train, fn2wf, X_train, y_train_raw,
                              target_per_species=15, max_aug_per_species=50,
                              sr=cfg.SR, seed=cfg.SEED):
    """一站式: 识别低资源物种 → 扩展 DataFrame → 注入增强波形 → 重建训练数组。

    返回 (df_expanded, fn2wf, X_expanded, y_expanded, report)。
    report: dict {species: {before, after, added}}。
    """
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

    # 2. 扩展 DataFrame 元数据 (为每个低资源物种生成增强条目)
    df = df_train.copy()
    df["_is_augmented"] = False
    df["_original_filename"] = ""
    new_rows = []
    aug_info = []
    rng = np.random.RandomState(seed)

    for species in low_resource.index.tolist():
        sp_mask = df["primary_label"] == species
        sp_df = df[sp_mask]
        current_count = len(sp_df)
        needed = min(target_per_species - current_count, max_aug_per_species)
        if needed <= 0:
            continue

        # 每个原始样本生成多少个变体
        if needed >= current_count:
            variants_per_sample = needed // current_count
            extra = needed % current_count
            sample_variants = []
            for i in range(current_count):
                n = variants_per_sample + (1 if i < extra else 0)
                n = min(n, 16)
                sample_variants.append(n)
        else:
            chosen_idx = set(rng.choice(current_count, needed, replace=False))
            sample_variants = [1 if i in chosen_idx else 0 for i in range(current_count)]

        for i, (_, row) in enumerate(sp_df.iterrows()):
            n_variants = sample_variants[i]
            if n_variants <= 0:
                continue
            original_fn = str(row["filename"])
            stem = Path(original_fn).stem
            for v in range(n_variants):
                new_fn = f"{stem}_aug{v:02d}.wav"
                new_row = row.to_dict()
                new_row["filename"] = new_fn
                new_row["filepath"] = ""
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
        df_expanded = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_expanded = df
    print(f"[增强] DataFrame: {len(df_train)} -> {len(df_expanded)} 行 "
          f"(+{len(aug_info)} 条增强)")

    # 3. 注入增强波形到缓存
    by_original = {}
    for info in aug_info:
        by_original.setdefault(info["original_filename"], []).append(info)

    for orig_fn, infos in by_original.items():
        if orig_fn not in fn2wf:
            print(f"  [WARN] 原始波形不在缓存: {orig_fn}, 跳过 {len(infos)} 条")
            continue
        base_wf = fn2wf[orig_fn].copy()
        for info in infos:
            aug_wf, tag = augment_waveform(base_wf, sr=sr,
                                            seed=seed + info["variant_index"])
            fn2wf[info["new_filename"]] = aug_wf
            info["actual_tag"] = tag
    print(f"[增强] 波形缓存: {len(fn2wf)} 条 (含增强)")

    # 4. 重建训练数组
    aug_mask = df_expanded["_is_augmented"] == True
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
        X_expanded = np.concatenate([X_train, X_aug], axis=0)
        y_expanded = np.concatenate([y_train_raw, y_aug], axis=0)
    else:
        X_expanded = X_train
        y_expanded = y_train_raw
    print(f"[增强] X_train: {X_train.shape} -> {X_expanded.shape}")

    # 5. 报告
    report = {}
    for sp in low_resource.index:
        before = int(low_resource[sp])
        after = int((df_expanded["primary_label"] == sp).sum())
        report[sp] = {"before": before, "after": after, "added": after - before}

    return df_expanded, fn2wf, X_expanded, y_expanded, report


def label_to_idx(y_raw, label2idx):
    """字符串标签 → 整数索引数组 (int64)。"""
    return np.array([label2idx[str(l)] for l in y_raw], dtype=np.int64)


# ============================================================
# 5. 类别权重: 读 class_weights.csv (loss_class_weight, 逆平方根+clip[0.25,4.00])
# ============================================================
def compute_class_weights(y, num_classes, label2idx=None,
                          class_weights_csv=cfg.CLASS_WEIGHTS_CSV):
    """读 processed data-129834/class_weights.csv 的 loss_class_weight 列。
    数据集已按 1/sqrt(count) 预计算并 clip 到 [0.25, 4.00], 均值归一化到 1。
    未在 CSV 中出现的类 (test-only) 权重 1.0。
    返回 shape=(num_classes,) 的 float32 数组。"""
    if not os.path.exists(class_weights_csv):
        # 回退: 1/count 加权 (仅当 CSV 不可用时; 不推荐, 与文档不符)
        counts = Counter(y.tolist() if hasattr(y, "tolist") else list(y))
        total = len(y)
        weights = np.ones(num_classes, dtype=np.float32)
        for c in range(num_classes):
            n = counts.get(c, 0)
            weights[c] = total / (num_classes * max(n, 1))
        return weights / weights.mean()

    df = pd.read_csv(class_weights_csv)
    if label2idx is None:
        raise ValueError("label2idx 必须传入, 才能把 per-species 权重映射到类别下标")
    weights = np.ones(num_classes, dtype=np.float32)
    for _, row in df.iterrows():
        lab = str(row["primary_label"])
        if lab in label2idx:
            weights[label2idx[lab]] = float(row["loss_class_weight"])
    # 归一化均值为 1 (与训练循环里 sample_weight 默认尺度对齐)
    weights = weights / weights.mean()
    return weights


# ============================================================
# 6. 端到端模型构建: YAMNet (解冻顶层) + 分类头
# ============================================================
def _identify_yamnet_layer_names(yamnet_layer):
    """从 hub.KerasLayer 的变量名提取卷积层名, 用于选择性解冻。
    MobileNetV1 层命名: conv2d_0 (底层) -> conv2d_26 (顶层)。
    返回按层索引排序的层名列表, 失败时返回空列表。"""
    layer_names = set()
    for v in yamnet_layer.variables:
        for p in v.name.split("/"):
            if p.startswith("conv2d"):
                layer_names.add(p)
                break

    def _idx(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0
    return sorted(layer_names, key=_idx)


def build_e2e_model(num_classes, cfg_obj=cfg):
    """构建端到端模型: 原始波形 -> YAMNet(可训练) -> 帧均值池化 -> 分类头。

    选择性解冻: hub.KerasLayer 不支持逐变量 trainable, 但在训练循环中
    只为解冻变量 apply_gradients 即可。

    返回 (model, yamnet_layer, trainable_yamnet_vars)。
    """
    yamnet_layer = hub.KerasLayer(
        cfg_obj.YAMNET_HANDLE,
        trainable=True,
        arguments={"_squeeze": True},
        name="yamnet_backbone",
    )

    input_wav = tf.keras.Input(
        shape=(int(cfg_obj.SR * cfg_obj.CLIP_SECONDS),),
        dtype=tf.float32, name="waveform_input")

    embeddings = yamnet_layer(input_wav)
    if len(embeddings.shape) == 3:
        pooled = tf.keras.layers.GlobalAveragePooling1D(name="frame_avg_pool")(embeddings)
    else:
        pooled = embeddings

    x = tf.keras.layers.Dense(256, activation="relu", name="head_fc")(pooled)
    x = tf.keras.layers.Dropout(cfg_obj.DROPOUT, name="head_dropout")(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=input_wav, outputs=output, name="yamnet_e2e")

    # 选择性解冻: 识别顶层变量
    conv_names = _identify_yamnet_layer_names(yamnet_layer)
    all_yamnet_vars = list(yamnet_layer.trainable_variables)

    if conv_names and cfg_obj.UNFREEZE_LAYERS < len(conv_names):
        unfreeze_names = set(conv_names[-cfg_obj.UNFREEZE_LAYERS:])
        trainable_yamnet_vars = [v for v in all_yamnet_vars
                                 if any(uf in v.name for uf in unfreeze_names)]
        frozen_names = [n for n in conv_names if n not in unfreeze_names]
        print(f"[端到端] 解冻 YAMNet 顶层 {cfg_obj.UNFREEZE_LAYERS} 层: {sorted(unfreeze_names)}")
        print(f"[端到端] 冻结底层 {len(frozen_names)} 层: {frozen_names[:10]}{'...' if len(frozen_names)>10 else ''}")
    else:
        trainable_yamnet_vars = all_yamnet_vars
        print(f"[端到端] 无法识别层名或 UNFREEZE_LAYERS>=总层数, 整体解冻 YAMNet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg_obj.HEAD_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    yamnet_trainable = sum(int(np.prod(v.shape)) for v in trainable_yamnet_vars)
    yamnet_frozen = sum(int(np.prod(v.shape)) for v in all_yamnet_vars) - yamnet_trainable
    head_vars = [v for v in model.trainable_variables if v not in all_yamnet_vars]
    head_params = sum(int(np.prod(v.shape)) for v in head_vars)
    print(f"\n[端到端] 参数统计:")
    print(f"  YAMNet 可训练 (顶层): {yamnet_trainable:,}")
    print(f"  YAMNet 冻结 (底层):   {yamnet_frozen:,}")
    print(f"  分类头可训练:         {head_params:,}")
    print(f"  总可训练:             {yamnet_trainable + head_params:,}")

    return model, yamnet_layer, trainable_yamnet_vars


# ============================================================
# 7. 自定义训练循环: 差分学习率 + MixUp + 类别权重
# ============================================================
class E2ETrainer:
    """自定义训练循环: MixUp + 差分学习率 + 类别权重。
    YAMNet 解冻变量用 TOP_LAYER_LR (小), 分类头用 HEAD_LR (大)。
    """

    def __init__(self, model, yamnet_layer, trainable_yamnet_vars,
                 num_classes, y_train, label2idx, cfg_obj=cfg):
        self.model = model
        self.num_classes = num_classes
        self.cfg = cfg_obj

        self.yamnet_vars = trainable_yamnet_vars
        self.head_vars = [v for v in model.trainable_variables
                          if v not in yamnet_layer.trainable_variables]
        self.update_vars = self.yamnet_vars + self.head_vars

        self.opt_head = tf.keras.optimizers.Adam(cfg_obj.HEAD_LR)
        self.opt_yamnet = tf.keras.optimizers.Adam(cfg_obj.TOP_LAYER_LR)

        if cfg_obj.USE_CLASS_WEIGHTS:
            self.class_weights = compute_class_weights(y_train, num_classes, label2idx=label2idx)
            print(f"[训练] 类别权重: min={self.class_weights.min():.3f}, "
                  f"max={self.class_weights.max():.3f}, mean={self.class_weights.mean():.3f}")
        else:
            self.class_weights = None

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(self, X_batch, y_batch, sample_weight):
        with tf.GradientTape() as tape:
            logits = self.model(X_batch, training=True)
            loss = self.loss_fn(y_batch, logits, sample_weight=sample_weight)
        grads = tape.gradient(loss, self.update_vars)
        n_yam = len(self.yamnet_vars)
        self.opt_yamnet.apply_gradients(zip(grads[:n_yam], self.yamnet_vars))
        self.opt_head.apply_gradients(zip(grads[n_yam:], self.head_vars))
        self.train_acc.update_state(y_batch, logits)
        return loss

    @tf.function
    def train_step_mixup(self, X_batch, y_onehot, sample_weight):
        with tf.GradientTape() as tape:
            logits = self.model(X_batch, training=True)
            per_sample = tf.keras.losses.categorical_crossentropy(y_onehot, logits)
            loss = tf.reduce_mean(per_sample * sample_weight)
        grads = tape.gradient(loss, self.update_vars)
        n_yam = len(self.yamnet_vars)
        self.opt_yamnet.apply_gradients(zip(grads[:n_yam], self.yamnet_vars))
        self.opt_head.apply_gradients(zip(grads[n_yam:], self.head_vars))
        self.train_acc.update_state(tf.argmax(y_onehot, axis=-1), logits)
        return loss

    @tf.function
    def val_step(self, X_batch, y_batch):
        logits = self.model(X_batch, training=False)
        self.val_acc.update_state(y_batch, logits)

    def fit(self, X_train, y_train, X_val, y_val, weights_path=None):
        """自定义训练循环: MixUp + 类别权重 + 差分学习率 + 学习率衰减 + 早停。"""
        n = len(X_train)
        bs = self.cfg.BATCH_SIZE
        steps = (n + bs - 1) // bs
        best_val_acc = 0.0
        patience = 0
        reduce_cnt = 0

        for epoch in range(self.cfg.EPOCHS):
            perm = np.random.permutation(n)
            X_s, y_s = X_train[perm], y_train[perm]
            self.train_acc.reset_state()
            ep_loss = 0.0

            for step in range(steps):
                s = step * bs
                e = min(s + bs, n)
                Xb, yb = X_s[s:e], y_s[s:e]
                if self.class_weights is not None:
                    sw = self.class_weights[yb].astype(np.float32)
                else:
                    sw = np.ones(len(yb), dtype=np.float32)

                if self.cfg.MIXUP_ALPHA > 0 and len(Xb) > 1:
                    lam = np.random.beta(self.cfg.MIXUP_ALPHA, self.cfg.MIXUP_ALPHA)
                    idx = np.random.permutation(len(Xb))
                    Xb_mix = (lam * Xb + (1 - lam) * Xb[idx]).astype(np.float32)
                    y_oh = (lam * tf.one_hot(yb, depth=self.num_classes).numpy() +
                            (1 - lam) * tf.one_hot(yb[idx], depth=self.num_classes).numpy()
                           ).astype(np.float32)
                    sw_mix = (lam * sw + (1 - lam) * sw[idx]).astype(np.float32)
                    loss = self.train_step_mixup(
                        tf.constant(Xb_mix), tf.constant(y_oh), tf.constant(sw_mix))
                else:
                    loss = self.train_step(
                        tf.constant(Xb), tf.constant(yb), tf.constant(sw))
                ep_loss += float(loss)

            self.val_acc.reset_state()
            vsteps = (len(X_val) + bs - 1) // bs
            for step in range(vsteps):
                s = step * bs
                e = min(s + bs, len(X_val))
                self.val_step(X_val[s:e], y_val[s:e])

            v_acc = float(self.val_acc.result())
            t_acc = float(self.train_acc.result())
            ep_loss /= steps
            print(f"  Epoch {epoch+1}/{self.cfg.EPOCHS}: loss={ep_loss:.4f} "
                  f"acc={t_acc:.4f} val_acc={v_acc:.4f}")

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                patience = 0
                if weights_path:
                    self.model.save_weights(str(weights_path))
            else:
                patience += 1
                if patience > 0 and patience % 4 == 0:
                    old_h = self.opt_head.learning_rate.numpy()
                    self.opt_head.learning_rate.assign(max(old_h * 0.5, 1e-7))
                    old_y = self.opt_yamnet.learning_rate.numpy()
                    self.opt_yamnet.learning_rate.assign(max(old_y * 0.5, 1e-8))
                    reduce_cnt += 1
                    print(f"  Reduce LR (#{reduce_cnt}): "
                          f"head={self.opt_head.learning_rate.numpy():.2e}, "
                          f"yamnet={self.opt_yamnet.learning_rate.numpy():.2e}")
                if patience >= 8:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if weights_path and Path(weights_path).exists():
            self.model.load_weights(str(weights_path))
            print(f"  Restored best weights (val_acc={best_val_acc:.4f})")
        return {"best_val_acc": best_val_acc}


# ============================================================
# 8. 主流程: 5 折端到端微调
# ============================================================
def main_cv_e2e(n_folds=cfg.N_FOLDS, cfg_obj=cfg):
    """5 折端到端微调主流程。

    与 LightGBM cell-2 对称: 同一 5 折划分, 同一测试集, 同一标签映射,
    结果可直接横比。产物存入 OUT_DIR/fold{N}/。
    """
    print("=" * 60)
    print("  YAMNet 端到端微调 (解冻顶层 + 差分学习率 + MixUp + 类别权重)")
    print("=" * 60)

    # 扫描挂载
    print("\n===== 挂载自检 =====")
    print_mounted_inputs(cfg_obj.INPUT_BASE)
    year2root, csv_paths_local = scan_kaggle_inputs(cfg_obj.INPUT_BASE)
    print(f"[音频发现] 共 {len(year2root)} 个年度音频根:")
    for y, r in sorted(year2root.items(), key=lambda kv: str(kv[0])):
        print(f"  {y} -> {r}")
    print("====================\n")
    if not year2root:
        raise SystemExit("[错误] 未在 /kaggle/input 下发现 train_audio / train_short_audio 目录")

    # 标签映射 (与 LightGBM / 旧版 YAMNet 完全一致)
    classes, label2idx, idx2label = build_label_map(csv_paths_local)
    num_classes = len(classes)

    clean_accs = []
    for fold in range(1, n_folds + 1):
        print(f"\n{'#'*20} FOLD {fold}/{n_folds} {'#'*20}")
        tf.random.set_seed(cfg_obj.SEED + fold)
        np.random.seed(cfg_obj.SEED + fold)

        tr_csv, va_csv = cfg_obj.fold_csvs(fold)
        df_train = pd.read_csv(csv_paths_local[tr_csv])
        df_val = pd.read_csv(csv_paths_local[va_csv])
        df_test = pd.read_csv(csv_paths_local[cfg_obj.TEST_CSV])
        print(f"[CSV] train={len(df_train)} val={len(df_val)} test={len(df_test)}")

        # 预计算波形缓存 (跨折复用, 保留 fn2wf 供低资源增强)
        X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw, test_filenames, fn2wf = \
            build_waveform_cache(df_train, df_val, df_test, year2root)

        # 低资源物种音频增强 (on-the-fly)
        df_train_aug, fn2wf, X_train, y_train_raw, aug_report = expand_with_augmentation(
            df_train, fn2wf, X_train, y_train_raw,
            target_per_species=15, sr=cfg_obj.SR, seed=cfg_obj.SEED + fold)

        y_train = label_to_idx(y_train_raw, label2idx)
        y_val = label_to_idx(y_val_raw, label2idx)
        y_test = label_to_idx(y_test_raw, label2idx)
        print(f"[划分] fold{fold}: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

        # 构建端到端模型
        model, yamnet_layer, trainable_yamnet_vars = build_e2e_model(num_classes, cfg_obj)
        fold_out = Path(cfg_obj.OUT_DIR) / f"fold{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        # 训练
        trainer = E2ETrainer(model, yamnet_layer, trainable_yamnet_vars,
                             num_classes, y_train, label2idx, cfg_obj)
        weights_path = fold_out / "best_weights.weights.h5"
        print(f"[训练] fold{fold} 开始 ...")
        t0 = time.time()
        trainer.fit(X_train, y_train, X_val, y_val, weights_path=weights_path)
        elapsed = (time.time() - t0) / 60
        print(f"[训练] fold{fold} 完成, 用时 {elapsed:.1f} 分钟")

        # 测试集评估 (分批避免 GPU OOM)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0, batch_size=8)
        print(f"[结果] fold{fold} 测试集准确率 = {test_acc:.4f}")
        clean_accs.append(float(test_acc))

        # 保存权重 + 预测
        model.save_weights(str(fold_out / "best_weights.weights.h5"))
        with open(fold_out / "model_arch.json", "w", encoding="utf-8") as f:
            f.write(model.to_json())
        y_pred = np.argmax(model.predict(X_test, verbose=0, batch_size=8), axis=1)
        np.savez(fold_out / "test_predictions.npz",
                 y_true=y_test, y_pred=y_pred,
                 classes=np.array(classes),
                 test_filenames=np.array(test_filenames))
        print(f"[结果] fold{fold} 已存: {fold_out}")

    # 汇总
    clean_arr = np.array(clean_accs)
    e2e_cv = pd.DataFrame({"fold": range(1, n_folds + 1), "clean_acc": clean_arr})
    e2e_cv.to_csv(os.path.join(cfg_obj.OUT_DIR, "cv_per_fold.csv"), index=False)
    summary = pd.DataFrame([{
        "metric": "clean_acc",
        "mean": float(clean_arr.mean()),
        "std": float(clean_arr.std(ddof=0)),
    }])
    summary.to_csv(os.path.join(cfg_obj.OUT_DIR, "cv_summary.csv"), index=False)
    print(f"\n[CV] e2e clean accuracy: {clean_arr.mean():.4f} +/- {clean_arr.std(ddof=0):.4f}")
    print(f"[CV] 逐折: {[f'{a:.4f}' for a in clean_accs]}")
    print(f"[CV] 产物目录: {cfg_obj.OUT_DIR}")
    return clean_accs


# ============================================================
# 9. 噪声鲁棒性评估 (与 LightGBM cell-3 对称, 可直接横比)
# ============================================================
SNR_TIERS = ["clean", "5dB", "0dB", "-5dB"]


def add_gaussian_noise(waveform, snr_db, rng):
    """按指定 SNR 叠加高斯白噪声, 叠后峰值归一化到 [-1,1]。
    SNR(dB)=10*log10(P_signal/P_noise); rng 由外部传入保证可复现。"""
    signal_power = np.mean(waveform ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    noisy = waveform + noise
    peak = np.max(np.abs(noisy)) + 1e-9
    return (noisy / peak).astype(np.float32)


def _load_e2e_model(fold_dir, num_classes, cfg_obj=cfg):
    """重建端到端模型并加载权重 (不依赖 .keras, 避免 hub.KerasLayer 加载联网)。"""
    weights_path = Path(fold_dir) / "best_weights.weights.h5"
    if not weights_path.exists():
        keras_path = Path(fold_dir) / "yamnet_e2e_model.keras"
        if keras_path.exists():
            return tf.keras.models.load_model(str(keras_path))
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")
    model, _, _ = build_e2e_model(num_classes, cfg_obj)
    model.load_weights(str(weights_path))
    return model


def _batched_predict(model, X, batch_size=16):
    """分批预测, 避免 GPU OOM (端到端前向占显存大)。"""
    preds_all = []
    n = len(X)
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size]
        preds = model.predict(batch, verbose=0)
        preds_all.append(np.argmax(preds, axis=1))
    return np.concatenate(preds_all)


def main_noise_eval(n_folds=cfg.N_FOLDS, cfg_obj=cfg):
    """端到端模型 5 折噪声评估: 对 03_test_holdout.csv 每条音频叠噪后直喂 e2e 模型。
    噪声 rng 与 LightGBM 完全一致 (SEED=42, 跨样本跨档推进)。"""
    print("=" * 60)
    print("  YAMNet 端到端模型噪声鲁棒性评估")
    print("=" * 60)

    # 重新扫描挂载 + 读 CSV (假设本函数作为独立 cell 运行)
    year2root_n, csv_paths_n = scan_kaggle_inputs(cfg_obj.INPUT_BASE)
    df_test = pd.read_csv(csv_paths_n[cfg_obj.TEST_CSV])
    print(f"[噪声评估] 测试集 {len(df_test)} 条")

    # 标签映射 (复用训练时存的)
    label_map = json.loads(Path(cfg_obj.LABEL_MAP_PATH).read_text(encoding="utf-8"))
    label2idx = label_map["label2idx"]
    num_classes = len(label2idx)

    # 预计算所有测试集波形
    all_roots = list(year2root_n.values())
    print(f"[噪声评估] 预加载 {len(df_test)} 条测试波形 ...")
    clean_wfs = []
    test_rows_kept = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        path = find_audio_path(row, year2root_n, all_roots)
        if path is None:
            continue
        clean_wfs.append(load_waveform(path))
        test_rows_kept.append(row)
        if (i + 1) % 100 == 0 or (i + 1) == len(df_test):
            print(f"  {i+1}/{len(df_test)}")
    print(f"[噪声评估] 命中 {len(clean_wfs)} 条测试波形")

    y_true = np.array([label2idx[str(r["primary_label"])] for r in test_rows_kept], dtype=np.int64)

    # 噪声种子 (与 LightGBM / 旧版 YAMNet 完全一致)
    rng = np.random.default_rng(cfg_obj.SEED)

    # 预计算各 SNR 档的叠噪波形
    print("[噪声评估] 预计算噪声波形 ...")
    snr_vals = [("clean", None), ("5dB", 5.0), ("0dB", 0.0), ("-5dB", -5.0)]
    noisy_wfs_by_snr = {}
    for snr_key, snr_val in snr_vals:
        if snr_val is None:
            noisy_wfs_by_snr[snr_key] = np.stack(clean_wfs).astype(np.float32)
        else:
            wfs = [add_gaussian_noise(wf, snr_val, rng) for wf in clean_wfs]
            noisy_wfs_by_snr[snr_key] = np.stack(wfs).astype(np.float32)
        print(f"  {snr_key}: shape={noisy_wfs_by_snr[snr_key].shape}")

    rows = []
    for fold in range(1, n_folds + 1):
        fold_dir = Path(cfg_obj.OUT_DIR) / f"fold{fold}"
        weights_path = fold_dir / "best_weights.weights.h5"
        keras_path = fold_dir / "yamnet_e2e_model.keras"
        if not weights_path.exists() and not keras_path.exists():
            print(f"[跳过] fold{fold} 模型不存在: {fold_dir}")
            continue

        model = _load_e2e_model(fold_dir, num_classes, cfg_obj)
        print(f"\n[噪声评估] fold{fold} ...")

        acc_by_snr = {}
        preds_by_snr = {}
        for snr_key in SNR_TIERS:
            X = noisy_wfs_by_snr[snr_key]
            preds = _batched_predict(model, X, batch_size=16)
            acc = float(np.mean(preds == y_true))
            acc_by_snr[snr_key] = acc
            preds_by_snr[snr_key] = preds
            print(f"  {snr_key}: {acc:.4f}")

        np.savez(fold_dir / "noise_results.npz",
                 snr_tiers=np.array(SNR_TIERS),
                 acc=np.array([acc_by_snr[s] for s in SNR_TIERS]),
                 y_true=y_true,
                 preds_clean=preds_by_snr["clean"],
                 preds_5dB=preds_by_snr["5dB"],
                 preds_0dB=preds_by_snr["0dB"],
                 preds_n5dB=preds_by_snr["-5dB"])
        rows.append({"fold": fold, **{f"acc_{s}": acc_by_snr[s] for s in SNR_TIERS}})

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(cfg_obj.OUT_DIR, "cv_noise_per_fold.csv"), index=False)

    # 合并到 cv_summary
    summary_path = os.path.join(cfg_obj.OUT_DIR, "cv_summary.csv")
    existing = pd.read_csv(summary_path) if os.path.exists(summary_path) \
        else pd.DataFrame(columns=["metric", "mean", "std"])
    noise_metrics = [f"acc_{s}" for s in SNR_TIERS]
    existing = existing[~existing["metric"].isin(noise_metrics)]
    new_rows = pd.DataFrame([{
        "metric": m,
        "mean": float(per_fold[m].mean()),
        "std": float(per_fold[m].std(ddof=0)),
    } for m in noise_metrics])
    pd.concat([existing, new_rows], ignore_index=True).to_csv(summary_path, index=False)

    print(f"\n[CV 噪声] e2e 汇总 (mean +/- std):")
    for m in noise_metrics:
        print(f"  {m}: {per_fold[m].mean():.4f} +/- {per_fold[m].std(ddof=0):.4f}")
    print(f"对比 LightGBM: clean=85.5%, 5dB=?, 0dB=?, -5dB=?")
    print(f"对比旧版 YAMNet (冻结): clean=1.91%, 5dB=0.42%, 0dB=0.37%, -5dB=0.12%")


def plot_noise_decay(noise_results, save_path=None):
    """noise_results: dict[模型名] = {"clean":acc, "5dB":acc, "0dB":acc, "-5dB":acc}。"""
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
    path = save_path or os.path.join(cfg.OUT_DIR, "noise_robustness.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图] 噪声衰减曲线已存: {path}")
    return path


# ============================================================
# 10. 推理速度与显存测量 (与 LightGBM cell-4 对称)
# ============================================================
def measure_inference(n_samples=50, n_warmup=5, n_folds=cfg.N_FOLDS, cfg_obj=cfg):
    """测量端到端模型推理延迟与显存 (逐条预测, 5 折 mean +/- std)。"""
    print("=" * 60)
    print("  YAMNet 端到端推理速度与显存测量")
    print("=" * 60)

    year2root_m, csv_paths_m = scan_kaggle_inputs(cfg_obj.INPUT_BASE)
    df_test = pd.read_csv(csv_paths_m[cfg_obj.TEST_CSV])
    all_roots = list(year2root_m.values())

    n_test = min(n_samples, len(df_test))
    print(f"[推理测量] e2e 模型, 测试集取前 {n_test} 条, {n_folds} 折")

    label_map = json.loads(Path(cfg_obj.LABEL_MAP_PATH).read_text(encoding="utf-8"))
    num_classes = len(label_map["label2idx"])

    # 预加载波形
    print(f"[推理测量] 预加载 {n_test} 条波形 ...")
    waveforms = []
    for i in range(n_test):
        path = find_audio_path(df_test.iloc[i], year2root_m, all_roots)
        if path is None:
            path = ""  # 兜底
        waveforms.append(load_waveform(path) if path else np.zeros(int(cfg_obj.SR * cfg_obj.CLIP_SECONDS), dtype=np.float32))

    fold_details = []
    for fold in range(1, n_folds + 1):
        fold_dir = Path(cfg_obj.OUT_DIR) / f"fold{fold}"
        weights_path = fold_dir / "best_weights.weights.h5"
        keras_path = fold_dir / "yamnet_e2e_model.keras"
        if not weights_path.exists() and not keras_path.exists():
            print(f"[跳过] fold{fold} 模型不存在")
            continue

        model = _load_e2e_model(fold_dir, num_classes, cfg_obj)
        print(f"\n[推理测量] fold{fold} ({n_warmup} 预热 + {n_test} 正式) ...")

        for _ in range(n_warmup):
            _ = np.argmax(model.predict(waveforms[0][None, :], verbose=0), axis=1)[0]

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
        print(f"  fold{fold}: {arr.mean():.1f} +/- {arr.std():.1f} ms/条")

    if not fold_details:
        print("[推理测量] 无可用模型")
        return

    means = np.array([d["mean_ms"] for d in fold_details])
    print(f"\n{'='*60}")
    print(f"[e2e 推理速度] {len(fold_details)} 折汇总: {means.mean():.1f} +/- {means.std():.1f} ms/条")
    print(f"[对比 LightGBM] 端到端 ~特征+预测")
    print(f"[对比旧版 YAMNet (冻结)] 86.0 +/- 0.6 ms/条")
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

    # 保存
    details_csv = os.path.join(cfg_obj.OUT_DIR, "inference_details.csv")
    with open(details_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "mean_ms", "std_ms", "min_ms", "max_ms"])
        w.writeheader()
        for d in fold_details:
            w.writerow(d)
    print(f"[推理测量] 逐折细节已存: {details_csv}")

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
    out_csv = os.path.join(cfg_obj.OUT_DIR, "inference_metrics.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metrics.keys())
        w.writeheader()
        w.writerow(metrics)
    print(f"[推理测量] 汇总已存: {out_csv}")


# ============================================================
# 11. 主入口: 顺序执行训练 -> 噪声评估 -> 推理测量
# ============================================================
# 与 LightGBm notebook 多 cell 的对应关系:
#   LightGBM cell-2 (训练)            -> 本文件 main_cv_e2e()
#   LightGBM cell-3 (噪声)            -> 本文件 main_noise_eval()
#   LightGBM cell-4 (推理)            -> 本文件 measure_inference()
# 本文件合并为单次执行, 三步顺序不可颠倒: 噪声/推理依赖训练产物。

def main():
    """顺序执行: 5 折训练 -> 5 折噪声评估 -> 推理速度测量。"""
    t0 = time.time()
    print("\n" + "#" * 60)
    print("# 阶段 1/3: 端到端 5 折训练")
    print("#" * 60 + "\n")
    main_cv_e2e()

    print("\n" + "#" * 60)
    print("# 阶段 2/3: 端到端噪声鲁棒性评估")
    print("#" * 60 + "\n")
    main_noise_eval()

    print("\n" + "#" * 60)
    print("# 阶段 3/3: 端到端推理速度与显存测量")
    print("#" * 60 + "\n")
    measure_inference()

    elapsed = (time.time() - t0) / 60
    print(f"\n[完成] 全流程用时 {elapsed:.1f} 分钟")
    print(f"[产物] {cfg.OUT_DIR}")
    print(f"  - cv_per_fold.csv          5 折 clean accuracy")
    print(f"  - cv_noise_per_fold.csv    5 折噪声档 accuracy")
    print(f"  - cv_summary.csv           汇总 (mean +/- std)")
    print(f"  - inference_metrics.csv    推理速度 + 显存")
    print(f"  - inference_details.csv     逐折推理细节")
    print(f"  - fold{{N}}/                各折模型权重 + 预测")


if __name__ == "__main__":
    main()
