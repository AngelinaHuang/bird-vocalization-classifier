"""
YAMNet 迁移学习鸟鸣分类管道
负责人: Wenjuan Huang

利用 Google 预训练的 YAMNet 将音频映射为 1024 维嵌入, 在其上训练一个轻量
分类头, 完成 1229 类鸟鸣识别。流程: 音频 -> 16kHz 单声道波形 -> YAMNet 嵌入
-> 缓存 -> 训练分类头 -> 保存模型。

数据来源: 读取小组切分的 5 折交叉验证 CSV (ml_cv_fold1..5_train/val) 与
留出测试集 ml_test, 在挂载的 BirdCLEF 年度数据集中定位音频, 适用于 Kaggle。
默认入口 main() 跑 5 折取均值±标准差; main_csv() 仍可单折手动验证。

采用"预计算嵌入 + 训练分类头"的轻量方案 (与 YAMNet 官方教程一致), 对算力
最为友好; proposal 中所述的解冻顶层卷积块属于进阶, 见文末 TODO。
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


# ============================================================
# 1. 配置
# ============================================================
class Config:
    # YAMNet 输入要求: 16kHz、单声道、float32, 幅值归一化到 [-1, 1]
    SAMPLE_RATE = 16000
    # 统一音频长度 (秒); 过长取中段, 过短末尾补零
    CLIP_SECONDS = 5.0

    # CSV 路径: Kaggle 上将 ml_*.csv 作为名为 bird-metadata 的数据集挂载;
    # 本地默认指向正式数据目录
    CSV_DIR = Path("/kaggle/input/bird-metadata") if Path("/kaggle/input").exists() \
        else Path("../../data/data")
    # 以下为单折默认值 (供 main_csv 使用); 5 折模式下由 fold_csvs() 按折号生成
    TRAIN_CSV = "ml_cv_fold1_train.csv"   # fold1 训练集 (单折默认)
    VAL_CSV = "ml_cv_fold1_val.csv"       # fold1 验证集 (单折默认)
    TEST_CSV = "ml_test.csv"              # 留出测试集 (5 折共用, 不随折变化)

    @staticmethod
    def fold_csvs(fold: int):
        """返回 (train_csv, val_csv) 文件名, 如 fold=3 -> (ml_cv_fold3_train.csv, ...)。"""
        return (f"ml_cv_fold{fold}_train.csv", f"ml_cv_fold{fold}_val.csv")

    def fold_dir(self, fold: int):
        """返回某折的输出子目录 (调用方负责 mkdir)。"""
        return self.OUT_DIR / f"fold{fold}"

    # 音频发现: 扫描 KAGGLE_INPUT 下各年度数据集子目录
    KAGGLE_INPUT = Path("/kaggle/input")
    # 年度数据集中音频目录的可能命名 (不同年份不一致)
    AUDIO_ROOT_CANDIDATES = ("train_audio", "train_short_audio")

    # 输出目录: Kaggle 写 /kaggle/working (随 notebook 持久化); 本地写 ../outputs
    @staticmethod
    def _default_out_dir():
        if Path("/kaggle/working").exists():
            return Path("/kaggle/working/yamnet")
        return Path("../outputs/yamnet")

    OUT_DIR = _default_out_dir.__func__()
    MODEL_PATH = OUT_DIR / "yamnet_bird_model.keras"
    LABEL_MAP_PATH = OUT_DIR / "label_map.json"
    EMBED_CACHE = OUT_DIR / "embeddings.npz"   # 嵌入缓存, 避免重复运行 YAMNet
    NOISE_EMBED_CACHE = OUT_DIR / "noise_embeddings.npz"   # 噪声嵌入缓存 (5 折共享)

    # YAMNet 模型 (首次运行联网下载, 约 17MB)
    YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

    # 训练超参数
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    DROPOUT = 0.3
    SEED = 42


cfg = Config()
cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Kaggle 环境与音频发现
# ============================================================
def _is_kaggle() -> bool:
    return Path("/kaggle/input").exists() or bool(os.environ.get("KAGGLE_CONTAINER_NAME"))


def print_mounted_inputs(kaggle_input: Path = cfg.KAGGLE_INPUT, max_per_dir: int = 8):
    """列出 /kaggle/input 下挂载的数据集及其一层子目录, 用于排查。"""
    kaggle_input = Path(kaggle_input)
    if not kaggle_input.exists():
        print(f"[挂载] {kaggle_input} 不存在, 可能不在 Kaggle 环境或未挂载数据集。")
        return
    entries = sorted(p for p in kaggle_input.iterdir() if p.is_dir())
    if not entries:
        print(f"[挂载] {kaggle_input} 下无子目录, 请在 notebook 右侧 Add Input 挂载数据集。")
        return
    print(f"[挂载] {kaggle_input} 下发现 {len(entries)} 个数据集:")
    for d in entries:
        sub = [c.name for c in d.iterdir() if c.is_dir()][:max_per_dir]
        print(f"  - {d.name}  子目录: {sub}")


def _scan_inputs(kaggle_input: Path = cfg.KAGGLE_INPUT):
    """
    遍历 /kaggle/input 一次, 同时收集音频根目录与目标 CSV 路径。

    返回 (year2root, unknown_roots, csv_paths):
      - year2root:    {年份: 音频根路径}
      - unknown_roots: 无法识别年份的音频根
      - csv_paths:    {csv 文件名: 路径}

    遇到音频根目录后剪枝, 不再深入遍历数十万个音频文件, 因此扫描很快,
    且不受数据集嵌套深度影响。
    """
    kaggle_input = Path(kaggle_input)
    year2root = {}
    unknown_roots = []
    csv_paths = {}
    # 扫描收录名单: 单折默认 (fold1 train/val + test) 外, 还要包含 5 折全部
    # train/val CSV, 否则 fold2~5 即使已挂载也会被扫描器跳过而报 "找不到 CSV"。
    target_csvs = {cfg.TRAIN_CSV, cfg.VAL_CSV, cfg.TEST_CSV}
    for _fold in range(1, 6):
        target_csvs.update(cfg.fold_csvs(_fold))
    if not kaggle_input.exists():
        return year2root, unknown_roots, csv_paths

    for root, dirs, files in os.walk(kaggle_input):
        base = os.path.basename(root)
        if base in cfg.AUDIO_ROOT_CANDIDATES:
            rp = Path(root)
            m = re.search(r"(20\d{2})", root)
            if m:
                year = int(m.group(1))
                year2root.setdefault(year, rp)
            else:
                if rp not in unknown_roots:
                    unknown_roots.append(rp)
            dirs[:] = []          # 剪枝: 不进入音频文件目录
            continue
        for f in files:
            if f in target_csvs and f not in csv_paths:
                csv_paths[f] = Path(root) / f
    return year2root, unknown_roots, csv_paths


def _find_csv(filename: str, cfg_obj=cfg, scan=None) -> Path:
    """在扫描结果中查找 CSV; scan 为 None 时重新扫描一次。"""
    if scan is None:
        _, _, scan_csvs = _scan_inputs(cfg_obj.KAGGLE_INPUT)
        scan = scan_csvs
    cand = Path(cfg_obj.CSV_DIR) / filename
    if cand.exists():
        return cand
    return scan.get(filename)


def resolve_kaggle_audio_roots(kaggle_input: Path, scan=None) -> dict:
    """返回 {年份: 音频根路径}; scan 为 None 时重新扫描。"""
    if scan is None:
        year2root, _, _ = _scan_inputs(kaggle_input)
        return year2root
    if isinstance(scan, tuple):
        return scan[0]
    return scan


def parse_source_years(source_year_str) -> list:
    """
    解析 source_year 字段, 返回年份列表 (整数, 降序, 最新优先)。
    例: "2025"       -> [2025]
        "2025,2026"  -> [2026, 2025]
    无效或空值返回 []。
    """
    if source_year_str is None or (isinstance(source_year_str, float) and pd.isna(source_year_str)):
        return []
    years = re.findall(r"20\d{2}", str(source_year_str))
    years = sorted({int(y) for y in years}, reverse=True)
    return years


def resolve_audio_path(row, audio_roots: dict):
    """
    依据一行的 filename / primary_label / source_year, 在年度音频根中按候选
    路径逐个试探, 返回首个存在的路径; 全部未命中返回 None。

    候选顺序 (各年份按最新优先):
      1) {root}/{filename}                 命中 2022+ 及平铺命名
      2) {root}/{primary_label}/{basename} 命中 2021 (按 label 分目录)
      3) {root}/{basename}                 平铺兜底
    """
    filename = str(row["filename"])
    primary_label = str(row["primary_label"])
    basename = os.path.basename(filename)
    for year in parse_source_years(row.get("source_year")):
        audio_root = audio_roots.get(year)
        if audio_root is None:
            continue
        candidates = [
            Path(audio_root) / filename,
            Path(audio_root) / primary_label / basename,
            Path(audio_root) / basename,
        ]
        for c in candidates:
            if c.exists():
                return c
    # source_year 缺失或全部未命中: 用所有已知年份兜底 (降序)
    for year in sorted(audio_roots.keys(), reverse=True):
        audio_root = audio_roots[year]
        candidates = [
            Path(audio_root) / filename,
            Path(audio_root) / primary_label / basename,
            Path(audio_root) / basename,
        ]
        for c in candidates:
            if c.exists():
                return c
    return None


# ============================================================
# 3. CSV 数据加载
# ============================================================
def load_csv_splits(cfg_obj=cfg, train_csv=None, val_csv=None, test_csv=None, scan=None):
    """
    读取 train/val/test 三个 CSV 并解析每行音频路径。

    train_csv/val_csv/test_csv 为 None 时用 cfg_obj 默认值 (fold1), 否则用传入
    值——5 折模式下 main_cv_all_folds 据此切换各折 CSV。
    scan 为已扫描结果 (year2root, unknown_roots, csv_paths) 时直接复用, 跳过
    重复 os.walk 与挂载自检打印, 供 5 折循环避免刷屏; None 时重新扫描并打印。

    返回 (df_train, df_val, df_test, missing_report), 其中 missing_report
    为 {split: [缺失 filename]}。缺失音频的行不在此处剔除, 仅标记, 由
    preflight_report 统一处理。
    """
    if scan is None:
        print("\n===== 挂载自检 =====")
        print_mounted_inputs(cfg_obj.KAGGLE_INPUT)
        print("====================\n")
        # 一次扫描同时拿到音频根与 CSV 路径 (剪枝, 很快)
        audio_roots, unknown_roots, csv_paths = _scan_inputs(cfg_obj.KAGGLE_INPUT)
    else:
        audio_roots, unknown_roots, csv_paths = scan
    if audio_roots:
        print(f"[音频发现] 共 {len(audio_roots)} 个年度音频根:")
        for y in sorted(audio_roots.keys()):
            print(f"  {y} -> {audio_roots[y]}")
    if unknown_roots:
        print(f"[音频发现] 另有 {len(unknown_roots)} 个无年份的音频根 (将作兜底):")
        for r in unknown_roots:
            print(f"  ? -> {r}")
    if not audio_roots and not unknown_roots:
        print("[警告] 未在 KAGGLE_INPUT 下发现年度音频目录 (train_audio / train_short_audio)。\n"
              "  可能原因: (1) 未挂载 birdclef-2021…2026 数据集;\n"
              "           (2) 音频目录命名不同, 需在 AUDIO_ROOT_CANDIDATES 中补充。")
    # 无年份的根一并纳入, key 为 -1, resolve_audio_path 的兜底循环会试到
    for r in unknown_roots:
        audio_roots.setdefault(-1, r)

    splits = {
        "train": train_csv or cfg_obj.TRAIN_CSV,
        "val": val_csv or cfg_obj.VAL_CSV,
        "test": test_csv or cfg_obj.TEST_CSV,
    }
    dfs = {}
    missing = {}
    not_found = []
    for name, fname in splits.items():
        path = _find_csv(fname, cfg_obj, scan=csv_paths)
        if path is None:
            not_found.append(fname)
            continue
        print(f"[CSV] {name}: 读取 {path}")
        df = pd.read_csv(path)
        resolved = []
        miss_list = []
        for _, row in df.iterrows():
            p = resolve_audio_path(row, audio_roots) if audio_roots else None
            if p is None:
                miss_list.append(row["filename"])
            resolved.append(str(p) if p is not None else None)
        df = df.copy()
        df["filepath"] = resolved
        dfs[name] = df
        missing[name] = miss_list
        print(f"  {name}: {len(df)} 行, 找到 {len(df) - len(miss_list)}, 缺失 {len(miss_list)}")
    if not_found:
        raise FileNotFoundError(
            f"找不到 CSV: {not_found}\n"
            f"  请确认 ml_cv_fold1..5_train/val.csv / ml_test.csv "
            f"已作为 Kaggle 数据集挂载 (名称任意, 会自动扫描)。"
        )
    return dfs["train"], dfs["val"], dfs["test"], missing


def preflight_report(missing: dict, df_train, df_val, df_test):
    """打印缺失情况报告, 并从各 split 中剔除缺失音频的行。"""
    print("\n===== 预检报告 =====")
    for name, miss in missing.items():
        print(f"  {name}: 缺失 {len(miss)} 条")
        if miss:
            print(f"    缺失示例 (前 5): {miss[:5]}")
    print("====================\n")

    def drop_missing(df, miss):
        miss_set = set(miss)
        return df[~df["filename"].isin(miss_set)].reset_index(drop=True)

    df_train = drop_missing(df_train, missing["train"])
    df_val = drop_missing(df_val, missing["val"])
    df_test = drop_missing(df_test, missing["test"])
    print(f"[预检后] train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    return df_train, df_val, df_test


# ============================================================
# 4. 音频预处理: 统一为 16kHz 单声道 float32 [-1,1]
# ============================================================
def load_waveform(path: str, sr: int = cfg.SAMPLE_RATE,
                  clip_seconds: float = cfg.CLIP_SECONDS) -> np.ndarray:
    """
    读取音频, 重采样至 16kHz, 转单声道, 固定长度, 峰值归一化。
    返回 shape=[samples] 的 float32 数组。
    """
    import librosa  # 局部导入, 避免环境未安装时脚本无法加载
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * clip_seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))          # 补零
    else:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]                  # 取中段
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak).astype(np.float32)


# ============================================================
# 5. YAMNet 嵌入提取与缓存
# ============================================================
def load_yamnet():
    """加载 YAMNet; 调用方式: scores, embeddings, spectrogram = model(waveform)。"""
    print(f"[YAMNet] 加载模型: {cfg.YAMNET_HANDLE} (首次会下载) ...")
    yamnet = hub.load(cfg.YAMNET_HANDLE)
    print("[YAMNet] 加载完成。")
    return yamnet


def extract_embedding(yamnet, waveform: np.ndarray) -> np.ndarray:
    """
    将一条音频转为 1024 维向量。YAMNet 按每 0.48s 输出一帧嵌入, 此处对帧
    维度取平均, 得到该条音频的整体表示。
    """
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform_tf)          # [num_frames, 1024]
    return tf.reduce_mean(embeddings, axis=0).numpy()  # -> [1024]


def build_embeddings_for_splits(df_train, df_val, df_test, label2idx, cfg_obj=cfg):
    """
    为三个 split 计算嵌入。缓存按 filename 索引, 与行顺序无关, 可跨运行复用。

    返回 X_train,y_train, X_val,y_val, X_test,y_test, test_filenames。
    test_filenames 供噪声测试按相同顺序取用干净嵌入。

    缓存策略: 存 filenames / X / y; 命中部分则只补算缺失项并回写。
    """
    def to_records(df):
        return [(str(r["filename"]), str(r["filepath"]), str(r["primary_label"]))
                for _, r in df.iterrows()]

    rec_train = to_records(df_train)
    rec_val = to_records(df_val)
    rec_test = to_records(df_test)
    all_recs = rec_train + rec_val + rec_test

    # filename 全局去重 (三 split 文件名互不相交, 保留首次出现)
    seen = set()
    unique_recs = []
    for fn, fp, lab in all_recs:
        if fn not in seen:
            seen.add(fn)
            unique_recs.append((fn, fp, lab))

    cache_path = cfg_obj.EMBED_CACHE
    fn2emb = {}
    fn2y = {}
    if cache_path.exists():
        print(f"[Embedding] 读缓存: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cached_fn = data["filenames"]
        cached_X = data["X"]
        cached_y = data["y"]
        for i, fn in enumerate(cached_fn):
            fn2emb[str(fn)] = cached_X[i]
            fn2y[str(fn)] = int(cached_y[i])
        print(f"  缓存命中 {len(fn2emb)} 条")

    need = [(fn, fp, lab) for fn, fp, lab in unique_recs
            if fn not in fn2emb]
    if need:
        print(f"[Embedding] 需补算 {len(need)} 条 ...")
        yamnet = load_yamnet()
        for i, (fn, fp, lab) in enumerate(need):
            wf = load_waveform(fp)
            emb = extract_embedding(yamnet, wf)
            fn2emb[fn] = emb
            fn2y[fn] = label2idx[lab]
            if (i + 1) % 50 == 0 or (i + 1) == len(need):
                print(f"  [Embedding] 已补算 {i+1}/{len(need)}")
        all_fn = list(fn2emb.keys())
        X_all = np.stack([fn2emb[f] for f in all_fn]).astype(np.float32)
        y_all = np.array([fn2y[f] for f in all_fn], dtype=np.int64)
        np.savez(cache_path,
                 filenames=np.array(all_fn), X=X_all, y=y_all)
        print(f"[Embedding] 已回写缓存: {cache_path}")
    else:
        print("[Embedding] 全部命中缓存, 无需补算")

    def slice_xy(records):
        Xs = np.stack([fn2emb[fn] for fn, _, _ in records]).astype(np.float32)
        ys = np.array([label2idx[lab] for _, _, lab in records], dtype=np.int64)
        return Xs, ys

    X_train, y_train = slice_xy(rec_train)
    X_val, y_val = slice_xy(rec_val)
    X_test, y_test = slice_xy(rec_test)
    test_filenames = [fn for fn, _, _ in rec_test]
    return X_train, y_train, X_val, y_val, X_test, y_test, test_filenames


# ============================================================
# 6. 分类头
# ============================================================
def build_classifier(num_classes: int, embedding_dim: int = 1024,
                     dropout: float = cfg.DROPOUT) -> tf.keras.Model:
    """1024 维嵌入 -> 全连接 256 -> Dropout -> softmax 输出 num_classes 类。"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="yamnet_bird_head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ============================================================
# 7. 主流程
# ============================================================
def main_csv():
    """CSV 模式: 正式数据, Kaggle 直接运行。"""
    # 读取 CSV 并发现音频
    df_train, df_val, df_test, missing = load_csv_splits(cfg)

    # 预检并剔除缺失音频的行
    df_train, df_val, df_test = preflight_report(missing, df_train, df_val, df_test)
    if len(df_train) == 0 or len(df_test) == 0:
        raise RuntimeError("剔除缺失后训练/测试集为空, 请检查音频挂载。")

    # 标签映射: 用 train+val+test 的 primary_label 全集, 保证各类均有下标
    all_labels = pd.concat([df_train["primary_label"],
                            df_val["primary_label"],
                            df_test["primary_label"]]).astype(str).unique().tolist()
    classes = sorted(all_labels)
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    cfg.LABEL_MAP_PATH.write_text(json.dumps(
        {"label2idx": label2idx, "idx2label": {str(k): v for k, v in idx2label.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[标签] {len(classes)} 类, 映射已存: {cfg.LABEL_MAP_PATH}")

    # 提取/读取嵌入 (按 filename 索引, 跨 split 复用)
    X_train, y_train, X_val, y_val, X_test, y_test, test_filenames = \
        build_embeddings_for_splits(df_train, df_val, df_test, label2idx)
    print(f"[划分] train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # 训练
    model = build_classifier(num_classes=len(classes))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            cfg.MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]
    print("[训练] 开始 ...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # 测试集评估
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[结果] 测试集准确率 = {test_acc:.4f}")
    print(f"[结果] 模型已保存: {cfg.MODEL_PATH}")

    # 保存测试预测与文件名顺序, 供 unified_evaluation 与噪声测试使用
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    np.savez(cfg.OUT_DIR / "test_predictions.npz",
             y_true=y_test,
             y_pred=y_pred,
             classes=np.array(classes),
             test_filenames=np.array(test_filenames))
    print(f"[结果] 测试预测已存: {cfg.OUT_DIR / 'test_predictions.npz'}")
    print("\n下一步: 用 unified_evaluation.py 画混淆矩阵, 并与其它两个模型对比。")


def main_cv_all_folds(n_folds: int = 5):
    """
    5 折交叉验证: 依次用 ml_cv_fold1..5_train/val 训练, 共享 ml_test 作测试,
    汇总 clean 准确率的 mean±std。

    embedding 缓存在各折间复用 (5 折样本同为 ml_train 的 4780 条重切, 缓存已覆盖
    全部 5976 条), 故 fold2..5 只 slice 不补算 YAMNet。每折模型与测试预测存入
    fold{N}/ 子目录, 供 noise_robustness_eval 按 fold 复用。
    """
    # 扫描一次: 挂载自检 + 音频根 + CSV 路径, 5 折循环复用, 避免重复 os.walk
    print("\n===== 挂载自检 =====")
    print_mounted_inputs(cfg.KAGGLE_INPUT)
    print("====================\n")
    scan = _scan_inputs(cfg.KAGGLE_INPUT)
    csv_paths = scan[2]

    # 共享标签映射: 1229 类 = train+val+test 的 primary_label 并集
    # (各折 train+val 同为 4780 全集, 标签集一致; 用 fold1 的 CSV 取并集即可)
    f1_tr, f1_va = cfg.fold_csvs(1)
    label_sources = [f1_tr, f1_va, cfg.TEST_CSV]
    label_frames = []
    for fname in label_sources:
        p = _find_csv(fname, cfg, scan=csv_paths)
        if p is None:
            raise FileNotFoundError(f"找不到构建标签映射所需的 CSV: {fname}")
        label_frames.append(pd.read_csv(p, usecols=["primary_label"])["primary_label"])
    classes = sorted(pd.concat(label_frames).astype(str).unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    cfg.LABEL_MAP_PATH.write_text(json.dumps(
        {"label2idx": label2idx, "idx2label": {str(k): v for k, v in idx2label.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[标签] {len(classes)} 类, 共享映射已存: {cfg.LABEL_MAP_PATH}")

    clean_accs = []
    for fold in range(1, n_folds + 1):
        print(f"\n########## FOLD {fold}/{n_folds} ##########")
        # 确定性种子: best-effort 缓解 TF 训练不可复现, 使各折结果可比
        tf.random.set_seed(cfg.SEED + fold)
        np.random.seed(cfg.SEED + fold)
        os.environ["PYTHONHASHSEED"] = "0"

        tr_csv, va_csv = cfg.fold_csvs(fold)
        df_train, df_val, df_test, missing = load_csv_splits(
            cfg, train_csv=tr_csv, val_csv=va_csv, scan=scan)
        df_train, df_val, df_test = preflight_report(missing, df_train, df_val, df_test)
        if len(df_train) == 0 or len(df_test) == 0:
            raise RuntimeError(f"fold{fold} 剔除缺失后训练/测试集为空, 请检查音频挂载。")

        X_train, y_train, X_val, y_val, X_test, y_test, test_filenames = \
            build_embeddings_for_splits(df_train, df_val, df_test, label2idx)
        print(f"[划分] fold{fold}: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

        fold_out = cfg.fold_dir(fold)
        fold_out.mkdir(parents=True, exist_ok=True)
        model = build_classifier(num_classes=len(classes))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                fold_out / "yamnet_bird_model.keras", save_best_only=True,
                monitor="val_accuracy"),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ]
        print(f"[训练] fold{fold} 开始 ...")
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"[结果] fold{fold} 测试集准确率 = {test_acc:.4f}")
        clean_accs.append(float(test_acc))

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        np.savez(fold_out / "test_predictions.npz",
                 y_true=y_test,
                 y_pred=y_pred,
                 classes=np.array(classes),
                 test_filenames=np.array(test_filenames))
        print(f"[结果] fold{fold} 模型与测试预测已存: {fold_out}")

    # 汇总 clean 准确率
    clean_arr = np.array(clean_accs)
    cv_per_fold = pd.DataFrame({"fold": range(1, n_folds + 1), "clean_acc": clean_arr})
    cv_per_fold.to_csv(cfg.OUT_DIR / "cv_per_fold.csv", index=False)
    print(f"\n[CV] clean accuracy: {clean_arr.mean():.4f} ± {clean_arr.std():.4f}")
    # cv_summary.csv: cell1 写 clean 行, cell3 (噪声) 追加噪声档行
    summary = pd.DataFrame([{
        "metric": "clean_acc",
        "mean": float(clean_arr.mean()),
        "std": float(clean_arr.std(ddof=0)),
    }])
    summary.to_csv(cfg.OUT_DIR / "cv_summary.csv", index=False)
    print(f"[CV] 汇总已存: {cfg.OUT_DIR / 'cv_summary.csv'}")
    print("\n下一步: 在 cell3 运行 noise_robustness_eval.main_cv() 完成噪声 5 折评估。")


def main():
    main_cv_all_folds()

    # TODO: 将 YAMNet 作为可训练 Keras 层 (hub.KerasLayer(cfg.YAMNET_HANDLE,
    #   trainable=True) 嵌入端到端模型, 采用差异化学习率 (底层小、新分类头大),
    #   对应 proposal 中"解冻顶层卷积块"的完整 fine-tune。基础版对作业已够用。


if __name__ == "__main__":
    main()
