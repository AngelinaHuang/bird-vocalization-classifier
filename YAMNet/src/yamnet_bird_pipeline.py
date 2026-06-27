"""
YAMNet 迁移学习 —— 鸟鸣分类管道骨架
负责人: Wenjuan Huang

整体思路 (大白话):
  YAMNet 是 Google 训练好的"通用声音识别大脑", 它能把一段声音变成一个
  1024 维的"声音指纹向量"(embedding)。我们借用它, 再接一个小型分类头,
  让它学会区分我们这 86 种鸟。

  流程: 音频文件 -> 16kHz 单声道波形 -> YAMNet 提取 embedding -> 缓存 ->
        训练一个小神经网络分类头(输出 86 类) -> 保存模型

注意: 这是"预计算 embedding + 训练分类头"的轻量做法(也是 YAMNet 官方教程的做法)。
      对本作业的算力(普通笔记本)最友好。proposal 里提到的"解冻顶层卷积块做
      真正的 fine-tune"属于进阶, 代码里留了 TODO 注释, 等基础版跑通再升级。
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 配置区 —— 你只需要改这里的路径和参数
# ============================================================
class Config:
    # YAMNet 强制要求: 16kHz, 单声道, float32, 幅值归一化到 [-1, 1]
    SAMPLE_RATE = 16000
    # 每条音频统一取固定长度(秒)。太长截断取中段, 太短末尾补零
    CLIP_SECONDS = 5.0

    # ---- 数据路径 ----
    # 把你抽来的样本按 物种/音频文件 的文件夹结构放:
    #   data/raw/american_robin/xxx.wav
    #   data/raw/northern_cardinal/yyy.wav
    RAW_DATA_DIR = Path("../data/raw")          # TODO: 等 Jianan 出正式数据后改成他的目录

    # ---- 输出路径 ----
    OUT_DIR = Path("../outputs/yamnet")
    MODEL_PATH = OUT_DIR / "yamnet_bird_model.keras"
    LABEL_MAP_PATH = OUT_DIR / "label_map.json"
    EMBED_CACHE = OUT_DIR / "embeddings.npz"    # 预计算 embedding 缓存, 避免重复跑 YAMNet

    # ---- YAMNet 模型 (首次运行会联网下载, 约 ~17MB) ----
    YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

    # ---- 训练参数 ----
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    DROPOUT = 0.3
    SEED = 42


cfg = Config()
cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 数据加载: 从文件夹读出 (音频路径, 物种标签)
# ============================================================
def build_dataframe_from_folders(raw_dir: Path) -> pd.DataFrame:
    """
    扫描 data/raw/<物种名>/xxx.{wav,mp3,flac}, 返回一张表:
        filepath | label
    每个物种 = 一个子文件夹名 = 一个标签。
    """
    records = []
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"找不到数据目录 {raw_dir.resolve()}。请先按上面的文件夹结构放几份样本。"
        )
    for species_dir in sorted(raw_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        for ext in ("*.wav", "*.WAV", "*.mp3", "*.flac"):
            for f in species_dir.glob(ext):
                records.append({"filepath": str(f), "label": species_dir.name})
    if not records:
        raise RuntimeError(f"在 {raw_dir} 下没有找到任何音频, 检查文件夹结构。")
    df = pd.DataFrame(records)
    print(f"[数据] 共找到 {len(df)} 条音频, 涉及 {df['label'].nunique()} 个物种。")
    print(df["label"].value_counts())
    return df


# ============================================================
# 3. 音频预处理: 任意格式 -> 16kHz 单声道 float32 [-1,1]
# ============================================================
def load_waveform(path: str, sr: int = cfg.SAMPLE_RATE,
                  clip_seconds: float = cfg.CLIP_SECONDS) -> np.ndarray:
    """
    用 librosa 读音频, 重采样到 16kHz, 转单声道, 固定长度, 归一化。
    返回 shape=[samples] 的 float32 数组。
    """
    import librosa  # 局部导入, 避免没装时整个脚本起不来
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * clip_seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))          # 太短补零
    else:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]                  # 太长取中段
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak).astype(np.float32)


# ============================================================
# 4. YAMNet 提取 embedding 并缓存
# ============================================================
def load_yamnet():
    """加载 YAMNet 模型。返回的模型调用方式: scores, embeddings, spectrogram = model(waveform)"""
    print(f"[YAMNet] 正在加载模型: {cfg.YAMNET_HANDLE} (首次会下载) ...")
    yamnet = hub.load(cfg.YAMNET_HANDLE)
    print("[YAMNet] 加载完成。")
    return yamnet


def extract_embedding(yamnet, waveform: np.ndarray) -> np.ndarray:
    """
    一条音频 -> 一个 1024 维向量。
    YAMNet 对一条音频会输出多帧 embedding (每 0.48s 一帧),
    我们对帧维度取平均, 得到这条音频的整体指纹。
    """
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform_tf)          # embeddings: [num_frames, 1024]
    return tf.reduce_mean(embeddings, axis=0).numpy()  # -> [1024]


def build_or_load_embeddings(df: pd.DataFrame, label2idx: dict):
    """
    给所有音频算 embedding, 存到 npz 缓存。下次直接读缓存, 不再跑 YAMNet。
    """
    if cfg.EMBED_CACHE.exists():
        print(f"[Embedding] 读缓存: {cfg.EMBED_CACHE}")
        data = np.load(cfg.EMBED_CACHE, allow_pickle=True)
        return data["X"], data["y"]

    print(f"[Embedding] 首次计算, 共 {len(df)} 条, 请稍候 ...")
    yamnet = load_yamnet()
    X, y = [], []
    for i, row in enumerate(df.itertuples()):
        wf = load_waveform(row.filepath)
        emb = extract_embedding(yamnet, wf)
        X.append(emb)
        y.append(label2idx[row.label])
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(df)}")
    X = np.stack(X).astype(np.float32)              # [N, 1024]
    y = np.array(y, dtype=np.int64)                 # [N]
    np.savez(cfg.EMBED_CACHE, X=X, y=y)
    print(f"[Embedding] 完成, 已缓存到 {cfg.EMBED_CACHE}")
    return X, y


# ============================================================
# 5. 分类头 (小型神经网络)
# ============================================================
def build_classifier(num_classes: int, embedding_dim: int = 1024,
                     dropout: float = cfg.DROPOUT) -> tf.keras.Model:
    """
    YAMNet 输出 1024 维 embedding -> 全连接 256 -> Dropout -> 输出 num_classes 类。
    """
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
# 6. 主流程
# ============================================================
def main():
    # 6.1 读数据
    df = build_dataframe_from_folders(cfg.RAW_DATA_DIR)

    # 6.2 建标签映射 (物种名 <-> 数字下标), 保存 json 供评估/部署用
    classes = sorted(df["label"].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    cfg.LABEL_MAP_PATH.write_text(json.dumps(
        {"label2idx": label2idx, "idx2label": {str(k): v for k, v in idx2label.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[标签] {len(classes)} 类, 映射已存: {cfg.LABEL_MAP_PATH}")

    # 6.3 提取 / 读缓存 embedding
    X, y = build_or_load_embeddings(df, label2idx)

    # 6.4 划分 train / val / test (分层抽样, 保持类别比例)
    #     先切出 test, 再在剩余里切 val
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SPLIT, random_state=cfg.SEED, stratify=y)
    val_ratio = cfg.VAL_SPLIT / (1 - cfg.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, random_state=cfg.SEED, stratify=y_tmp)
    print(f"[划分] train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # 6.5 训练
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
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # 6.6 测试集评估
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[结果] 测试集准确率 = {test_acc:.4f}")
    print(f"[结果] 模型已保存: {cfg.MODEL_PATH}")

    # 6.7 把测试集预测也存一份, 供 unified_evaluation.py 出图用
    np.savez(cfg.OUT_DIR / "test_predictions.npz",
             y_true=y_test,
             y_pred=np.argmax(model.predict(X_test, verbose=0), axis=1),
             classes=np.array(classes))
    print(f"[结果] 测试预测已存: {cfg.OUT_DIR / 'test_predictions.npz'}")
    print("\n下一步: 用 unified_evaluation.py 画混淆矩阵、和其它两个模型对比。")

    # TODO (进阶, 跑通基础版后再做): 把 YAMNet 当作 trainable 的 Keras 层,
    #   用 hub.KerasLayer(cfg.YAMNET_HANDLE, trainable=True) 嵌进端到端模型,
    #   设置差异化学习率 (底层小、新分类头大), 这才对应 proposal 里"解冻顶层卷积块"。
    #   基础版(本文件)对作业已经够用, 先求跑通。


if __name__ == "__main__":
    main()
