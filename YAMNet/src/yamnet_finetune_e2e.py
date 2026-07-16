"""
YAMNet 端到端微调: 选择性解冻顶层 + 差分学习率 + MixUp + 类别权重
负责人: Wenjuan Huang

与 yamnet_bird_pipeline.py 的区别:
  - 旧版: 冻结 YAMNet → 预计算 1024 维嵌入 → 只训练分类头 (Dense 256 → Dense 1229)
  - 本版: YAMNet 作为可训练 Keras 层 → 解冻顶层 → 原始波形直通 → 端到端联合训练

训练策略:
  - 底层卷积 (靠近输入): 冻结 (梯度不 apply), 保留 AudioSet 预训练知识
  - 顶层卷积 (靠近输出): 解冻, 用小学习率 (1e-5) 微调以适应鸟鸣特征
  - 分类头: 全新初始化, 用大学习率 (1e-3) 快速收敛
  - 差分学习率: 通过两个 Adam optimizer 分别 apply_gradients 实现
  - 类别权重: 按各类样本数倒数加权, 缓解长尾不平衡
  - MixUp (alpha=0.2): 每批随机混合两条波形, 标签按比例混合
  - 所有训练步骤均用 @tf.function 加速

选择性解冻实现:
  hub.KerasLayer 不支持逐变量 trainable, 但可以通过梯度级控制实现——
  在训练循环中只为解冻变量 apply_gradients, 冻结变量的梯度不更新。

数据: 与旧版完全一致 (同一 CSV, 同一 5 折划分), 保证公平对比。
输出: 存入 OUT_DIR/e2e/ 下的 fold{N}/ 子目录, 不覆盖旧版产物。

Kaggle 运行方式 (3 个 cell):
  cell1: %run -i src/yamnet_finetune_e2e.py   (训练 + 测试预测, 30-60 min/fold)
  cell2: %run -i src/noise_eval_e2e.py          (噪声评估, 波形直通 e2e 模型)
  cell3: %run -i src/measure_inference_e2e.py   (推理速度 + 显存)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

# 复用管道中的数据加载、音频预处理、Kaggle 挂载扫描等函数
# Kaggle notebook 中 src/ 不在 sys.path, 需手动加入
import sys as _sys
_SRC_DIR = str(Path(__file__).resolve().parent) if "__file__" in dir() else str(Path.cwd() / "src")
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from yamnet_bird_pipeline import (
    Config, load_csv_splits, preflight_report, load_waveform,
    _scan_inputs, _find_csv, print_mounted_inputs,
)


# ============================================================
# 1. 端到端超参数配置
# ============================================================
class FinetuneConfig(Config):
    """继承基础 Config, 覆盖端到端微调特有的超参数。"""

    # 输出子目录: 不覆盖旧版冻结产物
    E2E_OUT_DIR = Config.OUT_DIR / "e2e"

    # 端到端训练超参数 (比旧版更保守, 因为可训练参数多)
    BATCH_SIZE = 8          # YAMNet 前向占显存大, 小 batch 更稳
    EPOCHS = 40             # 端到端收敛更快但容易过拟合, epoch 数适当减少
    HEAD_LR = 1e-3          # 分类头学习率 (大, 快速收敛)
    TOP_LAYER_LR = 1e-5    # 解冻的顶层卷积学习率 (小, 保护预训练知识)

    # 解冻 YAMNet 顶层的层数 (从输出端倒数)
    # YAMNet 基于 MobileNetV1, 共 27 层卷积; 解冻最后 N 层
    # 建议从 6 开始, 过多会导致显存不足 + 过拟合
    UNFREEZE_LAYERS = 6

    # MixUp 增强 (长尾分类关键技巧)
    MIXUP_ALPHA = 0.2       # Beta 分布参数, 0 表示关闭

    # 类别平衡损失: 按各类样本数倒数加权
    USE_CLASS_WEIGHTS = True

    # 音频缓存: 端到端模式仍需读取波形, 缓存避免重复 I/O
    WAVEFORM_CACHE = Config.OUT_DIR / "waveforms_cache.npz"


fcfg = FinetuneConfig()
fcfg.E2E_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 波形缓存 (避免端到端训练时反复读音频)
# ============================================================
def build_waveform_cache(df_train, df_val, df_test, cfg_obj=fcfg):
    """
    为三个 split 预计算波形, 缓存到 npz。端到端训练每 epoch 都要读波形,
    缓存到内存避免反复磁盘 I/O。

    缓存 key: filenames (与 embedding 缓存策略一致, 跨折复用)
    返回 dict: {filename -> waveform} 以及切分后的数组。
    """
    cache_path = cfg_obj.WAVEFORM_CACHE
    fn2wf = {}
    if cache_path.exists():
        print(f"[波形缓存] 读已有: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        for i, fn in enumerate(d["filenames"]):
            fn2wf[str(fn)] = d["waveforms"][i]
        print(f"  命中 {len(fn2wf)} 条")

    def to_records(df):
        return [(str(r["filename"]), str(r["filepath"]), str(r["primary_label"]))
                for _, r in df.iterrows()]

    rec_train = to_records(df_train)
    rec_val = to_records(df_val)
    rec_test = to_records(df_test)
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


def label_to_idx(y_raw, label2idx):
    """将字符串标签转为整数索引。"""
    return np.array([label2idx[str(l)] for l in y_raw], dtype=np.int64)


# ============================================================
# 3. 类别权重计算
# ============================================================
def compute_class_weights(y, num_classes):
    """
    计算类别权重: 尾部类给更大权重 (按各类样本数倒数加权)。
    返回 shape=(num_classes,) 的 float32 数组, 均值为 1。
    """
    from collections import Counter
    counts = Counter(y.tolist() if hasattr(y, 'tolist') else list(y))
    total = len(y)
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        n = counts.get(c, 0)
        weights[c] = total / (num_classes * max(n, 1))
    mean_w = weights.mean()
    weights = weights / mean_w
    return weights


# ============================================================
# 4. 端到端模型构建
# ============================================================
def _identify_yamnet_layer_names(yamnet_layer):
    """
    从 hub.KerasLayer 的变量名中提取卷积层名, 用于选择性解冻。
    MobileNetV1 层命名: conv2d_0 (底层) 到 conv2d_26 (顶层)。
    返回按层索引排序的层名列表, 失败时返回空列表。
    """
    import re
    layer_names = set()
    for v in yamnet_layer.variables:
        parts = v.name.split("/")
        for p in parts:
            if p.startswith("conv2d"):
                layer_names.add(p)
                break
    # 按数字后缀排序
    def _idx(name):
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else 0
    return sorted(layer_names, key=_idx)


def build_e2e_model(num_classes, cfg_obj=fcfg):
    """
    构建端到端模型: 原始波形 -> YAMNet(可训练) -> 帧均值池化 -> 分类头。

    选择性解冻: hub.KerasLayer 不支持逐变量 trainable, 但可以通过
    梯度级控制实现——在训练循环中只为解冻变量 apply_gradients。

    返回 (model, yamnet_layer, trainable_yamnet_vars)
    """
    # 加载 YAMNet 为 Keras 层 (trainable=True 让梯度能流过)
    yamnet_layer = hub.KerasLayer(
        cfg_obj.YAMNET_HANDLE,
        trainable=True,
        arguments={"_squeeze": True},
        name="yamnet_backbone",
    )

    # 构建端到端模型
    input_wav = tf.keras.Input(
        shape=(int(cfg_obj.SAMPLE_RATE * cfg_obj.CLIP_SECONDS),),
        dtype=tf.float32, name="waveform_input")

    # YAMNet 输出: (batch, num_frames, 1024) -> 帧均值 -> (batch, 1024)
    embeddings = yamnet_layer(input_wav)
    if len(embeddings.shape) == 3:
        pooled = tf.keras.layers.GlobalAveragePooling1D(
            name="frame_avg_pool")(embeddings)
    else:
        pooled = embeddings

    # 分类头 (与旧版结构一致, 便于公平对比)
    x = tf.keras.layers.Dense(256, activation="relu", name="head_fc")(pooled)
    x = tf.keras.layers.Dropout(cfg_obj.DROPOUT, name="head_dropout")(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax",
                                  name="predictions")(x)

    model = tf.keras.Model(inputs=input_wav, outputs=output, name="yamnet_e2e")

    # --- 选择性解冻: 识别顶层变量 ---
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

    # 为 evaluate/predict 编译 (训练用 E2ETrainer 的自定义循环)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg_obj.HEAD_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 打印参数统计
    yamnet_trainable_params = sum(int(np.prod(v.shape)) for v in trainable_yamnet_vars)
    yamnet_frozen_params = sum(int(np.prod(v.shape)) for v in all_yamnet_vars) - yamnet_trainable_params
    head_vars = [v for v in model.trainable_variables if v not in all_yamnet_vars]
    head_params = sum(int(np.prod(v.shape)) for v in head_vars)
    print(f"\n[端到端] 参数统计:")
    print(f"  YAMNet 可训练 (顶层):  {yamnet_trainable_params:,}")
    print(f"  YAMNet 冻结 (底层):    {yamnet_frozen_params:,}")
    print(f"  分类头可训练:          {head_params:,}")
    print(f"  总可训练:              {yamnet_trainable_params + head_params:,}")

    return model, yamnet_layer, trainable_yamnet_vars


# ============================================================
# 5. 自定义训练循环 (差分学习率 + MixUp + 类别权重)
# ============================================================
class E2ETrainer:
    """
    自定义训练循环: 支持 MixUp + 差分学习率 + 类别权重。

    差分学习率: YAMNet 解冻变量用 TOP_LAYER_LR (小), 分类头用 HEAD_LR (大)。
    类别权重: 按各类样本数倒数加权, 缓解长尾不平衡。
    MixUp: 每批随机混合两条波形, 标签按比例混合。
    所有训练步骤均用 @tf.function 加速。
    """

    def __init__(self, model, yamnet_layer, trainable_yamnet_vars,
                 num_classes, y_train, cfg_obj=fcfg):
        self.model = model
        self.num_classes = num_classes
        self.cfg = cfg_obj

        # YAMNet 解冻变量 + 分类头变量
        self.yamnet_vars = trainable_yamnet_vars
        self.head_vars = [v for v in model.trainable_variables
                         if v not in yamnet_layer.trainable_variables]
        self.update_vars = self.yamnet_vars + self.head_vars

        # 两个 optimizer (差分学习率)
        self.opt_head = tf.keras.optimizers.Adam(cfg_obj.HEAD_LR)
        self.opt_yamnet = tf.keras.optimizers.Adam(cfg_obj.TOP_LAYER_LR)

        # 类别权重
        if cfg_obj.USE_CLASS_WEIGHTS:
            self.class_weights = compute_class_weights(y_train, num_classes)
            print(f"[训练] 类别权重已计算: min={self.class_weights.min():.3f}, "
                  f"max={self.class_weights.max():.3f}, mean={self.class_weights.mean():.3f}")
        else:
            self.class_weights = None

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(self, X_batch, y_batch, sample_weight):
        """普通训练步 (无 MixUp): 前向 -> 加权损失 -> 差分学习率更新。"""
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
        """MixUp 训练步: 前向 -> 加权 categorical_crossentropy -> 差分学习率更新。"""
        with tf.GradientTape() as tape:
            logits = self.model(X_batch, training=True)
            per_sample = tf.keras.losses.categorical_crossentropy(y_onehot, logits)
            loss = tf.reduce_mean(per_sample * sample_weight)
        grads = tape.gradient(loss, self.update_vars)
        n_yam = len(self.yamnet_vars)
        self.opt_yamnet.apply_gradients(zip(grads[:n_yam], self.yamnet_vars))
        self.opt_head.apply_gradients(zip(grads[n_yam:], self.head_vars))
        # MixUp 下用 one-hot 的 argmax 作为伪标签统计准确率
        self.train_acc.update_state(tf.argmax(y_onehot, axis=-1), logits)
        return loss

    @tf.function
    def val_step(self, X_batch, y_batch):
        """验证步 (不更新参数)。"""
        logits = self.model(X_batch, training=False)
        self.val_acc.update_state(y_batch, logits)

    def fit(self, X_train, y_train, X_val, y_val, weights_path=None):
        """
        自定义训练循环: 支持 MixUp + 类别权重 + 差分学习率 + 学习率衰减。
        weights_path: 最佳权重保存路径。
        """
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

                # 计算样本权重 (类别权重)
                if self.class_weights is not None:
                    sw = self.class_weights[yb].astype(np.float32)
                else:
                    sw = np.ones(len(yb), dtype=np.float32)

                if self.cfg.MIXUP_ALPHA > 0 and len(Xb) > 1:
                    # MixUp (numpy 操作在 @tf.function 外)
                    lam = np.random.beta(self.cfg.MIXUP_ALPHA, self.cfg.MIXUP_ALPHA)
                    idx = np.random.permutation(len(Xb))
                    Xb_mix = (lam * Xb + (1 - lam) * Xb[idx]).astype(np.float32)
                    # 混合标签的 one-hot
                    y_oh = (lam * tf.one_hot(yb, depth=self.num_classes).numpy() +
                            (1 - lam) * tf.one_hot(yb[idx], depth=self.num_classes).numpy()
                           ).astype(np.float32)
                    # 混合样本权重
                    sw_mix = (lam * sw + (1 - lam) * sw[idx]).astype(np.float32)
                    # 调用 @tf.function 加速的训练步
                    loss = self.train_step_mixup(
                        tf.constant(Xb_mix),
                        tf.constant(y_oh),
                        tf.constant(sw_mix),
                    )
                else:
                    loss = self.train_step(
                        tf.constant(Xb),
                        tf.constant(yb),
                        tf.constant(sw),
                    )
                ep_loss += float(loss)

            # 验证
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
                # 每 4 次无改善减一次学习率 (可多次)
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
# 6. 主流程: 5 折端到端微调
# ============================================================
def main_cv_e2e(n_folds=5, cfg_obj=fcfg):
    """
    5 折端到端微调主流程。

    与旧版 main_cv_all_folds() 的区别:
      - 不预计算嵌入缓存, 每批前向过完整 YAMNet
      - 用 E2ETrainer 自定义训练循环 (差分学习率 + MixUp)
      - 输出存入 e2e/fold{N}/ 不覆盖旧版产物
      - 噪声评估复用 noise_robustness_eval.py (改为端到端推理)

    Kaggle 运行: 作为 cell1 执行, 约 30-60 min/fold。
    """
    print("="*60)
    print("  YAMNet 端到端微调 (解冻顶层 + 差分学习率 + MixUp)")
    print("="*60)

    # 扫描 Kaggle 挂载
    print("\n===== 挂载自检 =====")
    print_mounted_inputs(cfg_obj.KAGGLE_INPUT)
    print("====================\n")
    scan = _scan_inputs(cfg_obj.KAGGLE_INPUT)
    csv_paths = scan[2]

    # 共享标签映射 (与旧版一致)
    f1_tr, f1_va = cfg_obj.fold_csvs(1)
    label_sources = [f1_tr, f1_va, cfg_obj.TEST_CSV]
    label_frames = []
    for fname in label_sources:
        p = _find_csv(fname, cfg_obj, scan=csv_paths)
        if p is None:
            raise FileNotFoundError(f"CSV not found: {fname}")
        label_frames.append(pd.read_csv(p, usecols=["primary_label"])["primary_label"])
    classes = sorted(pd.concat(label_frames).astype(str).unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    cfg_obj.LABEL_MAP_PATH.write_text(json.dumps(
        {"label2idx": label2idx, "idx2label": {str(k): v for k, v in idx2label.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[标签] {len(classes)} 类")

    clean_accs = []
    for fold in range(1, n_folds + 1):
        print(f"\n{'#'*20} FOLD {fold}/{n_folds} {'#'*20}")
        tf.random.set_seed(cfg_obj.SEED + fold)
        np.random.seed(cfg_obj.SEED + fold)

        tr_csv, va_csv = cfg_obj.fold_csvs(fold)
        df_train, df_val, df_test, missing = load_csv_splits(
            cfg_obj, train_csv=tr_csv, val_csv=va_csv, scan=scan)
        df_train, df_val, df_test = preflight_report(missing, df_train, df_val, df_test)
        if len(df_train) == 0 or len(df_test) == 0:
            raise RuntimeError(f"fold{fold} 剔除缺失后数据为空")

        # 预计算波形缓存 (跨折复用)
        X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw, test_filenames, _ = \
            build_waveform_cache(df_train, df_val, df_test, cfg_obj)
        y_train = label_to_idx(y_train_raw, label2idx)
        y_val = label_to_idx(y_val_raw, label2idx)
        y_test = label_to_idx(y_test_raw, label2idx)
        print(f"[划分] fold{fold}: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

        # 构建端到端模型 (返回 model, yamnet_layer, trainable_yamnet_vars)
        model, yamnet_layer, trainable_yamnet_vars = build_e2e_model(len(classes), cfg_obj)
        fold_out = cfg_obj.E2E_OUT_DIR / f"fold{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        # 训练
        trainer = E2ETrainer(model, yamnet_layer, trainable_yamnet_vars,
                            len(classes), y_train, cfg_obj)
        weights_path = fold_out / "best_weights.weights.h5"
        print(f"[训练] fold{fold} 开始 ...")
        t0 = time.time()
        trainer.fit(X_train, y_train, X_val, y_val, weights_path=weights_path)
        elapsed = (time.time() - t0) / 60
        print(f"[训练] fold{fold} 完成, 用时 {elapsed:.1f} 分钟")

        # 测试集评估 (分批避免 GPU OOM, 端到端前向占显存大)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0, batch_size=8)
        print(f"[结果] fold{fold} 测试集准确率 = {test_acc:.4f}")
        clean_accs.append(float(test_acc))

        # 保存模型权重和架构 (不保存完整 .keras 因 hub.KerasLayer 加载时需联网)
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
    e2e_cv = pd.DataFrame({"fold": range(1, n_folds+1), "clean_acc": clean_arr})
    e2e_cv.to_csv(cfg_obj.E2E_OUT_DIR / "cv_per_fold.csv", index=False)
    summary = pd.DataFrame([{
        "metric": "clean_acc",
        "mean": float(clean_arr.mean()),
        "std": float(clean_arr.std(ddof=0)),
    }])
    summary.to_csv(cfg_obj.E2E_OUT_DIR / "cv_summary.csv", index=False)
    print(f"\n[CV] e2e clean accuracy: {clean_arr.mean():.4f} ± {clean_arr.std(ddof=0):.4f}")
    print(f"[CV] 对比旧版 (冻结): 1.91% ± 0.33%")
    print(f"\n下一步: 运行 noise_eval_e2e.py 做噪声评估 (cell2)")


def main():
    main_cv_e2e()


if __name__ == "__main__":
    main()
