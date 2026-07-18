# 稀有物种音频增强 - 队友接入指南

## 一、项目背景（30 秒看懂）

鸟鸣分类项目，数据集有 **1,126 种鸟**，其中 **159 种**训练样本不足 15 条。
稀有物种样本太少，模型学不到。解决方案：对这 159 种做音频增强，每类补到至少 15 条。

三个算法并行开发，各自在 Kaggle 独立跑：
- **YAMNet 端到端**（波形->分类）— 增强已集成完毕，不用改
- **LightGBM**（波形->32 维手工特征->分类）— 需接入
- **FastAI**（波形->mel 频谱图 PNG->ResNet34）— 需接入

数据已处理完成，位于 processed data-129834/（129,834 条训练，14,416 条测试，5 折 CV）。

---

## 二、为什么这么做（3 个约束）

1. **增强只在训练集做**：增强样本不能进验证/测试集，否则评估失真。
2. **三个算法用同一套增强方法**：否则最后比 Top-1 分不清是算法差异还是增强差异。
3. **增强发生在波形阶段**：在特征/频谱图之后做会失真，必须在原始波形上做。

因此策略是：**共用底层增强模块 + 各自写胶水代码**。

---

## 三、增强方法（共用，4 种）

来自 audio_augmentation.py，所有算法共用同一套：

| 方法 | 参数 | 作用 |
|------|------|------|
| 时间拉伸 | 0.85x/0.90x/0.95x/1.05x/1.10x | 自然变速 |
| 音高偏移 | 正负1/2 半音 | 个体/地理差异 |
| 加高斯噪声 | 5/10/15 dB SNR | 录音条件变化 |
| 音量缩放 | 0.70x/0.85x/1.15x/1.30x | 距离/增益变化 |

规则：每条原始音频最多 16 个变体，每物种最多新增 50 条，目标每类至少 15 条。
种子用 SEED + fold，保证每折可复现。

---

## 四、文件清单

你需要两个 .py 文件，上传到 Kaggle 的 src/ 目录：

```
src/audio_augmentation.py       # 共用底层（YAMNet 那位已上传，用同一份）
src/augmentation_glue.py        # 各自那份（见下表）
```

| 队友 | 胶水文件位置 | 核心函数 |
|------|-------------|---------|
| YAMNet | 已集成，无需上传 | expand_with_augmentation()（已调用） |
| LightGBM | bird-vocalization-classifier/lightgbm/augmentation_glue.py | augment_for_lightgbm() |
| FastAI | bird-vocalization-classifier/fastaicode/augmentation_glue.py | augment_for_fastai() |

下面 A/B 两节分别给出 LightGBM 和 FastAI 的前后代码对比，可直接照抄。

---

## 五、怎么整合（按队友分）

### A. LightGBM 队友

**改动位置**：notebook6312d60402-0715.ipynb 的 cell-2，5 折循环里。

**原代码**（不改）：
```
df_train = pd.read_csv(csv_paths[f"ml_cv_fold{fold}_train.csv"])
df_val   = pd.read_csv(csv_paths[f"ml_cv_fold{fold}_val.csv"])
df_test  = pd.read_csv(csv_paths["ml_test.csv"])
X_tr, y_tr, _ = split_xy(df_train)
X_va, y_va, _ = split_xy(df_val)
X_te, y_te, te_fns = split_xy(df_test)
print(f"   train={len(X_tr)} val={len(X_va)} test={len(X_te)}")
```

**加 3 行**（在 split_xy 之后、lgb.train 之前）：
```python
import sys; sys.path.insert(0, "/kaggle/working/src")
from augmentation_glue import augment_for_lightgbm
X_aug, y_aug, _ = augment_for_lightgbm(
    df_train, find_audio, _features_from_wave, label2idx,
    target_per_species=15, sr=SR, duration=MAX_DURATION, seed=SEED + fold)
X_tr = np.concatenate([X_tr, X_aug], axis=0)
y_tr = np.concatenate([y_tr, y_aug], axis=0)
```

**原代码继续**（不改）：
```
train_set = lgb.Dataset(X_tr, label=y_tr)
...
```

**注意**：增强特征是临时计算的，每折重算，不会污染原始 `fn2feat` 缓存或泄漏到别的折。

---

### B. FastAI 队友

**改动位置**：fastaikaggle.txt 的 main_cv()，5 折循环里。

**原代码**（df_train 和 df_val 一起转频谱图）：
```
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_train['is_valid'] = False
df_val['is_valid'] = True
df_fold = pd.concat([df_train, df_val], ignore_index=True)

df_fold['spectrogram_path'] = df_fold.apply(audio_to_spectrogram, axis=1)
df_fold = df_fold.dropna(subset=['spectrogram_path']).reset_index(drop=True)
```

**改为**（train 先转频谱图，增强只加在 train，val 不动）：
```python
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_train['is_valid'] = False
df_val['is_valid'] = True

# 原始训练集先转频谱图（原逻辑）
df_train['spectrogram_path'] = df_train.apply(audio_to_spectrogram, axis=1)
df_train = df_train.dropna(subset=['spectrogram_path']).reset_index(drop=True)

# 新增：低资源物种增强（4 行）
import sys; sys.path.insert(0, "/kaggle/working/src")
from augmentation_glue import augment_for_fastai
df_train, _ = augment_for_fastai(
    df_train, resolve_audio_path, sr=Config.SAMPLE_RATE,
    clip_seconds=Config.CLIP_SECONDS, n_mels=Config.N_MELS,
    img_dir=Config.IMG_DIR, target_per_species=15, seed=42 + fold)
df_train = df_train.dropna(subset=['spectrogram_path']).reset_index(drop=True)

# 验证集转频谱图（不增强）
df_val['spectrogram_path'] = df_val.apply(audio_to_spectrogram, axis=1)
df_val = df_val.dropna(subset=['spectrogram_path']).reset_index(drop=True)

df_fold = pd.concat([df_train, df_val], ignore_index=True)
```

**增强 PNG 细节**：文件名格式 `{stem}_aug{v}_{species}.png`（如 `barpet1_aug02_barpet1.png`），存在 `Config.IMG_DIR` 目录下，不会与原始 PNG 冲突。

---

## 六、跑的顺序

每个人各自在 Kaggle 跑自己的 notebook，顺序不变：

```
Cell 1: 训练（含增强，每折自动触发，不需要单独 cell）
Cell 2: 噪声鲁棒性评估
Cell 3: 推理速度测量
```

增强在 Cell 1 的 5 折循环里每折自动跑一次。跑完会打印类似：

```
[LGB增强] 低资源物种 159 种 (<15 条)
[LGB增强] 新增增强特征 1144 条 -> X_tr: 99748 + 1144 = 100892
```
或
```
[FastAI增强] 低资源物种 159 种 (<15 条)
[FastAI增强] df_train: 99748 -> 100892 (+1144 条增强频谱图)
```

看到这个就说明接入成功了。YAMNet 那位已经集成好了，会打印类似：
```
[增强] 低资源物种 159 种 (少于 15 条)
[增强] X_train: (103853, 160000) -> (104997, 160000)
```

---

## 七、一致性保证

| 项目 | YAMNet | LightGBM | FastAI |
|------|--------|----------|--------|
| 共用底层 | audio_augmentation.py | 同 | 同 |
| 4 种增强方法 | 同 | 同 | 同 |
| 目标每类 | 15 | 15 | 15 |
| 种子 | SEED + fold | SEED + fold | 42 + fold |
| 验证集 | 不增强 | 不增强 | 不增强 |
| 增强阶段 | 波形 | 波形 | 波形 |

三个算法增强策略完全一致，Top-1 和噪声鲁棒性结果可直接横比。

---

## 八、常见问题

**Q: 为什么不直接生成增强文件到磁盘，大家共用？**
A: 两个原因：1) 在线增强每次 epoch 可以对同一条音频随机选不同方法，等价于无限增强空间，离线生成是固定的；2) 离线生成后很难保证增强样本不泄漏进同折验证集。每折在训练集上调一次最安全。

**Q: 增强后训练集变大，会不会拖慢训练？**
A: 只对 159 个低资源物种增强，新增约 1,144 条，相对 12.9 万总量只增加不到 1%，基本不影响速度。

**Q: sys.path 里的路径写什么？**
A: 看 src/ 目录在 Kaggle 上实际挂载在哪。常见是 /kaggle/working/src 或 /kaggle/input/<dataset>/src。上传后看一眼路径填进去。

---

*生成时间: 2026-07-18*
