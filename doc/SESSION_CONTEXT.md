# Session Context Summary — 2026-07-17 (+ 2026-07-18 修复)

## 会话目标

教授反馈：当前数据集样本太少（~5,000条），建议两条路：
1. 低资源类数据增强
2. 数据集扩容到 10 万条

本会话完成了两件事：**元数据流水线（本地）** 和 **音频增强模块（Kaggle 集成）**。

**最终结果**：训练池扩容到 **129,834 条**（超出 10 万目标），覆盖 1,126 种鸟类，无每类封顶；稀有物种（训练 <15 条，共 159 种）进行音频增强到目标 15 条。

---

## 2026-07-18 更新: audio_augmentation.py Bug 修复

| 问题 | 详情 | 状态 |
|------|------|------|
| 增强超发 bug | `expand_train_df` 中 `max(1, needed//current_count)` 导致 `needed < current_count` 时每个原始样本至少生成 1 条，造成 77% 超发（2025 条 vs 预期 1144 条） | ✅ 已修复 |
| 缺少 50 条/物种上限 | 文档规定 "maximum of 50 new variants per species"，代码未实现 | ✅ 已添加 `max_aug_per_species=50` |
| 随机选择缺少种子控制 | 修复后 `needed < current_count` 使用随机选择，需 seed 保证可复现 | ✅ 已添加 `seed` 参数 |

**修复后效果**: 159/159 低资源物种精确命中目标，0 超发 0 欠发。

---

## 2026-07-18 更新: V2.0 全量数据策略

| 指标 | V1 (07-17 会话) | V2 (07-18 正式) |
|------|-----------------|------------------|
| 训练样本 | 99,960 (分层采样+封顶) | **129,834** (保留全部清洗数据) |
| 每类上限 | 300 | 无上限 |
| 策略 | 采样控制不平衡 | 全量保留 + 训练时平衡控制 |

V2 正式输出文件位于 `processed data-129834/`，包含 `sampler_weight` / `loss_class_weight` / `cv_fold` 列。

---

## 产出文件清单

### 新建文件

| 文件 | 位置 | 用途 |
|------|------|------|
| `prepare_full_training_data.py` | `E:\MLwork\MLwork\data\augmenteddata\` | 完整数据处理流水线 (过滤非鸟类/划分测试集/计算权重/CV) |
| `Data Augmentation Documentation.md` | `E:\MLwork\MLwork\data\augmenteddata\` | 数据处理文档 (含 Kaggle 集成指南) |
| `audio_augmentation.py` | `E:\MLwork\MLwork\bird-vocalization-classifier\YAMNet\src\` | 实时音频增强模块 (训练时调用) |

### 生成的数据文件

V2 正式输出全部位于 `processed data-129834/`：

| 文件 | 行数 | 说明 |
|------|------|------|
| `01_bird_only_full.csv` | 144,250 | 过滤非鸟类+低质量后的全量鸟类 |
| `02_train_full_weighted.csv` | 129,834 | 全量训练集（含 `sampler_weight`/`loss_class_weight`/`cv_fold` 列） |
| `03_test_holdout.csv` | 14,416 | 留出测试集 |
| `class_weights.csv` | 1,126 | 每类一行：计数 + 两种权重 |
| `cv_fold{1-5}_train.csv` | ~103,853 | 5折CV训练（V2 无封顶） |
| `cv_fold{1-5}_val.csv` | ~25,967 | 5折CV验证（范围 25,949–25,981） |
| `summary.csv` | — | 各划分的行数/物种数汇总 |

### 已修改文件

| 文件 | 改动 |
|------|------|
| `YAMNet\src\yamnet_finetune_e2e.py` L500-506 | 保留 fn2wf + 调用增强函数 (4行新增) |

---

## 关键数据变化（V2.0 全量数据口径）

| 指标 | 旧 (V0 原始) | V1 (07-17 会话) | V2 (07-18 正式) |
|------|------------|-----------------|------------------|
| 训练样本 | ~4,780 | 99,960 (分层采样+封顶) | **129,834** (保留全部清洗数据) |
| 非鸟类污染 | 101 种 | 0 (已过滤) | **0** (已过滤) |
| 鸟类物种 | 1,229 (含非鸟) | 1,126 (纯鸟类) | **1,126** (纯鸟类) |
| 每类上限 | — | 300 | **无上限** |
| 每类中位数 | ~4 | ~72 | **≈72** (实测 71.5) |
| 每类最小 | 5 | 1 | **1** (增强后目标 15) |
| 每类最大 | ~5 | 300 | **1,300** |
| 训练时平衡控制 | 无 | 采样控制 | **采样 + 类权重 + 增强三重控制** |

---

## 数据处理4阶段

### Stage 1: 非鸟类过滤
- 移除 `primary_label` 为纯数字 ID 的物种 (101种: Amphibia/Insecta/Mammalia/Reptilia)
- 移除 rating ≤ 0.5 的低质量录音
- 结果: 167,308 → 144,250 条, 1,229 → 1,126 种

### Stage 2: 划分测试集（V2 全量保留，无每类封顶）
- 清洗后 144,250 条 → 90/10 pool/test 划分（SNR 分层对齐）
- 稀有物种（<5 条）全部留在训练池，不入测试集
- 训练池 129,834 条 / 1,126 种；测试集 14,416 条 / 1,093 种
- V2 不再对每类封顶（V1 的 162/200/300 上限已废弃）

### Stage 3: 低资源音频增强
- 识别训练集中 <15 条的物种（**159 种**）
- 4种增强: 时间拉伸 / 音高偏移 / 噪声叠加 / 音量变化
- **执行方式**:
  - **在线(推荐)**: 集成在 `yamnet_finetune_e2e.py` 中，由 `expand_with_augmentation` 在训练时实时生成（无需磁盘文件）
  - 单条测试: `python audio_augmentation.py <audio_file> [output_dir]`（仅用于验证增强效果，不批量生成）

### Stage 4: 5折交叉验证
- SNR分层 KFold
- 增强样本的原始文件泄漏检查

---

## Kaggle 运行流程

```
1. 上传文件到 Kaggle:
   - src/yamnet_bird_pipeline.py  (已有)
   - src/yamnet_finetune_e2e.py   (已修改)
   - src/audio_augmentation.py    (新增)
   - processed data-129834/ 下的 CV 划分 CSV  (12.9万条版本，替换旧的 ml_cv_fold*)

2. Notebook:
   Cell 1: %run -i src/yamnet_finetune_e2e.py   ← 增强自动触发
   Cell 2: %run -i src/noise_eval_e2e.py
   Cell 3: %run -i src/measure_inference_e2e.py

3. 预期输出:
   [增强] 低资源物种: 159 种 (少于 15 条)
   [增强] X_train: (~103,853, 160000) -> (~103,853 + 增强变体, 160000)
   [训练] fold1 开始 ...
```

---

## 待完成

| 事项 | 说明 | 状态 |
|------|------|------|
| 上传新CSV到Kaggle | 用 `processed data-129834/` 的 CV 文件替换旧的 ml_cv_fold* | ⬜ 待做 |
| 上传 audio_augmentation.py | 放到 Kaggle 的 src/ 目录 (已修复超发 bug) | ⬜ 待做 |
| 跑一遍完整训练 | 验证增强+12.9万条数据的实际效果 | ⬜ 待做 |
| 比较新旧结果 | 记录 Top-1/Top-5 提升幅度 | ⬜ 待做 |
| ~~修复增强超发 bug~~ | `expand_train_df` overshooting 已修复 (2026-07-18) | ✅ 已完成 |
| ~~添加 50条/物种上限~~ | `max_aug_per_species=50` 已实现 (2026-07-18) | ✅ 已完成 |

## 2026-07-18 注意事项

1. **两个 audio_augmentation.py 已同步**: `augmenteddata/` 和 `YAMNet/src/` 内容一致
2. **YAMNet 训练代码无需改动**: `expand_with_augmentation` 新增的 `max_aug_per_species`/`seed` 参数均有默认值，向后兼容
3. **上传 Kaggle 前确认使用修复后的 `audio_augmentation.py`**，避免旧版超发 bug 影响训练

---

## 相关文档

- 数据处理文档: `E:\MLwork\MLwork\data\augmenteddata\Data Augmentation Documentation.md`
- 项目总览: `E:\MLwork\MLwork\bird-vocalization-classifier\doc\CONTEXT_SUMMARY.md`
- 教授讨论文档: `E:\MLwork\MLwork\bird-vocalization-classifier\doc\教授讨论-项目进展汇报.docx`
- YAMNet 交接文档: `E:\MLwork\MLwork\bird-vocalization-classifier\YAMNet\HANDOFF.md`

---

*生成时间: 2026-07-17 | 最后更新: 2026-07-18*
