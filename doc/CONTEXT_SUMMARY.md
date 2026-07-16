# 项目上下文总结文档

> 本文档旨在帮助新会话快速了解项目全貌。阅读本文档后即可理解项目背景、目标、数据、模型架构及当前进度，无需重新阅读所有原始文档。

---

## 一、项目概览

| 项目名称 | **噪声鲁棒的鸟类发声分类——面向低资源场景** (Noise-Robust Bird Vocalization Classification for Low-Resource Scenarios) |
| --- | --- |
| 核心任务 | 多类别音频分类：从野外录音中识别鸟类物种，面临长尾分布和复杂环境噪声 |
| 团队规模 | 3 人 |
| 项目结构 | `doc/`（文档）、`lightgbm/`（LightGBM 代码）、`fastaicode/`（FastAI CNN 代码）、`YAMNet/`（YAMNet 迁移学习代码） |

---

## 二、团队成员与分工

| 成员 | 职责 | 负责模型 |
| --- | --- | --- |
| **Jianan Zhang（张建安）** | 文献调研、数据管道、LightGBM 模型 | LightGBM（表格特征 + 梯度提升树） |
| **Jincheng Chen（陈锦程）** | FastAI 图像模型 | FastAI 轻量 CNN（梅尔频谱图图像分类） |
| **Wenjuan Huang（黄文娟）** | YAMNet 迁移学习、统一评估框架、报告整合 | YAMNet（原始音频波形迁移学习） |

---

## 三、数据全景

### 3.1 数据来源

数据集来源于 **Kaggle Google Perch Bird Vocalization Classifier**（即 BirdCLEF 2021–2026 竞赛合集），来自 Xeno-Canto 和 iNaturalist 平台。

### 3.2 数据规模

| 阶段 | 记录数 | 说明 |
| --- | --- | --- |
| 原始合并 | **183,239 条** | 6 个年度 CSV 合并（2021–2026） |
| 字段级去重后 | **167,308 条** | 按 filename 智能合并重复记录，**零信息损失** |
| 分层抽样子集 | **5,976 条** | 双层分层抽样（物种 × SNR），覆盖全部 **1,229 个物种** |
| 训练集 | **4,780 条**（80%） | 按 SNR 分层划分 |
| 留出测试集 | **1,196 条**（20%） | 按 SNR 分层划分 |

> **注意**：原项目提案中描述为 12,400 条记录、86 个物种。实际合并 2021–2026 数据后扩大至 1,229 个物种，因此「每物种至少 5 样本」的硬约束使抽样子集从约 2,000 扩大到约 6,000。

### 3.3 关键元数据字段

| 字段 | 含义 |
| --- | --- |
| `filename` | 音频文件唯一标识符（主键） |
| `primary_label` | 物种编码（eBird 代码 / 分类单元 ID） |
| `secondary_labels` | 录音中出现的其他物种（列表） |
| `type` | 录音类型（call, song, flight call 等） |
| `rating` | 质量评分（0.0–5.0），用作 **SNR 代理指标** |
| `latitude` / `longitude` | 地理坐标 |
| `scientific_name` / `common_name` | 分类学名称 |
| `collection` | 来源（Xeno-Canto / iNaturalist） |
| `class_name` | 分类纲（Aves, Amphibia, Mammalia, Insecta 等） |
| `source_year` | 数据来源年份（可包含多年，如 `"2022,2025,2026"`） |

---

## 四、核心方法论

### 4.1 字段级去重（Zero-Loss Deduplication）

传统做法是直接删除重复 `filename` 的行，但这会**丢失信息**。本项目的创新做法是按字段类型智能合并：

| 字段类型 | 合并策略 | 示例 |
| --- | --- | --- |
| **列表型**（`secondary_labels`, `type`） | 取所有非空值的并集去重 | `['song']` + `['call']` → `['song', 'call']` |
| **数值型**（`rating`, `latitude`, `longitude`） | 取最新 `source_year` 对应的非空值 | 2019 rating=4.0, 2025 rating=3.5 → 保留 3.5 |
| **文本型**（`scientific_name`, `common_name`） | 取最新年份非空值 | — |
| **年份型**（`source_year`） | 合并去重后逗号分隔 | `"2022"` + `"2023"` → `"2022,2023"` |

### 4.2 SNR 分层体系

基于 `rating` 字段（0–5 分）生成三层 SNR 分类：

| SNR 等级 | 定义 | 数据集占比 |
| --- | --- | --- |
| **High SNR** | rating ≥ 4.0，干净录音 | 57.7% |
| **Medium SNR** | 2.0 ≤ rating < 4.0 | 27.2% |
| **Heavy SNR** | rating < 2.0，严重噪声 | 15.1% |

### 4.3 双层分层抽样

- **第一层**：按物种（1,229 种）分层
- **第二层**：每物种内按 SNR 等级比例分配
- **硬约束**：每物种至少 5 个样本（稀有物种全部保留）
- **代表性验证**：物种覆盖率 100%，SNR 分布偏差 < ±2pp

### 4.4 训练/测试划分与交叉验证

```
全部数据 (183,239)
  → 字段级去重 (167,308)
    → 分层抽样 (5,976)
      → 80/20 划分（按 SNR 分层）
        ├── 训练集 (4,780)
        │     → 5 折分层交叉验证
        │       ├── Fold 1-5: 训练 3,824 + 验证 956
        └── 测试集 (1,196)
```

---

## 五、三种模型方案

### 5.1 LightGBM（Jianan Zhang）

| 属性 | 说明 |
| --- | --- |
| **输入** | 手工提取的表格化音频统计特征（频谱质心、带宽、MFCC、过零率等） |
| **算法** | 梯度提升决策树（Gradient Boosting Tree） |
| **调优** | 网格搜索（树数量、最大深度、叶子数、类别权重处理不平衡） |
| **优势** | 硬件需求极低、训练推理快、特征重要性可解释 |
| **劣势** | 无法捕获频谱图的空间纹理特征 |

### 5.2 FastAI 轻量 CNN（Jincheng Chen）

| 属性 | 说明 |
| --- | --- |
| **输入** | 音频转梅尔频谱图（mel-spectrogram）的 2D 图像 |
| **算法** | 轻量级 CNN 图像分类模型（FastAI 高级 API） |
| **调优** | lr_find、early stopping、dropout 正则化 |
| **增强** | **双阶段增强**（波形级 + 频谱图级），含时间掩蔽、音高变换、亮度调整等 |
| **优势** | 重构了 BirdNET 思路但更轻量，配有 BirdNET 不具备的双阶段增强 |
| **劣势** | 低资源稀有物种上易过拟合 |

### 5.3 YAMNet 迁移学习（Wenjuan Huang）

YAMNet 部分包含**两种训练策略**，在同一数据和评估口径下对比：

#### 策略一：冻结编码器 + 训练分类头（已完成，5 折实跑）

| 属性 | 说明 |
| --- | --- |
| **输入** | YAMNet 预提取的 1024 维嵌入向量 |
| **算法** | 预训练 YAMNet（MobileNetV1 架构，AudioSet 521 类训练），**全部冻结** |
| **分类头** | Dense(256, ReLU) → Dropout(0.3) → Dense(1229, Softmax) |
| **训练** | 标准model.fit，EarlyStopping + ReduceLROnPlateau |
| **优势** | 训练极快（读嵌入缓存），硬件需求低 |
| **劣势** | 只调分类头，特征提取无法适配鸟鸣 |

#### 策略二：端到端微调（已实现，待 Kaggle 实跑）

| 属性 | 说明 |
| --- | --- |
| **输入** | 未经转换的原始音频波形（16000×5 采样点） |
| **算法** | YAMNet 作为可训练 Keras 层，**全部解冻**，差分学习率保护 |
| **分类头** | 同策略一（Dense 256 → Dropout → Dense 1229） |
| **训练** | 自定义 E2ETrainer：两个 Adam optimizer 分别更新 YAMNet(lr=1e-5) 和分类头(lr=1e-3) |
| **增强** | MixUp (α=0.2)：每批随机混合两条波形，标签按比例混合——长尾分类关键技巧 |
| **优势** | 特征提取可适配鸟鸣，拟合能力强 |
| **劣势** | 训练慢（每批过完整 YAMNet），需 GPU，容易过拟合 |

---

## 六、数据集输出文件清单

| 文件名 | 行数 | 用途 |
| --- | --- | --- |
| `ml_full_deduped.csv` | 167,308 | 完整去重主数据集（字段合并后） |
| `ml_sampled.csv` | 5,976 | 双层分层抽样子集 |
| `ml_train.csv` | 4,780 | 训练集（80%） |
| `ml_test.csv` | 1,196 | 留出测试集（20%） |
| `ml_cv_fold{1-5}_train.csv` | 各 3,824 | 交叉验证训练折 |
| `ml_cv_fold{1-5}_val.csv` | 各 956 | 交叉验证验证折 |
| `ml_data_pipeline.py` | — | 可复现的数据处理脚本 |

### YAMNet 产物目录结构

```
results/yamnet/
├── embeddings.npz              # 5976 条嵌入缓存 (策略一用)
├── noise_embeddings.npz        # 噪声嵌入缓存 (5 折共享)
├── waveforms_cache.npz         # 波形缓存 (策略二端到端用)
├── label_map.json              # 1229 类映射
├── cv_per_fold.csv             # 每折 clean 准确率
├── cv_noise_per_fold.csv       # 每折各 SNR 档准确率
├── cv_summary.csv              # 5 折汇总 mean±std (clean + 噪声)
├── fold1..5/                   # 策略一：冻结版各折产物
│   ├── yamnet_bird_model.keras
│   ├── test_predictions.npz
│   └── noise_results.npz
├── e2e/                        # 策略二：端到端微调产物 (待跑)
│   ├── fold1..5/
│   │   ├── yamnet_e2e_model.keras
│   │   ├── best_weights.weights.h5
│   │   ├── test_predictions.npz
│   │   └── noise_results.npz
│   ├── cv_per_fold.csv
│   ├── cv_noise_per_fold.csv
│   ├── cv_summary.csv
│   └── inference_metrics.csv
```

---

## 七、项目创新点

1. **字段级去重**：重复 filename 不简单删除而是智能按字段合并，实现**零信息损失**，保留全部历史标签和分类信息。

2. **三种异构 ML 范式并排比较**：传统表格 ML（LightGBM） vs 图像深度学习（FastAI CNN） vs 原始音频迁移学习（YAMNet），统一数据集公平对比。

3. **可控噪声实验**：设计基线对照组（高 SNR ≥15 dB 干净录音）和人工噪声测试组（高斯白噪声 / 低频风噪 / 间歇脉冲噪声，SNR 分 5 dB / 0 dB / -5 dB 三级），实现**完全量化**的抗噪性能对比——这是 BirdNET 原生工作流所不具备的。

4. **标准化双阶段增强管线**：波形级增强（白噪声注入、时间掩蔽、音高变换、音量缩放）+ 频谱图级增强（亮度调整、局部遮挡、缩放变换）。

5. **YAMNet 双策略微调对比**：同一预训练模型，"冻结编码器 + 训练分类头" vs "端到端微调 + MixUp + 差分学习率"，在同一数据和噪声评估口径下对比两种迁移学习策略的拟合能力和噪声鲁棒性差异。

6. **MixUp 长尾增强**：端到端策略中引入 MixUp 数据增强（α=0.2），通过跨类样本混合让尾部稀有类借用头部类的信息，是长尾分类场景下的标准技巧。

---

## 八、当前实验结果（2026-07-16）

### 8.1 已完成模型结果对比

LightGBM 和 YAMNet（策略一：冻结）的 5 折交叉验证 + 噪声评估已完成，结果存于 `result/` 目录。

| 指标 | LightGBM | YAMNet（冻结） | YAMNet 优势 |
| --- | --- | --- | --- |
| **Clean 准确率** | 0.48% ± 0.14% | 1.91% ± 0.33% | 4.0× |
| **5dB 准确率** | 0.08% ± 0.05% | 0.42% ± 0.19% | 5.0× |
| **0dB 准确率** | 0.07% ± 0.03% | 0.37% ± 0.15% | 5.5× |
| **−5dB 准确率** | 0.07% ± 0.03% | 0.12% ± 0.04% | 1.7× |
| **Clean F1 (weighted)** | 0.0045 | 0.0178 | 4.0× |
| **推理延迟** | 154.5 ms | 86.0 ms | YAMNet 快 44% |
| **GPU 显存峰值** | N/A（CPU） | 211.4 MB | — |

随机基线 = 1/818 = 0.122%（测试集 818 个类）。

### 8.2 关键发现

1. **YAMNet 在所有 SNR 档位均优于 LightGBM**，符合迁移学习抗噪能力强的预期。
2. **两者在 0dB 及以下基本塌缩为随机水平**，与文献中 BirdNET "SNR < 3dB 时显著下降"的结论一致。
3. **LightGBM 在 5dB 就已低于随机**，说明手工特征对噪声极其敏感。
4. **绝对准确率低是数据约束决定的**：1229 类 × 每类仅 ~3.9 个训练样本，测试集 63% 的类只有 1 个样本。两模型均显著优于随机（LightGBM 4× 随机，YAMNet 15.6× 随机），证明模型学到了有效特征。
5. **YAMNet 推理反而更快**（86ms vs 154ms），因为 LightGBM 的手工特征提取开销大于 YAMNet 嵌入提取。

### 8.3 待完成工作

| 任务 | 状态 | 说明 |
| --- | --- | --- |
| YAMNet 端到端微调（策略二） | ✅ 代码已完成，待 Kaggle 实跑 | `src/yamnet_finetune_e2e.py` + 噪声评估 + 推理测量 |
| FastAI CNN 结果 | ⏳ 待队友完成 | Jincheng Chen 负责的部分 |
| BirdNET 基线对比 | 📋 建议补做 | 用 `birdnetlib` 在同一测试集 + 噪声条件下跑基线 |
| 三模型统一对比图 | 📋 待 FastAI 结果 | 用 `unified_evaluation.py` 生成合并图表 |
| 扩大数据集评估 | 📋 建议考虑 | 每类样本从 ~4 扩到 ≥20，或筛选 Top-50 高频物种子集 |

---

## 九、文献综述核心发现

1. **BirdNET 的局限**：
   - 在 SNR < 3 dB 时性能显著下降
   - 存在地理偏差（非洲/亚洲 PR AUC 仅 0.03–0.04，北美 0.23）
   - 对常见物种效果好，稀有物种效率低下
   - 缺乏标准化的可控噪声测试管线

2. **Google Perch 的局限**：
   - "bittern lesson"：简单的监督模型仍难以被超越
   - 对不在训练集中的物种泛化能力有限（BirdCLEF+ 2025 仅约 60% 准确率）
   - 同样缺乏统一基准评估框架

3. **BirdCLEF 竞赛揭示的核心挑战**：
   - 长尾类分布、训练/部署之间的域偏移
   - 噪声野外录音、弱标签问题
   - 稀有物种训练样本不足

4. **噪声鲁棒性**：
   - 数据增强是提升噪声鲁棒性的主要手段
   - 噪声去除预处理在某些情况下反而降低分类准确率
   - 缺乏跨算法范式的系统性噪声鲁棒性对比实验

---

## 十、文档索引

| 文件 | 说明 |
| --- | --- |
| `Final Project Proposal-HJJ.pdf` | 原始项目提案（英文），含问题陈述、数据集描述、实施计划、团队分工 |
| `Literature Review-711.docx` | 英文文献综述，涵盖 BirdNET/Perch 局限、BirdCLEF 挑战、噪声鲁棒性、低资源分类等 8 个主题，含 27 篇参考文献 |
| `Data Processing Documentation.md` | 英文版数据处理文档，含完整 Python 代码 |
| `Data Processing Doc.pdf` | 英文版数据处理文档 PDF（10 页），与上同内容 |
| `Data Processing Doc.zh.md` | **最新版**中文数据处理文档（含字段级去重策略） |
| `数据处理文档.md` | 中文版数据处理文档（含完整 Python 代码） |

---

## 十一、快速上手指南

### 若要继续项目工作，建议按以下顺序操作：

1. **先读本文档**（CONTEXT_SUMMARY.md）建立全局认知
2. **阅读 `Data Processing Doc.zh.md`** 了解最新的数据管道设计（重点关注字段级去重策略）
3. **查阅 `Literature Review-711.docx`** 了解相关工作和研究空白
4. **查看 §八「当前实验结果」** 了解 LightGBM 和 YAMNet 冻结版的已有数字
5. **进入对应模型目录**（`lightgbm/`、`fastaicode/`、`YAMNet/`）查看各模型的具体实现代码
6. **YAMNet 端到端微调**：查看 `YAMNet/src/yamnet_finetune_e2e.py` 和 `YAMNet/HANDOFF.md` §9
7. **数据处理入口**：`ml_data_pipeline.py` 是完整可复现的数据处理脚本

### YAMNet Kaggle 运行方式

**策略一（冻结，已跑完）**：4 个 cell
```
cell1: %run -i src/yamnet_bird_pipeline.py     # 5 折训练
cell2: %run -i src/unified_evaluation.py        # 定义评估函数 (注释掉 demo)
cell3: %run -i src/noise_robustness_eval.py      # 噪声评估
cell4: %run -i src/measure_inference.py         # 推理速度
```

**策略二（端到端微调，待跑）**：3 个 cell
```
cell1: %run -i src/yamnet_finetune_e2e.py       # 5 折端到端训练 (30-60 min/fold)
cell2: %run -i src/noise_eval_e2e.py             # 噪声评估 (波形直通)
cell3: %run -i src/measure_inference_e2e.py     # 推理速度 + 显存
```

两种策略使用完全相同的数据（同一 CSV、同一 5 折划分、同一噪声种子），结果可直接对比。

---

*最后更新：2026-07-16*
