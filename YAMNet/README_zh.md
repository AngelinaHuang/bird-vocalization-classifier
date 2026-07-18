# 嘈杂野外环境下的鸟类鸣声分类

本项目对三种建模范式进行对比研究——基于表格特征的梯度提升树、基于梅尔频谱图图像的迁移学习、以及基于 YAMNet 原始波形迁移学习——用于细粒度鸟类鸣声音频分类，重点关注**对环境噪声的鲁棒性**。

---

## 1. 项目背景

野外录音中的鸟类鸣声分类面临三大挑战：

1. **背景噪声**——风声、雨声及人为噪声与目标鸣声相互重叠。
2. **长尾类别分布**——少数物种样本充足，大量物种样本稀少。
3. **细粒度类别**——声学特征相近的物种容易混淆。

本项目的核心贡献是一组**可控噪声注入实验**：每个模型均在同一干净基线上评估，并在三种信噪比（SNR）档位（5 dB、0 dB、−5 dB）的人工降质版本上重复评估。准确率随 SNR 档位的衰减曲线即为主要对比维度。

### 研究问题

- 哪种范式在干净条件下准确率最高？
- 哪种范式对加性噪声最鲁棒（准确率衰减最慢）？
- 哪些物种对最易混淆？噪声是否会放大这些混淆？

---

## 2. 数据集

- **来源**：Google Perch / BirdCLEF 2021–2026 竞赛的鸟类发声元数据。小组将六个年度 CSV 字段级去重后，过滤非鸟类（101 种 Amphibia/Insecta/Mammalia/Reptilia）与低质量录音（rating ≤ 0.5），得到 **144,250 条纯鸟类录音**，覆盖 **1,126 个物种**。音频文件挂载在 Kaggle 的 BirdCLEF 年度数据集目录下。
- **预处理**：所有片段重采样至 **16 kHz 单声道**，峰值归一化到 **[−1, 1]** 的 `float32`，并统一为固定长度（取中段截断或末尾补零）。
- **划分**：90/10 分层训练/测试切分（SNR 分层对齐），稀有物种（<5 条）全部留在训练池。训练池 **129,834 条** / 1,126 种，留出测试集 **14,416 条** / 1,093 种。训练集内嵌套 5 折 SNR 分层交叉验证（每折 ~103,853 训练 / ~25,967 验证）。切分使用固定随机种子 42，确定性可复现，使噪声鲁棒性评估能复用完全相同的测试集。详见 `data/augmenteddata/Data Augmentation Documentation.md`。
- **低资源增强**：训练集中样本数 <15 条的 159 种稀有物种，通过实时音频增强（时间拉伸 / 音高偏移 / 噪声叠加 / 音量变化）补足到目标 15 条，每物种最多生成 50 条变体。详见 `src/e2e/audio_augmentation.py`。

---

## 3. 方法

### 3.1 三种建模范式

| 模型 | 输入表示 | 方法 |
|---|---|---|
| **LightGBM** | 人工设计的表格化声学特征 | 梯度提升树 |
| **FastAI** | 梅尔频谱图（图像） | 图像分类迁移学习 |
| **YAMNet** | 原始波形 | 基于 Google YAMNet 音频编码器的迁移学习 + 小型分类头 |

三者均通过**同一套统一评估框架**进行评估，确保对比公平，不受指标或切分差异影响。

### 3.2 YAMNet 迁移学习（本仓库）

本仓库实现两种策略，可独立运行、公平对比：

**策略一：冻结嵌入 + 分类头**
1. 每条波形过冻结的 YAMNet 编码器，输出每 0.48 秒一帧的 1024 维嵌入。
2. 对帧维度取平均，得到该片段的 1024 维整体向量，并缓存到磁盘（嵌入计算开销大，跨运行复用）。
3. 在缓存嵌入上训练一个小型全连接分类头（`Dense(256, ReLU) → Dropout → Dense(类别数, softmax)`），配合早停、检查点保存、学习率 Plateau 衰减。
4. 代码：`src/yamnet_bird_pipeline.py`

**策略二：端到端微调（已实现）**
1. 解冻 YAMNet 顶层卷积块，原始波形直通，端到端联合训练。
2. 差分学习率：YAMNet 变量 lr=1e-5，分类头 lr=1e-3。
3. MixUp 增强（alpha=0.2）+ 类别平衡权重缓解长尾。
4. 训练时实时音频增强：159 种低资源物种（<15 条）在每折训练集内动态生成变体（最多 50 条/种）。
5. 代码：`src/e2e/yamnet_finetune_e2e.py` + `src/e2e/audio_augmentation.py`

### 3.3 可控噪声注入实验

对每条测试片段与每个 SNR 档位，按目标 SNR 将高斯白噪声叠加到干净波形上，将带噪波形重新送入 YAMNet 编码，并记录预测结果：

$$\text{SNR}_{\text{dB}} = 10 \log_{10}\frac{P_{\text{信号}}}{P_{\text{噪声}}}$$

SNR 越低 ⇒ 噪声越强。逐档计算准确率并绘制衰减曲线。高斯噪声作为可控、可复现的基线；噪声模块独立封装，后续可替换为真实风声/雨声，无需改动管道其余部分。

---

## 4. 项目结构

> 本目录是 [`bird-vocalization-classifier`](https://github.com/AngelinaHuang/bird-vocalization-classifier) 仓库的 **YAMNet 子文件夹**。另外两种建模范式（LightGBM、FastAI）位于同级目录中。

```
YAMNet/
├── README.md                          # 英文版
├── README_zh.md                       # 中文版（本文件）
├── requirements.txt
├── .gitignore
├── HANDOFF.md                         # 排查交接文档
├── _inspect.py                        # 快速查看缓存文件内容
├── src/
│   ├── yamnet_bird_pipeline.py        # 策略一：冻结嵌入 + 分类头训练（5 折 CV）
│   ├── noise_robustness_eval.py       # 策略一：SNR 档位噪声注入 + 衰减测量
│   ├── unified_evaluation.py          # 模型无关的指标计算与绘图
│   ├── measure_inference.py           # 策略一：推理速度 + 显存测量
│   ├── measure_inference_template.py  # 队友推理速度测量模板
│   └── e2e/                           # 策略二：端到端微调
│       ├── yamnet_finetune_e2e.py     # 端到端微调主管道（差分 lr + MixUp + 增强）
│       ├── audio_augmentation.py      # 低资源物种实时音频增强模块
│       ├── noise_eval_e2e.py          # 端到端模型噪声评估
│       └── measure_inference_e2e.py   # 端到端推理速度测量
└── outputs/
    ├── yamnet/                        # 策略一产物（冻结嵌入）
    │   ├── label_map.json
    │   ├── embeddings.npz
    │   ├── cv_per_fold.csv / cv_summary.csv
    │   └── fold{1-5}/
    │       ├── yamnet_bird_model.keras
    │       ├── test_predictions.npz
    │       └── noise_results.npz
    ├── e2e/                           # 策略二产物（端到端微调）
    │   └── fold{1-5}/
    │       ├── yamnet_e2e_model.keras
    │       ├── test_predictions.npz
    │       └── noise_results.npz
    └── figures/
        ├── confusion_matrix_YAMNet.png
        └── noise_robustness.png
```

---

## 5. 环境安装

```bash
git clone git@github.com:AngelinaHuang/bird-vocalization-classifier.git
cd bird-vocalization-classifier/YAMNet
pip install -r requirements.txt
```

依赖：`tensorflow>=2.10`、`tensorflow-hub`、`librosa`、`soundfile`、`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`seaborn`。

> YAMNet 首次运行时自动从 TensorFlow Hub 下载（约 17 MB）。策略一（冻结嵌入 + 分类头）无需 GPU，CPU 即可胜任。策略二（端到端微调）建议使用 GPU。

---

## 6. 使用方法

所有脚本使用相对于 `src/` 的路径，因此需在 **YAMNet/** 目录下的 `src/` 中运行。

```bash
cd src
```

### 6.1 策略一：冻结嵌入 + 分类头

```bash
python yamnet_bird_pipeline.py
```

读取 CV 划分 CSV，在挂载的 BirdCLEF 年度数据集目录下定位音频，提取并缓存 YAMNet 嵌入，5 折训练分类头，将模型、标签映射与测试预测写入 `outputs/yamnet/`。

### 6.2 策略二：端到端微调

```bash
cd e2e
python yamnet_finetune_e2e.py
```

解冻 YAMNet 顶层，差分学习率 + MixUp + 实时音频增强，5 折端到端训练。产物写入 `outputs/e2e/`，不覆盖策略一产物。

### 6.3 测试音频增强效果（单条）

```bash
python e2e/audio_augmentation.py <音频文件路径> [输出目录]
```

对单条音频应用 4 种增强方法（时间拉伸 / 音高偏移 / 噪声叠加 / 音量变化），生成变体样本到指定目录。仅用于验证增强效果，训练时增强由 `yamnet_finetune_e2e.py` 自动调用。

### 6.4 运行噪声鲁棒性实验

**策略一：**
```bash
python noise_robustness_eval.py
```

**策略二：**
```bash
cd e2e
python noise_eval_e2e.py
```

在 5 / 0 / −5 dB 下注入高斯噪声，重新编码每条带噪波形，记录各档准确率。结果写入 `outputs/yamnet/noise_results.npz`（或 `outputs/e2e/`），衰减曲线写入 `outputs/figures/noise_robustness.png`。

### 6.5 生成评估报告与图表

```bash
python unified_evaluation.py
```

计算准确率 / 精确率 / 召回率 / F1（macro 与 weighted）、逐类明细、混淆矩阵、多模型准确率对比、噪声衰减曲线。

### 6.6 Kaggle Notebook 运行流程

策略一（3 个 cell）：
```
cell1: %run -i src/yamnet_bird_pipeline.py       # 5 折训练
cell2: %run -i src/noise_robustness_eval.py       # 噪声评估
cell3: %run -i src/measure_inference.py           # 推理速度 + 显存
```

策略二（3 个 cell，与策略一独立，不需先跑策略一）：
```
cell1: %run -i src/e2e/yamnet_finetune_e2e.py      # 端到端训练（30-60 min/fold）
cell2: %run -i src/e2e/noise_eval_e2e.py            # 噪声评估
cell3: %run -i src/e2e/measure_inference_e2e.py     # 推理速度 + 显存
```

---

## 7. 评估

统一框架对所有三个模型一视同仁。每个模型最终提供 `(y_true, y_pred, class_names)`，框架据此计算：

- **分类指标**：准确率，macro 与 weighted 精确率 / 召回率 / F1，逐类报告。
- **混淆矩阵**：完整热力图，定位最易混淆的物种对。
- **多模型对比**：准确率 / macro-F1 / weighted-F1 的分组柱状图。
- **噪声衰减曲线**：各模型准确率随 SNR 档位变化——主要鲁棒性对比。
- **开销**：单条推理延迟及（适用时）GPU 显存占用。

---

## 8. 当前结果

### 策略一（冻结嵌入 + 分类头）— 5 折交叉验证

YAMNet 在 Kaggle 上跑通完整 BirdCLEF 数据（1,229 类，fold1：train 3,824 / val 956 / test 1,196）。绝对准确率低是 1,229 类×每类约 3 条样本的长尾分布决定的，非 bug；作业关注三个模型在噪声下的**相对衰减趋势**，而非绝对分。

**5 折 clean 准确率：**

| 指标 | Mean ± Std |
|------|-----------|
| 准确率 | 1.59% ± 0.27% |
| Macro-F1 | — |
| Weighted-F1 | — |

5 折逐折 clean 准确率：fold1=1.76%, fold2=1.09%, fold3=1.51%, fold4=1.76%, fold5=1.84%。

**噪声鲁棒性衰减（5 折 mean ± std）：**

| SNR 档位 | 准确率 |
|----------|--------|
| clean | 1.59% ± 0.27% |
| 5 dB | 0.47% ± 0.16% |
| 0 dB | 0.30% ± 0.10%（≈随机基线） |
| −5 dB | 0.07% ± 0.06% |

准确率随噪声增强单调下降，0 dB 档即坍塌至随机水平，表明强噪声下的鲁棒性是主要瓶颈。

### 策略二（端到端微调 + V2.0 全量数据）— 待跑

V2.0 数据集（129,834 训练条 / 1,126 类）已就绪，端到端微调代码已实现。预期：
- 129K 样本大幅缓解长尾过拟合，clean 准确率应显著提升
- MixUp + 音频增强应使噪声下衰减更平缓
- 差分学习率微调 YAMNet 顶层，特征提取可适配鸟鸣

---

## 9. 局限与未来工作

- **长尾少样本**：原始数据 1,229 类×每类约 3 条，过拟合明显。**V2.0 已扩容至 129,834 条 / 1,126 类**，采样权重 + 类别权重 + 音频增强三重控制，待 Kaggle 实跑验证效果。
- **噪声模型**：高斯白噪声为可控基线；替换为真实风声/雨声仅需改动噪声模块。
- **端到端微调**：已实现（`yamnet_finetune_e2e.py`），差分学习率 + MixUp + 实时增强，待 Kaggle 实跑。
- **可复现性**：sklearn 切分与 TF 训练均设固定种子，增强模块种子可控。重训可得近似结果。

---

## 10. 可复现性

- 数据加载、切分、训练、噪声注入全程使用固定随机种子。
- 分层切分确定性，噪声鲁棒性实验与干净基线操作于完全相同的测试片段。
- 嵌入缓存将昂贵的 YAMNet 前向计算与下游迭代解耦，确保重复实验既快又逐字节可复现。
