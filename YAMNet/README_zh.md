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

- **来源**：Google Perch / BirdCLEF 2021–2026 竞赛的鸟类发声元数据。小组将六个年度 CSV 字段级去重、双层分层抽样后切分为 `ml_cv_fold1_train.csv` / `ml_cv_fold1_val.csv` / `ml_test.csv`，音频文件挂载在 Kaggle 的 BirdCLEF 年度数据集目录下，共 **1,229 个物种**。
- **预处理**：所有片段重采样至 **16 kHz 单声道**，峰值归一化到 **[−1, 1]** 的 `float32`，并统一为固定长度（取中段截断或末尾补零）。
- **划分**：80/20 分层训练/测试切分，保持各档 SNR 比例；训练集内嵌套 5 折分层交叉验证（YAMNet 取 fold1：3,824 训练 / 956 验证），测试集 1,196 条留出。切分使用固定随机种子，确定性可复现，使噪声鲁棒性评估能复用完全相同的测试集。详见 `Data Processing Documentation.md`。

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

YAMNet 管道采用轻量的"预计算嵌入 + 训练分类头"策略：

1. 每条波形过冻结的 YAMNet 编码器，输出每 0.48 秒一帧的 1024 维嵌入。
2. 对帧维度取平均，得到该片段的 1024 维整体向量，并缓存到磁盘（嵌入计算开销大，跨运行复用）。
3. 在缓存嵌入上训练一个小型全连接分类头（`Dense(256, ReLU) → Dropout → Dense(类别数, softmax)`），配合早停、检查点保存、学习率 Plateau 衰减。

端到端微调变体（解冻 YAMNet 顶层卷积块并使用差异化学习率）作为后续工作在 `src/yamnet_bird_pipeline.py` 中列出。

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
├── _inspect.py                        # 快速查看缓存文件内容
├── data/
│   └── processed/                     # （预留，后续预处理用）
├── src/
│   ├── yamnet_bird_pipeline.py        # YAMNet 嵌入提取 + 分类头训练
│   ├── noise_robustness_eval.py       # SNR 档位噪声注入 + 衰减测量
│   └── unified_evaluation.py          # 模型无关的指标计算与绘图
└── outputs/
    ├── yamnet/
    │   ├── label_map.json             # 物种名 <-> 整数下标 映射
    │   ├── embeddings.npz             # YAMNet 嵌入缓存（gitignore）
    │   ├── yamnet_bird_model.keras    # 训练好的分类头（gitignore）
    │   ├── test_predictions.npz       # 留出测试集预测（gitignore）
    │   └── noise_results.npz          # 各 SNR 档准确率（gitignore）
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

依赖：`tensorflow>=2.10`、`tensorflow-hub`、`librosa`、`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`seaborn`。

> YAMNet 首次运行时自动从 TensorFlow Hub 下载（约 17 MB）。嵌入 + 分类头的工作流无需 GPU，CPU 即可胜任。

---

## 6. 使用方法

所有脚本使用相对于 `src/` 的路径，因此需在 **YAMNet/** 目录下的 `src/` 中运行。

```bash
cd src
```

### 6.1 训练 YAMNet 分类器

```bash
python yamnet_bird_pipeline.py
```

读取 `ml_cv_fold1_train/val.csv` 与 `ml_test.csv`，在挂载的 BirdCLEF 年度数据集目录下定位音频，提取并缓存 YAMNet 嵌入，训练分类头，将模型、标签映射与测试预测写入 `outputs/yamnet/`。

### 6.2 运行噪声鲁棒性实验

```bash
python noise_robustness_eval.py
```

复现训练时的测试集切分，在 5 / 0 / −5 dB 下注入高斯噪声，重新编码每条带噪波形，记录各档准确率。结果写入 `outputs/yamnet/noise_results.npz`，衰减曲线写入 `outputs/figures/noise_robustness.png`。

### 6.3 生成评估报告与图表

```bash
python unified_evaluation.py
```

计算准确率 / 精确率 / 召回率 / F1（macro 与 weighted）、逐类明细、混淆矩阵、多模型准确率对比、噪声衰减曲线。

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

YAMNet 在 Kaggle 上跑通完整 BirdCLEF 数据（1,229 类，fold1：train 3,824 / val 956 / test 1,196）。绝对准确率低是 1,229 类×每类约 3 条样本的长尾分布决定的，非 bug；作业关注三个模型在噪声下的**相对衰减趋势**，而非绝对分。

**干净条件性能（YAMNet）：**

| 指标 | 数值 |
|---|---|
| 准确率 | 0.0209（25/1196，约为随机基线 1/1229≈0.00081 的 25 倍） |
| Macro-F1 | 0.0150 |
| Weighted-F1 | 0.0186 |

**噪声鲁棒性衰减（YAMNet）：**

| SNR 档位 | 准确率 |
|---|---|
| clean（干净） | 0.0209 |
| 5 dB | 0.0059 |
| 0 dB | 0.00084（≈随机基线，模型基本失效） |
| −5 dB | 0.00167 |

准确率随噪声增强单调下降，0 dB 档即坍塌至随机水平，表明强噪声下的鲁棒性是主要瓶颈，为后续引入降噪前处理提供动机。

---

## 9. 局限与未来工作

- **长尾少样本**：1,229 类×每类约 3 条训练样本，过拟合明显，干净准确率仅 2.09%；长尾缓解（类别加权、focal loss）是提升方向。
- **噪声模型**：高斯白噪声为可控基线；替换为真实风声/雨声仅需改动噪声模块。
- **微调**：当前仅训练 YAMNet 分类头；端到端微调顶层卷积块是提升干净条件准确率的自然下一步。
- **可复现性**：当前仅 sklearn 切分设了固定种子，TF 随机数未设；重训会得到不同模型，补确定性种子可对齐数字。

---

## 10. 可复现性

- 数据加载、切分、训练、噪声注入全程使用固定随机种子。
- 分层切分确定性，噪声鲁棒性实验与干净基线操作于完全相同的测试片段。
- 嵌入缓存将昂贵的 YAMNet 前向计算与下游迭代解耦，确保重复实验既快又逐字节可复现。
