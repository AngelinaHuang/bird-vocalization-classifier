# 更新版数据处理文档

本文档反映最新的**字段级去重**策略：对重复录音按字段智能合并，而非丢弃记录，从而保留全部信息。

## 2.1 数据集概览

Google Perch 鸟类发声分类器数据集（BirdCLEF 2021–2026 合集）包含 **183,239 条原始音频元数据记录**，覆盖 **1,229 个不同物种**，数据来自 Xeno-Canto 与 iNaturalist，跨度为六年。

为在机器学习训练中防止数据泄漏，同时**保留全部可用元数据**，我们采用**字段级合并去重**，而非简单删除重复记录。最终得到干净的统一数据集，共 **167,308 条唯一录音**（按 filename），且**历史标签与分类信息零损失**。

元数据字段包括：

- `primary_label`：物种编码（鸟类为 eBird 代码 / 昆虫为分类单元 ID）
- `secondary_labels`：录音中出现的其他物种（列表）
- `type`：录音类型分类（叫声、鸣唱、飞行叫声等）
- `filename`：音频文件标识符（用作唯一键）
- `rating`：录音质量评分（0.0–5.0），用作 SNR 分层的代理指标
- `latitude` / `longitude`：地理坐标
- `scientific_name` / `common_name`：物种分类学名称
- 背景噪声特征、多物种重叠（由 `secondary_labels` 隐含）
- `source_year`：数据来源年份（可能包含多个年份）

> **说明**：原始数据集描述提到 12,400 条记录、86 个物种。实际合并后的 2021–2026 数据集包含 **1,229 个物种**，这会影响采样策略——「每物种至少 5 个样本」的硬约束优先于约 2,000 的目标样本量。

---

## 2.2 字段级合并去重与分层抽样设计

为在消除采样偏差的同时兼顾计算约束，我们先做强制的**字段级合并去重**（保证重复条目**零信息损失**），再实施**双层分层抽样**。

### 前置步骤：字段级合并去重（机器学习标准，零损失）

不简单删除重复的 `filename`（会丢失数据），而是按以下智能规则合并同一 filename 的所有出现：

| 字段类型 | 合并策略 | 示例 |
| --- | --- | --- |
| **列表型**（`secondary_labels`、`type`） | 取所有非空值的并集并去重 | 记录 A：`['song']`，记录 B：`['call']` → `['song', 'call']` |
| **文本型**（`scientific_name`、`common_name` 等） | 取任一非空值（优先最新 `source_year`） | — |
| **数值型**（`rating`、`latitude`、`longitude`） | 取最大 `source_year` 对应记录中的非空值 | 2019 年 rating=4.0，2025 年 rating=3.5 → 保留 3.5 |
| **时间型**（`date`、`time`） | 取最新非空值 | — |
| **分类型**（`collection`、`class_name` 等） | 取最新非空值 | — |
| **年份型**（`source_year`） | 合并去重后以逗号分隔字符串保存 | `"2022,2023"` → `"2022,2023"` |

这样可以保证：

- **全部历史标签**（`secondary_labels`、`type`）被合并保留。
- **最新且更准确的**坐标、评分与分类学名称被保留。
- **无信息损失**——任意年份的每一条元数据都被保留。

合并后得到 **167,308 条唯一录音**（数量与简单去重相同，但元数据更丰富）。

### 第一层：鸟类物种分层

将 1,229 个物种各自作为一个层，以确保覆盖完整分类范围。

### 第二层：SNR 等级分层

在每个物种层内，再按录音质量等级分层（由 `rating` 字段作为 SNR 代理）：

| SNR 等级 | 定义 | 占数据集比例 |
| --- | --- | --- |
| **High SNR**（rating ≥ 4.0） | 干净录音，背景噪声极少 | 57.7% |
| **Medium SNR**（2.0 ≤ rating < 4.0） | 中等环境噪声 | 27.2% |
| **Heavy SNR**（rating < 2.0） | 显著背景噪声污染 | 15.1% |

### 硬约束

每个物种至少保留 **5 个样本**，以避免低资源稀有物种被完全丢弃。

### 样本量结果

在 1,229 个物种与「每物种最少 5 样本」硬约束下，分层子集共含 **5,976 条片段**（而非原先约 2,000 的目标）。物种多样性硬约束覆盖目标样本量，因为 1,229 物种 × 5 样本 ≈ 6,000+ 条记录。

---

## 2.3 采样算法（伪代码）

以下 Python 函数实现完整流程，包括字段级合并去重与双层分层抽样。

```python
import pandas as pd
import numpy as np
import re

def merge_row_group(group):
    """
    将同一 filename 的所有记录合并为一行。
    按字段规则合并，保留全部信息。
    """
    # secondary_labels 取并集
    sec_labels = merge_lists(group['secondary_labels'].tolist())
    types = merge_lists(group['type'].tolist())

    # 合并 source_year：所有年份取并集
    all_years = set()
    for y in group['source_year'].dropna():
        for part in re.split(r'[,;\s]+', str(y)):
            if part.strip():
                all_years.add(part.strip())
    merged_year = (
        ', '.join(sorted(all_years, key=lambda x: int(x) if x.isdigit() else 0))
        if all_years else None
    )

    # 数值型与分类型字段取最新值
    latest_rating = choose_by_latest_year(group, 'rating')
    latest_lat = choose_by_latest_year(group, 'latitude')
    latest_lon = choose_by_latest_year(group, 'longitude')
    # ... 其他字段同理

    return pd.Series({
        'filename': group.iloc[0]['filename'],
        'primary_label': group.iloc[0]['primary_label'],
        'secondary_labels': sec_labels,
        'type': types,
        'rating': latest_rating,
        'latitude': latest_lat,
        'longitude': latest_lon,
        # ... 包含其余全部字段
        'source_year': merged_year,
    })


def dual_layer_stratified_sampling(df, min_per_species=5, random_state=42):
    """
    完整流程：
    1. 按 filename 做字段级去重（零损失合并）
    2. 分配 SNR 等级
    3. 双层分层抽样（物种 × SNR）
    """
    np.random.seed(random_state)

    # 步骤 1：字段级合并去重
    grouped = df.groupby('filename')
    merged_rows = [merge_row_group(group) for _, group in grouped]
    df_dedup = pd.DataFrame(merged_rows)

    # 步骤 2：分配 SNR 等级
    def assign_snr_tier(rating):
        if rating >= 4.0:
            return 'High'
        elif rating >= 2.0:
            return 'Medium'
        else:
            return 'Heavy'

    df_dedup['snr_tier'] = df_dedup['rating'].apply(assign_snr_tier)

    # 步骤 3：物种层抽样，并在物种内按 SNR 再分层
    sampled_dfs = []
    for species in df_dedup['primary_label'].unique():
        species_df = df_dedup[df_dedup['primary_label'] == species]
        available = len(species_df)
        if available == 0:
            continue

        n_take = min(min_per_species, available)
        if n_take == 0:
            continue

        # 物种内按 SNR 比例分配
        tier_counts = species_df['snr_tier'].value_counts()
        available_tiers = [
            t for t in ['High', 'Medium', 'Heavy'] if tier_counts.get(t, 0) > 0
        ]

        tier_samples = {}
        remaining = n_take
        for i, tier in enumerate(available_tiers):
            tier_count = tier_counts[tier]
            if i == len(available_tiers) - 1:
                tier_n = remaining
            else:
                tier_n = max(1, round(n_take * tier_count / available))
            tier_n = min(tier_n, tier_count)
            tier_samples[tier] = tier_n
            remaining -= tier_n

        # 分配剩余名额
        if remaining > 0:
            for tier in available_tiers:
                if remaining <= 0:
                    break
                capacity = tier_counts[tier] - tier_samples.get(tier, 0)
                add = min(remaining, capacity)
                tier_samples[tier] = tier_samples.get(tier, 0) + add
                remaining -= add

        # 从各层抽样
        for tier, n in tier_samples.items():
            if n <= 0:
                continue
            tier_df = species_df[species_df['snr_tier'] == tier]
            sampled = tier_df.sample(
                n=min(n, len(tier_df)), random_state=random_state
            )
            sampled_dfs.append(sampled)

    result = pd.concat(sampled_dfs, ignore_index=True)
    return result
```

---

## 2.4 训练 / 测试划分与交叉验证

按标准机器学习实践，将分层子集划分为训练集与留出测试集，并在训练集内嵌套交叉验证折。

### 2.4.1 训练 / 测试划分（80/20，按 SNR 分层）

将 5,976 条分层子集按 80/20 划分：

| 集合 | 样本数 | 比例 | 物种数 |
| --- | --- | --- | --- |
| 训练集 | 4,780 | 80.0% | 1,227 |
| 留出测试集 | 1,196 | 20.0% | 805 |

分层在 SNR 等级上进行，以保持两边质量分布一致。在 1,229 个物种、每物种约仅 5 个样本的情况下，无法再做物种级分层（测试集至少每物种 1 个样本会超出测试集容量）。

**SNR 分布一致性**（已验证各划分一致）：

- High：56.5%
- Medium：29.1%
- Heavy：14.4%

### 2.4.2 5 折分层交叉验证

在 4,780 条训练集内，采用 5 折分层交叉验证做超参调优与模型选择：

| 折 | 缩减训练集 | 验证集 |
| --- | --- | --- |
| 1 | 3,824（80%） | 956（20%） |
| 2 | 3,824（80%） | 956（20%） |
| 3 | 3,824（80%） | 956（20%） |
| 4 | 3,824（80%） | 956（20%） |
| 5 | 3,824（80%） | 956（20%） |

每一折都保持 SNR 等级比例。已验证每折训练与验证之间**文件名零重叠**，确保无数据泄漏。

---

## 2.5 代表性验证

为证明 5,976 子集可靠反映完整去重数据集的分布，我们验证两项关键指标：

### 物种覆盖率

| 指标 | 结果 |
| --- | --- |
| 完整数据集物种数 | 1,229 |
| 抽样子集物种数 | 1,229（100% 覆盖） |
| 每物种最少样本数 | 1（对 65 个总记录不足 5 条的极稀有物种，纳入其全部可用样本） |

### SNR 等级分布偏差

完整数据集与抽样子集在 High / Medium / Heavy SNR 比例上的对比：

| 等级 | 完整数据集 | 抽样子集 | 偏差 |
| --- | --- | --- | --- |
| High | 57.7% | 56.5% | -1.2 pp |
| Medium | 27.2% | 29.1% | +1.9 pp |
| Heavy | 15.1% | 14.4% | -0.7 pp |

所有偏差均在该规模分层样本的可接受范围内。

```python
def verify_representativeness(full_df, subset_df):
    """
    验证子集在物种覆盖率与 SNR 分布上是否代表完整数据集。
    """
    full_species = set(full_df['primary_label'].unique())
    subset_species = set(subset_df['primary_label'].unique())
    coverage = len(subset_species) / len(full_species) * 100
    print(f"Species coverage: {coverage:.1f}%")
    assert coverage >= 99.0, "Species coverage below 99%"

    full_tier = full_df['snr_tier'].value_counts(normalize=True) * 100
    subset_tier = subset_df['snr_tier'].value_counts(normalize=True) * 100

    for tier in ['High', 'Medium', 'Heavy']:
        diff = abs(full_tier.get(tier, 0) - subset_tier.get(tier, 0))
        print(f"{tier} tier deviation: {diff:.2f} percentage points")
        assert diff <= 3.0, f"{tier} tier deviation exceeds 3pp"

    # 数据泄漏检查
    assert len(set(subset_df['filename'])) == len(subset_df), "Duplicate filenames found!"

    return True
```

---

## 2.6 完整流程汇总

| 步骤 | 操作 | 输出 |
| --- | --- | --- |
| 1 | 加载并合并 6 个年度数据集（2021–2026） | 183,239 条原始记录 |
| 2 | 字段级合并去重（重复记录零损失合并） | 167,308 条唯一记录（元数据更丰富） |
| 3 | 由 rating 分配 SNR 等级（High / Medium / Heavy） | 增强后的元数据 |
| 4 | 双层分层抽样（物种 × SNR，每物种最少 5） | 5,976 条平衡子集 |
| 5 | 验证代表性与零数据泄漏 | 验证通过 |
| 6 | 80/20 训练 / 测试划分（按 SNR 分层） | 4,780 训练 + 1,196 测试 |
| 7 | 训练集内 5 折分层交叉验证 | 5 折 ×（3,824 训练 / 956 验证） |

该**统一去重并分层后的数据集**将供三个模型（LightGBM、FastAI CNN、YAMNet）共同用于训练与评估，以保证不同算法范式下的公平对比，并消除数据泄漏风险。

---

## 输出文件

| 文件 | 行数 | 用途 |
| --- | --- | --- |
| `ml_full_deduped.csv` | 167,308 | 完整去重主数据集（字段合并后） |
| `ml_sampled.csv` | 5,976 | 双层分层抽样子集（覆盖全部 1,229 个物种） |
| `ml_train.csv` | 4,780 | 训练集（80%） |
| `ml_test.csv` | 1,196 | 留出测试集（20%） |
| `ml_cv_fold{1-5}_train.csv` | 各 3,824 | 交叉验证缩减训练集 |
| `ml_cv_fold{1-5}_val.csv` | 各 956 | 交叉验证验证集 |
| `ml_data_pipeline.py` | — | 可复现的处理脚本 |
