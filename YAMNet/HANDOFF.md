# YAMNet 鸟鸣分类 · 排查交接文档

> 用途：新开对话时把这份文档丢给助手，可直接接续排查/修复，无需重新摸底。

---

## 1. 项目背景（一句话）

小组作业，对比 3 个模型（LightGBM / FastAI / YAMNet）做鸟类声音分类，重点是**噪声鲁棒性**。本会话负责 **YAMNet 部分**。用户是纯小白，代码要尽量稳、报错信息要可读。

- 仓库目录：`E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\`
- 正式数据：`E:\stevensprogram\MLwork\YAMNet-kaggle\data\data\`（只有 CSV 元数据，无音频）
- 数据说明文档：`E:\stevensprogram\MLwork\YAMNet-kaggle\Data Processing Documentation.md`
- 本地产物目录：`E:\stevensprogram\MLwork\results\yamnet\`（从 Kaggle 下载的结果 + 本地出的图）
- 运行平台：**Kaggle notebook**（不在本地跑正式数据，因为音频几十 GB）

## 2. 数据结构（已查清，关键事实）

正式 CSV 列：`filename, secondary_labels, type, primary_label, rating, latitude, longitude, scientific_name, common_name, author, license, url, date, time, collection, class_name, inat_taxon_id, source_year, snr_tier`

- `primary_label` = eBird 代码（如 `mastit1`、`incdov`、`elepai`），共 **1229 类**
- `filename` 是音频在年度 train_audio 里的相对路径：
  - 2021：平铺 `XC243160.ogg`
  - 2022+：带前缀 `mastit1/XC125979.ogg`
  - iNat：`65962/iNat81325.ogg`
- `source_year` 多为单年，去重合并的会出现 `"2025,2026"`

**数据切分（组里已切好，直接用，不能改）**：
| 文件 | 行数 | 用途 |
|---|---|---|
| `ml_sampled.csv` | 5976 | 双层采样子集（1229 种全覆盖，每种 ≥5） |
| `ml_train.csv` | 4780 | 80% |
| `ml_test.csv` | 1196 | 20% 留出测试 |
| `ml_cv_fold1_train.csv` | 3824 | fold1 训练 |
| `ml_cv_fold1_val.csv` | 956 | fold1 验证 |

YAMNet 用 **fold1**：train=3824 / val=956 / test=1196。1229 类 ÷ 3824 训练样本 ≈ 3.1 条/类，准确率必然偏低，这是数据设计决定，符合 README 的"长尾"叙事。

## 3. Kaggle 实际挂载布局（用户实跑确认）

```
/kaggle/input/
├── competitions/
│   ├── birdclef-2021/train_short_audio/
│   ├── birdclef-2022/train_audio/
│   ├── birdclef-2023/train_audio/
│   ├── birdclef-2024/train_audio/
│   ├── birdclef-2025/train_audio/
│   └── birdclef-2026/train_audio/
└── datasets/
    └── jennyyila/          ← 用户的 CSV 在这里（ml_*.csv 三个）
```

**坑点**：数据集是**嵌套**的（`competitions/birdclef-20XX/...`），不是直接 `/kaggle/input/birdclef-20XX/`。CSV 也在深一层 `datasets/jennyyila/`。

## 4. 代码现状（本会话已改完，已过语法校验 + 逻辑单测）

### `src/yamnet_bird_pipeline.py`（主文件）
- `Config`：`CSV_DIR`/`KAGGLE_INPUT`/`AUDIO_ROOT_CANDIDATES=("train_audio","train_short_audio")`；`OUT_DIR` 自动判 Kaggle→`/kaggle/working/yamnet`，本地→`../outputs/yamnet`。
- **核心新增**：`_scan_inputs(kaggle_input)` —— 一次 `os.walk` 扫遍 `/kaggle/input`，**遇音频目录剪枝**（不遍历几十万 .ogg，很快），同时收集：`year2root`（年份→音频根）、`unknown_roots`、`csv_paths`（filename→Path）。已用模拟用户嵌套布局单测过，6 年 + 3 CSV 全命中。
- `print_mounted_inputs()`：列挂载内容方便自查。
- `parse_source_years(s)`：解析 `"2025,2026"`→`[2026,2025]`（最新优先）。
- `resolve_audio_path(row, audio_roots)`：每行按 source_year 逐个试 3 种候选路径：`root/filename`、`root/primary_label/basename`、`root/basename`；source_year 缺失则用所有已知年份兜底。已单测通过（2021 平铺、2022+ 前缀、iNat、跨年、缺失文件）。
- `load_csv_splits(cfg)`：调 `_scan_inputs` → 打印挂载自检 + 音频根 → 读 3 个 CSV → 逐行 resolve → 返回 `(df_train, df_val, df_test, missing)`。CSV 找不到会列出挂载内容 + 明确报错。
- `preflight_report(missing, ...)`：打印缺失数 + 示例，剔除缺失行。
- `build_embeddings_for_splits(df_train, df_val, df_test, label2idx)`：合并 3 split 的 filename 去重（5976 个唯一），缓存按 **filename 索引**（npz 存 `filenames`/`X`/`y`），缺哪条补算哪条再回写。返回 `X_train,y_train,X_val,y_val,X_test,y_test,test_filenames`。
- `main_csv()`：load_csv_splits → 预检 → 标签用 eBird 代码（train+val+test 并集，1229 类）→ 存 `label_map.json` → embedding → 训练 → 存 `test_predictions.npz`（`classes` 是 eBird 代码数组，与 LightGBM/FastAI 对齐）。单折入口, 保留供手动验证。
- **5 折交叉验证（2026-07-13 新增, 详见 §5.3）**：`Config.fold_csvs(fold)`/`fold_dir(fold)`/`NOISE_EMBED_CACHE`；`load_csv_splits` 加 `train_csv/val_csv/test_csv/scan` 可选参数（None 用 fold1 默认, 向后兼容）；新增 `main_cv_all_folds(n_folds=5)`（一次扫描 + 共享 label_map + 5 折训练循环 + clean 汇总, 每折存 `fold{N}/` 模型与 `test_predictions.npz`, 写 `cv_per_fold.csv`/`cv_summary.csv`）。`main()` 改调 `main_cv_all_folds()`。`_scan_inputs` 的 `target_csvs` 收录名单已扩到 fold1~5 全部 train/val（见 §5.3 bug 修复）。
- `main()`：直接调 `main_cv_all_folds()`（早期本地 folders 玩具模式及其 `build_dataframe_from_folders`/`RAW_DATA_DIR`/`DATA_MODE` 已于 2026-07-13 一并删除, 正式跑走 5 折 CSV 模式）。
- 复用未动：`load_waveform`（16k 单声道 + 中段截断/补零 + 峰值归一）、`extract_embedding`（帧均值→1024 维）、`build_classifier`（Dense256→Dropout→Dense(n,softmax)）、`load_yamnet`。

### `src/noise_robustness_eval.py`
- 删了 `reproduce_test_split`（测试集就是 `ml_test.csv`，不再随机复现）。
- `main()`：`load_csv_splits` 取 `df_test` → 读 `label_map.json` + 模型 + YAMNet → clean 档复用缓存 embedding，3 噪声档（5/0/-5dB）叠噪重新 `extract_embedding` → 存 `noise_results.npz` + `plot_noise_decay`。
- import 已更新为 `Config, load_csv_splits, preflight_report, build_embeddings_for_splits, load_waveform, load_yamnet, extract_embedding`。

### `src/unified_evaluation.py`
- `plot_confusion_matrix` **已修 bug**：原函数收了 `top_n_classes` 参数却没用，会画 1229×1229。现已实现真正的 top-N 截断（取真实标签样本最多的前 N 类，预测落在前 N 类之外的计入末列"其它"），≤30 类时格子标数字。`demo()` 传 `top_n_classes=30`。
- 其余函数不动。

### `src/measure_inference.py`（2026-07-14 新增）
- 在 Kaggle notebook 全流程跑完后作为 cell4 运行，测量推理速度与显存。
- 测量策略：YAMNet 编码器测一次（5 折共享，冻结参数），分类头逐折加载测量，5 折 mean±std。
- 流程：预加载 50 条波形 → 预热 5 次 → YAMNet 编码器测 50 次 → 预计算嵌入缓存 → 逐折加载 fold1..5 模型测分类头 → 汇总端到端延迟 → 测 GPU 显存。
- 输出：`inference_metrics.csv`（汇总） + `inference_details.csv`（逐折细节）。
- 用法：`%run -i src/measure_inference.py`（Kaggle cell4）。

### `src/measure_inference_template.py`（2026-07-14 新增）
- 给 LightGBM / FastAI 队友的通用推理速度测量模板。
- 队友只需改三个函数：`load_model()` / `load_and_preprocess()` / `predict_one()`。
- 测量逻辑（预热、计时、统计、存 CSV）与 YAMNet 一致，确保三模型对比公平。

## 5. 当前进度

**状态：5 折交叉验证已跑完，结果已下载到本地。**

### 5.0 5 折交叉验证结果（2026-07-14 实跑）

| 指标 | Mean ± Std | 说明 |
|------|-----------|------|
| clean | 1.59% ± 0.27% | 约为随机基线的 20 倍，比单折(2.09%)低是正常的——fold1 恰好偏优 |
| 5 dB | 0.47% ± 0.16% | 噪声下明显衰减 |
| 0 dB | 0.30% ± 0.10% | 接近随机基线 |
| −5 dB | 0.07% ± 0.06% | 基本随机 |

5 折 clean 准确率逐折：fold1=1.76%, fold2=1.09%, fold3=1.51%, fold4=1.76%, fold5=1.84%。

**结论不变**：绝对准确率低是 1229 类×每类约 3 样本的长尾数据决定的，非 bug；噪声下衰减明显，0dB 即塌成随机。现在有 mean±std 可以与 LightGBM/FastAI 公平对比。

**推理速度与显存**：代码已写好（`src/measure_inference.py`），待下次上 Kaggle 作为 cell4 运行后下载结果。

### 产物清单（已下载到 `results/yamnet/`）

```
results/yamnet/
├── embeddings.npz              # 5976 条嵌入缓存
├── noise_embeddings.npz        # 噪声嵌入缓存 (5 折共享)
├── label_map.json              # 1229 类映射
├── cv_per_fold.csv             # 每折 clean 准确率
├── cv_noise_per_fold.csv       # 每折各 SNR 档准确率
├── cv_summary.csv              # 5 折汇总 mean±std (clean + 噪声)
├── fold1/                      # 各折独立模型 + 预测 + 噪声结果
│   ├── yamnet_bird_model.keras
│   ├── test_predictions.npz
│   └── noise_results.npz
├── fold2/ ... fold5/           # 同上
```

### 5.1 80/20 一致性清理（2026-07-13）

组里按课上要求把数据处理方式从 70/15/15 换成 80/20 分层。排查全项目后改动如下：
- `README.md` / `README_zh.md` 第 29 行切分描述：70/15/15 → 80/20 分层。
- `Data Processing Documentation.md` 及中文版：本就是 80/20，无需改。
- **`main_csv()`（Kaggle 正式跑的函数）未动**——它直接读预切好的 CSV，本身不做切分。
- **结论：Kaggle 不用重跑**，产物全部有效。仅文档改动。

### 5.2 删除早期自找数据（2026-07-13）

项目最初用 Xeno-Canto 自找 4 种鸟音频，后切到 Kaggle BirdCLEF。已清理全部代码与文档残留，`data/raw/` 音频保留本地不物理删除。详见原 HANDOFF §5.2。

### 5.3 5 折交叉验证（2026-07-13 实现 → 2026-07-14 实跑完成）

代码改动见 §4。fold2~5 CSV 扫描 bug 已修。**2026-07-14 实跑成功**，5 折全部训完，噪声评估全部完成，产物已下载到本地 `results/yamnet/`。

## 6. 已知风险 / 待观察

1. **1229 类只 3.1 样本/类**：过拟合，准确率低（clean 5 折均值=1.59%），非 bug。已实跑确认。LightGBM 同款结果（0.48%）也印证了数据约束是主因。
2. **2026 iNat 音频**目录结构无法提前核实，预检会列缺失，剔除后不影响跑通。实跑未见大面积缺失。
3. **embedding 缓存匹配逻辑**：`build_embeddings_for_splits` 读缓存按 filename 判断，filename 集合变化时补算缺失项。实跑命中正常。
4. **noise_robustness_eval 依赖 `test_filenames` 顺序**与 `df_test` 行顺序一致——实跑通过，`noise_results.npz` 产出正常。
5. **训练不可复现**：每折设了 `tf.random.set_seed(SEED+fold)` 等种子（best-effort），已在 5 折中生效。
6. **5 折交叉验证已完成（2026-07-14）**：见 §5.0。产物在 `results/yamnet/`。
7. **下游对比**：LightGBM 结果已拿到（0.48% clean），FastAI 的结果待队友完成。三者叠加衰减曲线待补。队友需提供数据清单见 §8。
8. **端到端微调已实现（2026-07-16）**：代码在 `src/yamnet_finetune_e2e.py` + `noise_eval_e2e.py` + `measure_inference_e2e.py`，待 Kaggle 实跑。详见 §9。
9. **端到端显存风险**：batch_size=8 时如果 OOM 可降到 4。YAMNet 前向 + 反向传播占显存较大。
10. **端到端训练时间**：预计 30-60 min/fold，5 折 2.5-5 小时，Kaggle 12h session 限制够用。

## 7. 接续排查时的建议

- 用户每次贴报错，先看是不是数据没接上（看「挂载自检」「音频发现」「CSV」打印段）。
- 若 embedding 卡住，确认 GPU/Internet 开了，且 YAMNet 下载成功。
- 若某年份音频大面积缺失，让用户贴 `ls /kaggle/input/competitions/birdclef-20XX/` 真实结构，再扩 `AUDIO_ROOT_CANDIDATES` 或候选路径。
- **notebook 跑法（策略一：冻结，已跑完）**：四个 `.py` 在 Kaggle 单 notebook 里按顺序跑：
  1. cell1: pipeline（`main()` 现跑 5 折训练）
  2. cell2: unified_evaluation（**注释掉 `demo()`**，只定义函数）
  3. cell3: noise_robustness_eval（`__main__` 现调 `main_cv()`，5 折噪声评估）
  4. cell4: `%run -i src/measure_inference.py`（推理速度 + 显存测量）
- **notebook 跑法（策略二：端到端微调，待跑）**：三个 cell，**与策略一独立，不需先跑策略一**：
  1. cell1: `%run -i src/yamnet_finetune_e2e.py`（5 折端到端训练，30-60 min/fold）
  2. cell2: `%run -i src/noise_eval_e2e.py`（噪声评估，波形直通 e2e 模型）
  3. cell3: `%run -i src/measure_inference_e2e.py`（推理速度 + 显存）
- **不想重训时**：注释掉 cell1 的 `if __name__=="__main__": main()`，只定义函数；cell3/4 直接读 `/kaggle/working/yamnet/` 里的现成模型 + 缓存。先 `ls` 确认产物在不在。
- **端到端 OOM 处理**：如果训练时显存不足，把 `FinetuneConfig.BATCH_SIZE` 从 8 改到 4。
- **训练不可复现**：代码只给 sklearn 的 split 设了 `SEED=42`，没设 `tf.random.set_seed`/`np.random.seed`/`PYTHONHASHSEED`，重训必得不同模型。若要可复现需补确定性种子（用户尚未决定是否做）。
- 代码改动遵循：先解释再动手（用户要求），改前可用 EnterPlanMode 列计划。
- 本地无 tensorflow（有 numpy/pandas/matplotlib/seaborn/sklearn），验证只能 `python -m py_compile` + 抽逻辑单测；出图/读 npz 可本地做。
- **三个 src/*.py 的注释已重写为正式中文**（去 AI 味：删了"大白话"、教程腔、感叹号、"你只需要改"等），代码逻辑未动。`make_figures.py`（在 results 目录）未重写注释，不在提交范围。

## 8. 队友数据需求清单 + 噪声实验规范

YAMNet 的 unified_evaluation.py 是模型无关的，只需队友提供各模型的最终输出即可生成全部对比图。**队友不需要跑 unified_evaluation.py**，只需按以下规范提供数据。

### 队友需提供的数据（LightGBM 和 FastAI 各一份）

| # | 数据 | 格式 | 说明 |
|---|------|------|------|
| 1 | 测试集预测 | `.npz` 或 `.csv`，含 `y_true`(真实标签)、`y_pred`(预测标签)、`class_names`(类别名列表) | 必须在同一个 `ml_test.csv` 上评估 |
| 2 | 噪声鲁棒性 | 四个 SNR 档（clean / 5dB / 0dB / −5dB）的准确率，各 4 个数字 | 必须用同样的噪声叠加方式（见下方规范） |
| 3 | 推理延迟 | 单条样本平均推理时间，单位毫秒 | 预热 5 次后测 50 次取平均 |
| 4 | 显存占用 | GPU 显存占用（MB），若用 GPU 的话 | 可用 `nvidia-smi` 或框架自带 API 查看 |

### 噪声实验规范（三个模型必须一致，否则对比不公平）

```
噪声类型:    高斯白噪声 (Gaussian white noise)
SNR 公式:    SNR_dB = 10 × log₁₀(P_signal / P_noise)
             P_signal = mean(waveform²)  ← 信号平均功率
             P_noise  = P_signal / (10^(SNR/10))  ← 由目标 SNR 反推
             noise    = random.normal(0, sqrt(P_noise), size=waveform.shape)
             
测试档位:    clean (不叠噪) / 5 dB / 0 dB / −5 dB
叠噪后处理: 峰值归一化到 [-1, 1]
随机种子:   固定种子 (如 42)，保证噪声可复现
测试集:     ml_test.csv (1196 条)，与干净基线完全一致
```

**给 LightGBM 队友的提示**：叠噪后需要重新提取手工特征（和训练时一样的特征提取流程），然后用训练好的 LightGBM 模型预测。

**给 FastAI 队友的提示**：叠噪后需要重新生成 mel-spectrogram 图片，然后用训练好的 FastAI 模型预测。

### 队友给了数据后，你这边做的事

1. 把三方的 `(y_true, y_pred)` 填入 `unified_evaluation.py`，生成：
   - 三模型准确率柱状图（accuracy / macro-F1 / weighted-F1）
   - 三模型混淆矩阵
   - 三模型噪声衰减曲线（同框对比）
2. 测量/收集三模型的推理延迟和显存，做性能对比表
3. 低资源泛化分析：按每类训练样本数分组（如 1-3 条 / 4-6 条 / 7+ 条），看三模型在各组的准确率

---

## 9. 端到端微调（已实现，待 Kaggle 实跑）

### 当前策略 vs 端到端策略

| | 策略一（已完成） | 策略二（已实现，待跑） |
|---|---|---|
| 做法 | 冻结 YAMNet → 预计算嵌入 → 训分类头 | 解冻 YAMNet 顶层 → 音频直通 → 联合训练 |
| YAMNet 参数 | 0 可训练 | 全部可训练 (差分学习率保护) |
| 输入 | 1024 维向量（预计算） | 原始波形 (16000×5) |
| 每轮计算 | 极快（读缓存） | 慢（每批要过 YAMNet） |
| 显存 | 极小 | 较大（需 GPU, batch=8） |
| 拟合能力 | 弱（只调分类头） | 强（可调整特征提取） |
| 增强 | 无 | MixUp (alpha=0.2) + 类别平衡权重 |
| 训练循环 | model.fit | 自定义 E2ETrainer (差分学习率) |

### 实现的三个新文件

| 文件 | 说明 |
|---|---|
| `src/yamnet_finetune_e2e.py` | 端到端微调主管道 (5 折 CV + MixUp + 差分学习率) |
| `src/noise_eval_e2e.py` | 端到端模型噪声评估 (波形直通, 不用嵌入缓存) |
| `src/measure_inference_e2e.py` | 端到端推理速度测量 |

### 关键设计

1. **差分学习率**: YAMNet 变量用 lr=1e-5, 分类头用 lr=1e-3, 通过两个 Adam optimizer 分别 apply_gradients
2. **MixUp**: 每批随机混合两条波形 (Beta(0.2,0.2)), 标签按比例混合, 长尾分类关键技巧
3. **波形缓存**: `waveforms_cache.npz` 缓存预处理后的波形, 避免每 epoch 重复磁盘 I/O, 跨折复用
4. **输出隔离**: 存入 `e2e/fold{N}/` 不覆盖旧版冻结产物, 便于直接对比

### Kaggle 运行方式 (3 个 cell)

```
cell1: %run -i src/yamnet_finetune_e2e.py       # 训练 + 测试预测 (30-60 min/fold)
cell2: %run -i src/noise_eval_e2e.py             # 噪声评估 (4 档 SNR × 5 折)
cell3: %run -i src/measure_inference_e2e.py      # 推理速度 + 显存
```

### 超参数 (FinetuneConfig)

| 参数 | 值 | 说明 |
|---|---|---|
| BATCH_SIZE | 8 | YAMNet 前向占显存大 |
| EPOCHS | 40 | 早停 patience=8 |
| HEAD_LR | 1e-3 | 分类头学习率 |
| TOP_LAYER_LR | 1e-5 | YAMNet 顶层学习率 |
| MIXUP_ALPHA | 0.2 | MixUp Beta 分布参数 |
| UNFREEZE_LAYERS | 6 | 解冻 YAMNet 顶层层数 (仅供参考, 实际整体解冻+差分lr) |

### 预期结果

- clean 准确率: 预期 3-8% (旧版 1.91%), 解冻后特征提取可适配鸟鸣
- 噪声衰减: MixUp 增强应使噪声下表现更平缓
- 推理速度: 预期 80-120 ms/条 (与旧版 86 ms 接近, 因分类头结构不变)

---

## 10. 关键文件路径速查

| 文件 | 说明 |
|---|---|
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\yamnet_bird_pipeline.py` | 主管道（5 折 CV 已实现并跑通） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\noise_robustness_eval.py` | 噪声评估（含 5 折 main_cv 模式） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\unified_evaluation.py` | 统一评估框架（模型无关，待三方数据填入） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\measure_inference.py` | 🆕 推理速度测量（Kaggle cell4，5 折 mean±std） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\measure_inference_template.py` | 🆕 队友推理速度模板（改三个函数即可用） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\yamnet_finetune_e2e.py` | 🆕 端到端微调主管道（解冻 YAMNet + MixUp + 差分学习率） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\noise_eval_e2e.py` | 🆕 端到端模型噪声评估 |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\measure_inference_e2e.py` | 🆕 端到端推理速度测量 |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\results\yamnet\cv_summary.csv` | 5 折汇总 mean±std（clean + 噪声） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\results\yamnet\fold1..5\` | 5 折独立产物（模型/预测/噪声结果） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\results\yamnet\make_figures.py` | 本地出图脚本（已更新为 5 折 mean±std 版） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\README_zh.md` | 项目说明 |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\README.md` | 项目说明英文版 |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\data\data\ml_*.csv` | 正式数据（含 fold1~5 全部 CSV） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\HANDOFF.md` | 本文档 |
