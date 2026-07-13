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

## 5. 当前进度

**状态：YAMNet 部分全流程跑通，结果已下载到本地。**

Kaggle 上按 `cell1: pipeline(训练) → cell2: unified_evaluation(仅定义函数) → cell3: noise_robustness_eval(噪声评估)` 顺序跑完。产物下载到 `E:\stevensprogram\MLwork\results\yamnet\`：
- `yamnet_bird_model.keras`（模型）
- `embeddings.npz`（5976 条嵌入缓存）
- `label_map.json`（1229 类映射）
- `test_predictions.npz`（干净测试预测，含 y_true/y_pred/classes/test_filenames）
- `noise_results.npz`（4 档 SNR 预测与准确率）

**实跑结果（关键数字）**：

| SNR | accuracy | 说明 |
|---|---|---|
| clean | 0.0209（25/1196） | 干净基线，约为随机(1/1229≈0.00081)的 25 倍 |
| 5 dB | 0.0059 | 开始衰减 |
| 0 dB | 0.00084 | ≈ 随机基线，模型基本失效 |
| −5 dB | 0.00167 | 随机附近（小样本抖动，非单调） |

分类指标（clean）：accuracy=0.0209、macro-F1=0.0150、weighted-F1=0.0186。

**结论**：绝对准确率低是 1229 类×每类约 3 样本的长尾数据决定的，非 bug；噪声下衰减明显，0dB 即塌成随机。作业比的是三个模型的相对衰减趋势，不是绝对分。

**本地出图**（`make_figures.py`，在 `results/yamnet/`，不依赖 tf）：
- `noise_robustness_YAMNet.png`（4 档衰减曲线 + 随机基线虚线）
- `confusion_matrix_YAMNet_top30.png`（top-30 类混淆矩阵，含"其它"列）
- `metrics_YAMNet.csv` / `noise_accuracy_YAMNet.csv`（指标表，可直接抄进报告）

> Kaggle 上 `plot_noise_decay`/`plot_confusion_matrix` 默认存到 `../outputs/figures`（相对 cwd），在 Kaggle 上会落到 `/kaggle/outputs/figures`，**不在 notebook 输出下载范围**——所以图本地重出。本地用 `make_figures.py` 即可，不用回 Kaggle。

### 5.1 80/20 一致性清理（2026-07-13，本会话）

组里按课上要求把数据处理方式从 70/15/15 换成 80/20 分层。排查全项目后改动如下：
- `README.md` / `README_zh.md` 第 29 行切分描述：70/15/15 → 80/20 分层。
- `Data Processing Documentation.md` 及中文版：本就是 80/20，无需改。
- **`main_csv()`（Kaggle 正式跑的函数）未动**——它直接读预切好的 `ml_cv_fold1_train/val.csv` + `ml_test.csv`，本身不做切分，这些 CSV 本就按 80/20 切的。
- **结论：Kaggle 不用重传、不用重跑**，`results\yamnet\` 里的产物全部有效。仅文档改动，不影响已跑结果。

### 5.2 删除早期自找数据（2026-07-13，本会话）

项目最初用的是从 Xeno-Canto 自行下载的 4 种鸟音频（`data/raw/<species>/XC*.wav|mp3`，即 README 旧 §8 的 preliminary 结果来源），后切到 Kaggle BirdCLEF 正式数据。本次清理删掉该自找数据在代码与文档中的全部残留：
- **代码**（`yamnet_bird_pipeline.py`）：删 `Config.DATA_MODE`/`RAW_DATA_DIR`、`build_dataframe_from_folders()`、`main_folders()`、`main()` 的 folders 分支、文件头两模式 docstring、250 行 `DATA_MODE='folders'` 提示；章节号 4–8 重排为 4–7。`main()` 现直接调 `main_csv()`。
- **文档**：`README.md`/`README_zh.md` §2 Source 改写为 Kaggle CSV、§4 结构树删 `raw/` 行、§6.1 改扫描描述、§8 preliminary(4物种/0.90) 换成真实 Kaggle 数字（clean 2.09% 等）、§9 删 small/balanced 子集句。
- **音频文件**：`data/raw/` 下 4 种鸟音频**保留在本地**（已被 `.gitignore` 的 `data/raw/*` 排除，不会随 git 上传同步），不物理删除。
- **结论：Kaggle 不用重传、不用重跑**，`main_csv()` 路径未变，`results\yamnet\` 产物仍有效。

### 5.3 5 折交叉验证（2026-07-13，本会话）

队友要求跑完 5 折取 `mean±std` 才能与 LightGBM/FastAI 公平对比（§6.6 标注的"可选增强"现落实）。改动见 §4。**关键有利事实**：5 折只是把 `ml_train.csv`(4780) 重切不同 train/val 归属, 样本仍那 4780 条; 测试集 `ml_test.csv`(1196) 全程不变; fold1 跑时 embedding 缓存已覆盖 `train+val+test=5976=ml_sampled.csv` 全部, 故 fold2~5 embedding 全命中缓存只重训轻量分类头。噪声嵌入与模型无关, 算一次被 5 折复用。

**Kaggle 跑法（cell 分工不变）**：cell1 `main()` 现跑 `main_cv_all_folds()`; cell2 `unified_evaluation` 只定义函数; cell3 `noise_robustness_eval` 改调 `main_cv()`（算一次噪声嵌入 → 5 折各预测 → 噪声 mean±std + 衰减曲线）。

**首次 Kaggle 实跑踩坑（已修）**：fold1 训完进 fold2 时报 `FileNotFoundError: 找不到 CSV: ['ml_cv_fold2_train.csv','ml_cv_fold2_val.csv']`。根因是 `_scan_inputs` 的 `target_csvs` 名单只含 `{TRAIN_CSV,VAL_CSV,TEST_CSV}`（fold1 三个名字）, 扫描器遍历挂载目录时**只收录名单内的文件名**, fold2~5 的 CSV 即便已挂载也被跳过, 退回扫描结果时找不到 → 报错。修复：名单扩为 fold1~5 全部 train/val（`for _fold in range(1,6): target_csvs.update(cfg.fold_csvs(_fold))`）。本地 `py_compile` 通过; **Kaggle 重跑待用户确认 fold2~5 CSV 已挂载后执行**。

**重跑前自检**（确认 fold2~5 是否真传了, 免白跑 16 分钟）：
```bash
!find /kaggle/input -name 'ml_cv_fold*.csv' | sort
```
应看到 10 个 fold CSV（fold1~5 各 train/val）。缺则先传数据集, 不缺则贴改好的 cell1 重跑。

**预期产物**：`fold1..5/` 各含 `yamnet_bird_model.keras`+`test_predictions.npz`+`noise_results.npz`; 根目录 `cv_per_fold.csv`(fold,clean_acc)、`cv_summary.csv`(各指标 mean/std, cell1 写 clean 行、cell3 追加噪声行)、`noise_embeddings.npz`。旧的单折 flat 产物（根目录 `yamnet_bird_model.keras` 等）遗留不动。后续本地 `make_figures.py` 需小改为读 `cv_per_fold.csv`+5 折 `noise_results.npz` 出 mean±std 图（不在提交范围, 跑完 Kaggle 下产物后单独做）。

## 6. 已知风险 / 待观察

1. **1229 类只 3.1 样本/类**：过拟合，准确率低（clean=2.09%），非 bug。已实跑确认。
2. **2026 iNat 音频**目录结构无法提前核实，预检会列缺失，剔除后不影响跑通。实跑未见大面积缺失。
3. **embedding 缓存匹配逻辑**：`build_embeddings_for_splits` 读缓存按 filename 判断，filename 集合变化时补算缺失项。实跑命中正常。
4. **noise_robustness_eval 依赖 `test_filenames` 顺序**与 `df_test` 行顺序一致——实跑通过，`noise_results.npz` 产出正常。
5. **训练不可复现**：仅 sklearn split 设了种子，TF 随机未设；重训得不同模型。若小组成员需对齐数字需补确定性种子（待定）。
6. **5 折交叉验证已实现（2026-07-13）**：见 §5.3。`main_cv_all_folds()` 跑 5 折取 `mean±std`; Kaggle 重跑待用户确认 fold2~5 CSV 已挂载。早期"只用 fold1 的单点准确率"已升级为 5 折口径。
7. **下游对比未做**：LightGBM / FastAI 的同款噪声结果还没拿到，三者叠加衰减曲线待补。

## 7. 接续排查时的建议

- 用户每次贴报错，先看是不是数据没接上（看「挂载自检」「音频发现」「CSV」打印段）。
- 若 embedding 卡住，确认 GPU/Internet 开了，且 YAMNet 下载成功。
- 若某年份音频大面积缺失，让用户贴 `ls /kaggle/input/competitions/birdclef-20XX/` 真实结构，再扩 `AUDIO_ROOT_CANDIDATES` 或候选路径。
- **notebook 跑法**：三个 `.py` 在 Kaggle 单 notebook 里按顺序跑：cell1 pipeline（`main()` 现跑 5 折训练）→ cell2 unified_evaluation（**注释掉 `demo()`**，只定义函数）→ cell3 noise_robustness_eval（`__main__` 现调 `main_cv()`，5 折噪声评估）。`noise_robustness_eval` 开头有 try/except 兼容层：能 import 就 import，import 不到就复用前序 cell 已定义的命名空间。
- **不想重训时**：注释掉 cell1 的 `if __name__=="__main__": main()`，只定义函数；cell3/4 直接读 `/kaggle/working/yamnet/` 里的现成模型 + 缓存。先 `ls` 确认产物在不在。
- **训练不可复现**：代码只给 sklearn 的 split 设了 `SEED=42`，没设 `tf.random.set_seed`/`np.random.seed`/`PYTHONHASHSEED`，重训必得不同模型。若要可复现需补确定性种子（用户尚未决定是否做）。
- 代码改动遵循：先解释再动手（用户要求），改前可用 EnterPlanMode 列计划。
- 本地无 tensorflow（有 numpy/pandas/matplotlib/seaborn/sklearn），验证只能 `python -m py_compile` + 抽逻辑单测；出图/读 npz 可本地做。
- **三个 src/*.py 的注释已重写为正式中文**（去 AI 味：删了"大白话"、教程腔、感叹号、"你只需要改"等），代码逻辑未动。`make_figures.py`（在 results 目录）未重写注释，不在提交范围。

## 8. 关键文件路径速查

| 文件 | 说明 |
|---|---|
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\yamnet_bird_pipeline.py` | 主管道（注释已正式化） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\noise_robustness_eval.py` | 噪声评估（注释已正式化，含 import 兼容层） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\src\unified_evaluation.py` | 统一评估（注释已正式化，plot_confusion_matrix bug 已修） |
| `E:\stevensprogram\MLwork\results\yamnet\make_figures.py` | 本地出图脚本（读 npz 出图，不依赖 tf） |
| `E:\stevensprogram\MLwork\results\yamnet\*.npz / *.keras / *.json` | Kaggle 实跑产物（已下载） |
| `E:\stevensprogram\MLwork\results\yamnet\*.png / *.csv` | 本地生成的图与指标表 |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\README_zh.md` | 项目说明（第 29 行切分描述已改为 80/20） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\README.md` | 项目说明英文版（第 29 行同步改为 80/20） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\data\data\ml_*.csv` | 正式数据 |
| `C:\Users\13151\.claude\plans\deep-doodling-cocke.md` | 最初的改动计划（已批准并执行完） |
| `E:\stevensprogram\MLwork\YAMNet-kaggle\YAMNet\HANDOFF.md` | 本文档 |
