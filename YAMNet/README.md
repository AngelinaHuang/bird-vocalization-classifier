# Bird Vocalization Classification under Noisy Field Conditions

A comparative study of three modeling paradigms — gradient-boosted tabular features, image-based mel-spectrogram transfer learning, and raw-waveform transfer learning via YAMNet — for fine-grained bird species classification from audio, with an emphasis on **robustness to environmental noise**.

---

## 1. Overview

Bird vocalization classification in real field recordings is challenging for three reasons:

1. **Background noise** — wind, rain, and anthropogenic sound overlap the vocalization of interest.
2. **Long-tailed class distribution** — a few species dominate the data while many are rare.
3. **Fine-grained classes** — acoustically similar species are easily confused.

This project investigates which modeling paradigm degrades most gracefully as noise increases. The core contribution is a **controlled noise-injection experiment**: every model is evaluated on the same clean baseline and on artificially degraded versions at three signal-to-noise ratio (SNR) tiers (5 dB, 0 dB, −5 dB). Accuracy decay across SNR tiers is the primary comparison axis.

### Research questions

- Which paradigm achieves the highest clean-condition accuracy?
- Which paradigm is most robust to additive noise (slowest accuracy decay)?
- Which species pairs are most confusable, and does noise amplify those confusions?

---

## 2. Dataset

- **Source**: bird vocalization metadata from the Google Perch / BirdCLEF 2021–2026 competitions. After field-level deduplication, non-bird taxa (101 species: Amphibia/Insecta/Mammalia/Reptilia) and low-quality recordings (rating ≤ 0.5) are filtered out, yielding **144,250 clean bird recordings** across **1,126 species**. Audio files are mounted under Kaggle's BirdCLEF yearly dataset directories.
- **Preprocessing**: all clips are resampled to **16 kHz mono**, peak-normalized to **[−1, 1]** in `float32`, and fixed to a uniform length (center-trim or zero-pad).
- **Splits**: 90/10 stratified train/test split (SNR-tier aligned), with rare species (<5 recordings) kept entirely in the training pool. Training pool: **129,834 recordings** / 1,126 species; held-out test set: **14,416 recordings** / 1,093 species. A 5-fold SNR-stratified CV is nested within the training set (~103,853 train / ~25,967 val per fold). The split uses fixed seed 42 for deterministic reproducibility. See `data/augmenteddata/Data Augmentation Documentation.md` for details.
- **Low-resource augmentation**: 159 species with <15 training samples receive on-the-fly audio augmentation (time stretch / pitch shift / noise addition / volume change) to reach a target of 15 effective examples, capped at 50 new variants per species. See `src/e2e/audio_augmentation.py`.

---

## 3. Methodology

### 3.1 Three modeling paradigms

| Model | Input representation | Approach |
|---|---|---|
| **LightGBM** | Hand-crafted tabular acoustic features | Gradient-boosted trees on summarized features |
| **FastAI** | Mel-spectrogram (image) | Image-classification transfer learning |
| **YAMNet** | Raw waveform | Transfer learning via Google's YAMNet audio encoder + a small classification head |

All three are evaluated through a **single unified evaluation harness** so that comparisons are fair and not an artifact of differing metrics or splits.

### 3.2 YAMNet transfer learning (this repository)

This repository implements two strategies, run independently for fair comparison:

**Strategy 1: Frozen embeddings + classification head**
1. Each waveform is passed through the frozen YAMNet encoder, producing a 1024-dimensional embedding per 0.48 s frame.
2. Frame embeddings are averaged into a single clip-level 1024-d vector and cached to disk (embeddings are expensive to compute and reused across runs).
3. A small fully-connected head (`Dense(256, ReLU) → Dropout → Dense(num_classes, softmax)`) is trained on the cached embeddings with early stopping, checkpointing, and on-plateau learning-rate reduction.
4. Code: `src/yamnet_bird_pipeline.py`

**Strategy 2: End-to-end fine-tuning (implemented)**
1. Top YAMNet convolutional blocks are unfrozen; raw waveforms pass through end-to-end for joint training.
2. Differential learning rates: YAMNet variables at lr=1e-5, classification head at lr=1e-3.
3. MixUp augmentation (alpha=0.2) + class-balanced loss weights to mitigate the long tail.
4. On-the-fly audio augmentation during training: 159 low-resource species (<15 samples) receive dynamically generated variants per CV fold (max 50 per species).
5. Code: `src/e2e/yamnet_finetune_e2e.py` + `src/e2e/audio_augmentation.py`

### 3.3 Controlled noise-injection experiment

For every test clip and every SNR tier, additive Gaussian white noise is mixed with the clean waveform at the target SNR, the noisy waveform is re-encoded through YAMNet, and the prediction is recorded:

$$\text{SNR}_{\text{dB}} = 10 \log_{10}\frac{P_{\text{signal}}}{P_{\text{noise}}}$$

Lower SNR ⇒ stronger noise. Accuracy is computed per tier and plotted as a decay curve. Gaussian noise is used as a controlled, reproducible baseline; the noise module is isolated so that real environmental noise (wind/rain) can be substituted without touching the rest of the pipeline.

---

## 4. Project Structure

> This directory is the **YAMNet subfolder** of the [`bird-vocalization-classifier`](https://github.com/AngelinaHuang/bird-vocalization-classifier) repository. The other two modeling paradigms (LightGBM, FastAI) live in sibling folders at the same level.

```
YAMNet/
├── README.md                          # English (this file)
├── README_zh.md                       # 中文版
├── requirements.txt
├── .gitignore
├── HANDOFF.md                         # Troubleshooting & handoff guide
├── _inspect.py                        # quick inspection of cached outputs
├── src/
│   ├── yamnet_bird_pipeline.py        # Strategy 1: frozen embeddings + head (5-fold CV)
│   ├── noise_robustness_eval.py       # Strategy 1: SNR-tier noise injection + decay
│   ├── unified_evaluation.py          # Model-agnostic metrics + plotting
│   ├── measure_inference.py           # Strategy 1: inference speed + memory
│   ├── measure_inference_template.py  # Inference measurement template for teammates
│   └── e2e/                           # Strategy 2: end-to-end fine-tuning
│       ├── yamnet_finetune_e2e.py     # E2E pipeline (diff lr + MixUp + augmentation)
│       ├── audio_augmentation.py      # On-the-fly augmentation for low-resource species
│       ├── noise_eval_e2e.py          # E2E model noise evaluation
│       └── measure_inference_e2e.py   # E2E inference speed + memory
└── outputs/
    ├── yamnet/                        # Strategy 1 outputs (frozen embeddings)
    │   ├── label_map.json
    │   ├── embeddings.npz
    │   ├── cv_per_fold.csv / cv_summary.csv
    │   └── fold{1-5}/
    │       ├── yamnet_bird_model.keras
    │       ├── test_predictions.npz
    │       └── noise_results.npz
    ├── e2e/                           # Strategy 2 outputs (end-to-end)
    │   └── fold{1-5}/
    │       ├── yamnet_e2e_model.keras
    │       ├── test_predictions.npz
    │       └── noise_results.npz
    └── figures/
        ├── confusion_matrix_YAMNet.png
        └── noise_robustness.png
```

---

## 5. Installation

```bash
git clone git@github.com:AngelinaHuang/bird-vocalization-classifier.git
cd bird-vocalization-classifier/YAMNet
pip install -r requirements.txt
```

Dependencies: `tensorflow>=2.10`, `tensorflow-hub`, `librosa`, `soundfile`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

> YAMNet is downloaded automatically from TensorFlow Hub on first run (~17 MB). Strategy 1 (frozen embeddings + head) works on CPU. Strategy 2 (end-to-end fine-tuning) benefits from a GPU.

---

## 6. Usage

All scripts use paths relative to `src/`, so run them from the `src/` directory **inside YAMNet/**.

```bash
cd src
```

### 6.1 Strategy 1: Frozen embeddings + classification head

```bash
python yamnet_bird_pipeline.py
```

Reads CV split CSVs, locates audio under the mounted BirdCLEF yearly dataset directories, extracts and caches YAMNet embeddings, runs 5-fold CV training, and writes outputs to `outputs/yamnet/`.

### 6.2 Strategy 2: End-to-end fine-tuning

```bash
cd e2e
python yamnet_finetune_e2e.py
```

Unfreezes YAMNet top layers, applies differential learning rates + MixUp + on-the-fly audio augmentation, runs 5-fold end-to-end training. Outputs go to `outputs/e2e/` without overwriting Strategy 1 artifacts.

### 6.3 Test audio augmentation (single file)

```bash
python e2e/audio_augmentation.py <audio_file> [output_dir]
```

Applies 4 augmentation methods (time stretch / pitch shift / noise addition / volume change) to a single audio file and saves variants. For verification only; training-time augmentation is invoked automatically by `yamnet_finetune_e2e.py`.

### 6.4 Run the noise-robustness experiment

**Strategy 1:**
```bash
python noise_robustness_eval.py
```

**Strategy 2:**
```bash
cd e2e
python noise_eval_e2e.py
```

Injects Gaussian noise at 5 / 0 / −5 dB, re-encodes each noisy waveform, and records per-tier accuracy. Results are written to `outputs/yamnet/noise_results.npz` (or `outputs/e2e/`) and the decay curve to `outputs/figures/noise_robustness.png`.

### 6.5 Generate evaluation reports and figures

```bash
python unified_evaluation.py
```

Computes accuracy / precision / recall / F1 (macro and weighted), per-class breakdown, confusion matrix, multi-model accuracy comparison, and the noise-decay curve.

### 6.6 Kaggle Notebook workflow

Strategy 1 (3 cells):
```
cell1: %run -i src/yamnet_bird_pipeline.py       # 5-fold CV training
cell2: %run -i src/noise_robustness_eval.py       # noise evaluation
cell3: %run -i src/measure_inference.py           # inference speed + memory
```

Strategy 2 (3 cells, independent from Strategy 1):
```
cell1: %run -i src/e2e/yamnet_finetune_e2e.py      # end-to-end training (30-60 min/fold)
cell2: %run -i src/e2e/noise_eval_e2e.py            # noise evaluation
cell3: %run -i src/e2e/measure_inference_e2e.py     # inference speed + memory
```

---

## 7. Evaluation

The unified harness treats all three models identically. Each model ultimately provides `(y_true, y_pred, class_names)`; the harness then computes:

- **Classification**: accuracy, macro and weighted precision / recall / F1, per-class report.
- **Confusion matrix**: full heatmap; identifies the most confusable species pairs.
- **Multi-model comparison**: grouped bar chart of accuracy / macro-F1 / weighted-F1.
- **Noise-decay curve**: accuracy vs. SNR tier per model — the primary robustness comparison.
- **Cost**: per-sample inference latency and (where applicable) GPU memory footprint.

---

## 8. Current Results

### Strategy 1 (Frozen embeddings + head) — 5-fold CV

YAMNet has been trained on the full BirdCLEF data on Kaggle (1,229 species; 5-fold CV). The low absolute accuracy is dictated by the long-tailed distribution — 1,229 species × ~3 samples per class — not by a bug; the assignment focuses on the **relative decay trend** across the three models under noise, not on absolute scores.

**5-fold clean accuracy:**

| Metric | Mean ± Std |
|--------|-----------|
| Accuracy | 1.59% ± 0.27% |
| Macro-F1 | — |
| Weighted-F1 | — |

Per-fold clean accuracy: fold1=1.76%, fold2=1.09%, fold3=1.51%, fold4=1.76%, fold5=1.84%.

**Noise-robustness decay (5-fold mean ± std):**

| SNR tier | Accuracy |
|----------|----------|
| clean | 1.59% ± 0.27% |
| 5 dB | 0.47% ± 0.16% |
| 0 dB | 0.30% ± 0.10% (≈ random baseline) |
| −5 dB | 0.07% ± 0.06% |

Accuracy decays monotonically with noise strength, collapsing to the random level at 0 dB, indicating that robustness to strong noise is the primary bottleneck.

### Strategy 2 (End-to-end fine-tuning + V2.0 full data) — pending Kaggle run

The V2.0 dataset (129,834 training recordings / 1,126 species) is ready, and end-to-end fine-tuning code is implemented. Expected improvements:
- 129K samples substantially mitigate long-tail overfitting; clean accuracy should improve significantly
- MixUp + audio augmentation should yield flatter noise-decay curves
- Differential learning rate fine-tuning of YAMNet top layers adapts features to bird vocalizations

---

## 9. Limitations & Future Work

- **Long-tailed few-shot**: The original dataset had 1,229 species × ~3 samples/class, causing severe overfitting. **V2.0 has expanded to 129,834 recordings / 1,126 species**, with sampler weights + class weights + audio augmentation as triple control mechanisms. Pending Kaggle run to verify effectiveness.
- **Noise model**: Gaussian white noise is a controlled baseline; substituting real wind/rain noise is a drop-in change to the noise module.
- **End-to-end fine-tuning**: Implemented (`yamnet_finetune_e2e.py`), with differential learning rates + MixUp + on-the-fly augmentation. Pending Kaggle run.
- **Reproducibility**: Both sklearn splits and TF training use fixed seeds; augmentation module has seed control. Re-runs should yield approximately identical results.

---

## 10. Reproducibility

- Fixed random seed across data loading, splitting, training, and noise injection.
- Stratified splits are deterministic, so the noise-robustness experiment operates on the exact same test clips as the clean baseline.
- Cached embeddings decouple the expensive YAMNet forward pass from downstream iteration, ensuring repeated experiments are both fast and byte-for-byte reproducible.
