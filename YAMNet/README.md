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

- **Source**: bird vocalization audio organized as `data/raw/<species>/<clip>.{wav,mp3,flac}`. Each subfolder name defines one class label.
- **Preprocessing**: all clips are resampled to **16 kHz mono**, peak-normalized to **[−1, 1]** in `float32`, and fixed to a uniform length (center-trim or zero-pad).
- **Splits**: stratified train / validation / test split (70 / 15 / 15 %) with class proportions preserved. The split is deterministic (fixed seed) so that the noise-robustness evaluation reuses the exact same test set.

> The pipeline reads any folder-structured audio collection, so it accepts both the full project dataset and small development subsets interchangeably.

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

The YAMNet pipeline uses the lightweight "precompute-embeddings + train-a-head" strategy:

1. Each waveform is passed through the frozen YAMNet encoder, producing a 1024-dimensional embedding per 0.48 s frame.
2. Frame embeddings are averaged into a single clip-level 1024-d vector and cached to disk (embeddings are expensive to compute and reused across runs).
3. A small fully-connected head (`Dense(256, ReLU) → Dropout → Dense(num_classes, softmax)`) is trained on the cached embeddings with early stopping, checkpointing, and on-plateau learning-rate reduction.

An end-to-end fine-tuning variant (unfreezing the top YAMNet convolutional blocks with differential learning rates) is outlined in `src/yamnet_bird_pipeline.py` as a follow-up.

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
├── _inspect.py                        # quick inspection of cached outputs
├── data/
│   ├── raw/                           # <species>/<clip>.{wav,mp3,flac}  (gitignored)
│   └── processed/                     # (reserved for future preprocessing)
├── src/
│   ├── yamnet_bird_pipeline.py        # YAMNet embedding extraction + head training
│   ├── noise_robustness_eval.py       # SNR-tier noise injection + decay measurement
│   └── unified_evaluation.py          # Model-agnostic metrics + plotting
└── outputs/
    ├── yamnet/
    │   ├── label_map.json             # species <-> integer index mapping
    │   ├── embeddings.npz             # cached YAMNet embeddings (gitignored)
    │   ├── yamnet_bird_model.keras    # trained classification head (gitignored)
    │   ├── test_predictions.npz       # held-out test predictions (gitignored)
    │   └── noise_results.npz          # per-SNR accuracy + predictions (gitignored)
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

Dependencies: `tensorflow>=2.10`, `tensorflow-hub`, `librosa`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

> YAMNet is downloaded automatically from TensorFlow Hub on first run (~17 MB). No GPU is required for the embedding+head workflow; a CPU is sufficient.

---

## 6. Usage

All scripts use paths relative to `src/`, so run them from the `src/` directory **inside YAMNet/**.

```bash
cd src
```

### 6.1 Train the YAMNet classifier

```bash
python yamnet_bird_pipeline.py
```

Scans `data/raw/`, extracts and caches YAMNet embeddings, trains the classification head, and writes the model, label map, and test predictions to `outputs/yamnet/`.

### 6.2 Run the noise-robustness experiment

```bash
python noise_robustness_eval.py
```

Reproduces the training-time test split, injects Gaussian noise at 5 / 0 / −5 dB, re-encodes each noisy waveform, and records per-tier accuracy. Results are written to `outputs/yamnet/noise_results.npz` and the decay curve to `outputs/figures/noise_robustness.png`.

### 6.3 Generate evaluation reports and figures

```bash
python unified_evaluation.py
```

Computes accuracy / precision / recall / F1 (macro and weighted), per-class breakdown, confusion matrix, multi-model accuracy comparison, and the noise-decay curve.

---

## 7. Evaluation

The unified harness treats all three models identically. Each model ultimately provides `(y_true, y_pred, class_names)`; the harness then computes:

- **Classification**: accuracy, macro and weighted precision / recall / F1, per-class report.
- **Confusion matrix**: full heatmap; identifies the most confusable species pairs.
- **Multi-model comparison**: grouped bar chart of accuracy / macro-F1 / weighted-F1.
- **Noise-decay curve**: accuracy vs. SNR tier per model — the primary robustness comparison.
- **Cost**: per-sample inference latency and (where applicable) GPU memory footprint.

---

## 8. Current Results (preliminary)

Preliminary results on a small, class-balanced evaluation subset (4 species, 63 clips, 10-clip held-out test set). Full-dataset figures will replace these once the complete dataset is processed; the qualitative trends are expected to hold.

**Clean-condition performance (YAMNet):**

| Metric | Value |
|---|---|
| Accuracy | 0.90 |
| Macro-F1 | 0.90 |
| Weighted-F1 | 0.90 |

The single clean-condition error is a `northern_cardinal` clip misclassified as `house_sparrow`, indicating these two acoustically similar species form the primary confusable pair.

**Noise-robustness decay (YAMNet):**

| SNR tier | Accuracy |
|---|---|
| clean | 0.90 |
| 5 dB | 0.50 |
| 0 dB | 0.30 |
| −5 dB | 0.20 |

Accuracy decays monotonically with noise strength. At the heaviest noise tier (−5 dB) the classifier collapses to a single output class for all inputs, indicating that robustness to strong noise is the limiting factor and motivating future denoising front-ends.

---

## 9. Limitations & Future Work

- **Sample size**: preliminary numbers are computed on a small held-out set; statistical confidence will improve with the full dataset.
- **Noise model**: Gaussian white noise is a controlled baseline; substituting real wind/rain noise is a drop-in change to the noise module.
- **Fine-tuning**: only the YAMNet classification head is trained; end-to-end fine-tuning of the top convolutional blocks is the natural next step for higher clean-condition accuracy.
- **Class imbalance**: the current subset is balanced; long-tail mitigation (class weighting, focal loss) will be evaluated on the full long-tailed distribution.

---

## 10. Reproducibility

- Fixed random seed across data loading, splitting, training, and noise injection.
- Stratified splits are deterministic, so the noise-robustness experiment operates on the exact same test clips as the clean baseline.
- Cached embeddings decouple the expensive YAMNet forward pass from downstream iteration, ensuring repeated experiments are both fast and byte-for-byte reproducible.
