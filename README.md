# Bird Vocalization Classifier

Noise-robust bird vocalization classification for low-resource scenarios. Three parallel algorithms are developed and compared on the same dataset and evaluation pipeline.

## Project overview

| Item | Detail |
|------|--------|
| Task | Multi-class audio classification: identify bird species from field recordings under long-tail imbalance and environmental noise |
| Team | 3 members, each owns one algorithm |
| Platform | Kaggle notebooks (training + evaluation); local for data prep and plotting |
| Dataset | BirdCLEF 2021–2026 competition合集, Xeno-Canto + iNaturalist |

## Data pipeline (V2.0)

| Stage | Records | Species | Note |
|-------|---------|---------|------|
| Raw merged (6 yearly CSVs) | 183,239 | — | 2021–2026 |
| Field-level deduplicated | 167,308 | 1,229 | Zero-loss merge by field type |
| Non-bird filtered + rating > 0.5 | 144,250 | 1,126 | Removed 101 non-avian taxa |
| Training pool | **129,834** | 1,126 | Full retention, no per-class cap |
| Fixed holdout test | **14,416** | 1,093 | SNR-stratified 10% |
| CV folds | ~103,853 train / ~25,967 val ×5 | 1,126 | SNR-stratified 5-fold |

Long-tail controls during training: balanced sampler (`1/class_count`) + mild class-weighted loss (inverse-sqrt, clipped [0.25, 4.00]) + targeted audio augmentation (159 species < 15 samples → target 15).

## Three algorithms

| Algorithm | Input | Model | Owner |
|-----------|-------|-------|-------|
| **YAMNet** | Raw waveform (16 kHz × 5 s) | YAMNet end-to-end fine-tune + classification head | Wenjuan Huang |
| **LightGBM** | 32-dim handcrafted features (MFCC + spectral stats) | Gradient boosting tree | Jianan Zhang |
| **FastAI** | Mel spectrogram PNG | ResNet34 via FastAI | Jincheng Chen |

All three use the same 5-fold CV splits, the same noise evaluation protocol (Gaussian white noise at 5/0/−5 dB SNR), and the same audio augmentation module, so Top-1 and noise robustness are directly comparable.

## Audio augmentation

Shared module `audio_augmentation.py` + per-algorithm glue code:

- **4 methods**: time stretch, pitch shift, noise addition, volume change (16 parameter combinations)
- **Target**: species with < 15 training samples → augment to ≥ 15
- **Caps**: 16 variants/recording, 50 new variants/species
- **Online only**: generated per-fold during training, no offline files
- **Seed**: `SEED + fold` for reproducibility

| File | Used by |
|------|---------|
| `audio_augmentation.py` | All three (shared bottom layer) |
| `lightgbm/augmentation_glue.py` | LightGBM |
| `fastaicode/augmentation_glue.py` | FastAI |
| YAMNet integrates inline in `yamnet_finetune_e2e.py` | YAMNet |

Teammates: see `full-kaggle-data/AUGMENTATION_INTEGRATION_GUIDE.md` for the one-page setup guide.

## Repository structure

```
bird-vocalization-classifier/
├── doc/                        # Project docs (proposal, literature review, data processing)
├── YAMNet/
│   ├── src/                     # YAMNet pipeline + augmentation + eval + inference
│   ├── HANDOFF.md               # YAMNet dev handoff doc
│   ├── README.md / README_zh.md
│   └── requirements.txt
├── lightgbm/
│   ├── notebook6312d60402-0715.ipynb   # LightGBM training notebook
│   └── augmentation_glue.py
├── fastaicode/
│   ├── fastaikaggle.txt                # FastAI training script
│   ├── fastaikaggle评估.txt             # FastAI noise eval script
│   └── augmentation_glue.py
├── full-kaggle-data/
│   ├── prepare_full_training_data.py   # Data pipeline (V2.0)
│   ├── metadata/ml_full_deduped.csv     # 167,308 deduped source
│   ├── processed data-129834/           # Final CV splits + weights
│   ├── Data Augmentation Documentation.md
│   └── AUGMENTATION_INTEGRATION_GUIDE.md
├── result-5000+sample/           # Previous-era results (5,976-sample subset)
└── README.md                    # This file
```

## Kaggle run order (per algorithm)

```
Cell 1: Training (5-fold CV, augmentation auto-triggers each fold)
Cell 2: Noise robustness evaluation (clean / 5 dB / 0 dB / −5 dB)
Cell 3: Inference speed + GPU memory measurement
```

## Key documents

| Document | Location | Purpose |
|----------|----------|---------|
| Data processing doc | `full-kaggle-data/Data Augmentation Documentation.md` | V2.0 full-data pipeline + augmentation spec |
| Augmentation integration guide | `full-kaggle-data/AUGMENTATION_INTEGRATION_GUIDE.md` | One-page teammate setup guide |
| Project context | `doc/CONTEXT_SUMMARY.md` | Full project overview (note: predates V2.0) |
| YAMNet handoff | `YAMNet/HANDOFF.md` | YAMNet dev/debug history |

## Reproduce data pipeline

```bash
cd full-kaggle-data
python prepare_full_training_data.py
```

Reads `metadata/ml_full_deduped.csv` (seed 42), outputs to `processed data-129834/`.

---
*Last updated: 2026-07-18*
