# YAMNet End-to-End Fine-tuning - Session Summary

**Date**: 2026-07-24
**File**: `yamnet_e2e_finetune.ipynb`
**Purpose**: Improve YAMNet classification from frozen baseline (13.27% Top-1) via end-to-end fine-tuning

---

## 1. Motivation

| Issue | Frozen Baseline | E2E Solution |
|-------|----------------|--------------|
| Professor requirement | Frozen encoder (not "learning like CNN") | End-to-end fine-tuning, model learns from data |
| Embedding discriminability | Inter-class/intra-class ratio = 0.53 | Top layers adapt to bird-specific features |
| Pooling | Fixed mean+max+std | Learnable Attention Pooling |
| Loss | Cross-entropy | Focal Loss (gamma=2) + Label Smoothing (0.1) |
| Noise robustness | 96.1% degradation at -5dB | Training-time noise augmentation (50% prob, 5-15 dB) |
| Head capacity | 2-layer Dense (787K params) | 3-layer + BN + Residual block |

## 2. Architecture

```
Input: 16kHz waveform (80000 samples, 5s)
  -> YAMNet MobileNetV1 (bottom frozen, top trainable in Stage 2)
  -> Frame embeddings (T, 1024)
  -> SpecAugment (training: freq mask 100, time mask 2)
  -> Attention Pooling (learnable, 128 units) -> (1024,)
  -> Dense(512) -> BN -> ReLU -> Dropout(0.3)
  -> Residual: proj(256) + [Dense(256)->BN->ReLU->Drop(0.2)->Dense(256)->BN->ReLU]
  -> Dense(1126, softmax)
```

## 3. Two-Stage Training

| Stage | Encoder | Head LR | Encoder LR | Epochs | Patience |
|-------|---------|---------|------------|--------|----------|
| 1 (warm-up) | Frozen | 1e-3 | - | 10 | 6 |
| 2 (fine-tune) | Top layers unfrozen | 5e-4 | 1e-5 | 20 | 8 |

- Differential learning rates via dual optimizers in custom `tf.GradientTape` training loop
- MixUp: 50% probability per batch, alpha=0.2, waveform-level
- Training-time noise: 50% probability, SNR uniform [5, 15] dB
- Focal Loss: gamma=2.0, label smoothing=0.1, with per-sample class weights

## 4. Evaluation Protocol

- **Identical 5-fold CV splits** as frozen baseline (for direct comparison)
- **Clean test**: Top-1, Top-5, Macro-F1, Balanced Accuracy, per-frequency-group
- **Noise test**: Gaussian white noise at 5dB, 0dB, -5dB SNR
- **Inference**: E2E latency (ms/sample)

## 5. Expected Improvements

| Metric | Frozen Baseline | E2E Expected |
|--------|----------------|--------------|
| Top-1 Accuracy | 13.27% | 25-45% |
| Top-5 Accuracy | 28.55% | 45-65% |
| Macro F1 | 0.0736 | 0.15-0.30 |
| -5dB degradation | 96.1% | 50-70% |
| Tail class acc | 4.56% | 10-20% |

## 6. Kaggle Execution Notes

- **GPU**: T4 or P100 (16GB), batch_size=32
- **Estimated time**: 3-5 hours per fold (Stage 1 + Stage 2), ~15-25 hours total for 5 folds
- **Can split**: Run 1-2 folds first to validate, then full 5-fold
- **Data**: Same Kaggle inputs as frozen baseline (processeddata-129834 + BirdCLEF audio)

## 7. File Inventory

| File | Description |
|------|-------------|
| `yamnet_e2e_finetune.ipynb` | Main notebook |
| Output: `cv_per_fold.csv` | Per-fold clean metrics |
| Output: `cv_summary.csv` | Summary mean +/- std |
| Output: `cv_noise_per_fold.csv` | Noise evaluation metrics |
| Output: `inference_metrics.csv` | Inference latency |
| Output: `fold{N}/e2e_model.keras` | Saved model per fold |
| Output: `label_map.json` | Label mapping (1126 classes) |

## 8. Relationship to Frozen Baseline

This notebook does NOT replace `yamnet_frozen_lightgbm_export.ipynb`. Both coexist:
- **Frozen baseline**: Provides embeddings for LightGBM teammate, establishes baseline numbers
- **E2E fine-tune**: Improved YAMNet-only results, satisfies professor's "CNN-like learning" requirement
- **Paper**: Report both as ablation (frozen vs fine-tuned), showing the contribution of end-to-end training

## 9. Key Design Decisions

1. **Why two-stage?** Direct fine-tuning from scratch is unstable; warm-up gives the head a good starting point
2. **Why differential LR?** Encoder is pre-trained (needs tiny updates), head is new (needs larger updates)
3. **Why Focal Loss?** 1126-class extreme long-tail; focal loss down-weights easy (head) classes
4. **Why Attention Pooling?** Fixed mean+max+std treats all frames equally; attention learns which frames matter
5. **Why training-time noise?** Paper title is "Noise-Robust"; model must see noise during training
6. **Why SpecAugment?** Regularization on frame embeddings, proven effective in speech/audio tasks

## 10. Context from This Session

- Confirmed data augmentation in frozen baseline DID work (log shows +1436~1453 samples/fold)
- The "augmentation not working" claim in training analysis was a misread of `embeddings_all.npz` (base cache by design excludes augmented samples)
- LightGBM is handled by teammate; this notebook is YAMNet-only
- Professor requires "CNN-like learning" = end-to-end training, not frozen feature extraction
