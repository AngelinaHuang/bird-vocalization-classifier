# Bird Vocalization: Full-Data Preparation, Balancing, and Augmentation

**Version 2.0 | Updated 2026-07-18**

## 1. Summary

The formal dataset uses every clean bird recording that remains after filtering. Instead of discarding frequent-species recordings to control the long tail, imbalance is controlled during training with:

1. **Balanced sampling**: every species has equal total sampling probability.
2. **Mild class-weighted loss**: tail-species errors have more influence, without allowing one-example species to destabilise training.
3. **Targeted audio augmentation**: only low-resource species are augmented, and only in the training portion of a CV fold.

This retains useful diversity from frequent species (different recordists, locations, seasons, devices, and background conditions) without allowing them to dominate batches or optimisation.

## 2. Source, cleaning, and final split

Input: `metadata/ml_full_deduped.csv`, containing 167,308 field-level deduplicated records.

| Cleaning rule | Rationale |
| --- | --- |
| Remove numeric-only `primary_label` values | They identify non-avian taxa; 101 non-bird species are excluded. |
| Remove `rating <= 0.5` | Unrated and extremely low-quality audio is excluded. |
| Build SNR tier from `rating` | High: >=4.0; Medium: 2.0-3.9; Heavy: <2.0. |

The cleaned population contains **144,250 recordings across 1,126 bird species**. It is split once with random seed 42:

| Split | Rows | Species | Use |
| --- | ---: | ---: | --- |
| Cleaned bird population | 144,250 | 1,126 | Source population |
| Training pool | **129,834** | **1,126** | Model development and CV |
| Fixed holdout test | **14,416** | 1,093 | Final clean and noisy evaluation |

Species with fewer than five recordings remain entirely in training so their only examples are not withheld. Other species contribute approximately 10% of their eligible records to test. The fixed holdout is also aligned to the 10% SNR allocation.

| SNR tier | Training rows | Test rows |
| --- | ---: | ---: |
| High | 86,731 | 9,631 |
| Medium | 41,011 | 4,553 |
| Heavy | 2,092 | 232 |

## 3. Why use the full training pool?

The earlier capped version had 99,960 training rows. Per-species caps reduce imbalance, but also discard roughly 30K valid recordings. Those recordings are valuable for within-species robustness.

The formal policy is:

```text
Keep all clean bird data -> reserve fixed test data -> train on all remaining data
                                              -> balance sampling and loss during training
```

This does **not** mean head classes are allowed to dominate. The sampler controls how often each class enters batches, and the loss weights control how strongly each class affects optimisation.

## 4. Long-tail controls

### 4.1 Balanced sampling

Every row in `02_train_full_weighted.csv` has:

```text
sampler_weight = 1 / number of training recordings for its species
```

Use this column with a weighted sampler, such as PyTorch `WeightedRandomSampler`. The weights of all recordings from one species sum to 1, so each species has the same total sampling mass. This is a sampling control only; it is not a loss coefficient.

### 4.2 Mild class-weighted loss

The CSV also includes one class-level `loss_class_weight` per row:

```text
raw_weight(c) = 1 / sqrt(train_count(c))
loss_class_weight(c) = clip(raw_weight(c) / mean(raw_weight), 0.25, 4.00)
```

Inverse-square-root weighting is intentionally milder than `1 / count`. It helps rare classes contribute useful gradients while clipping to **[0.25, 4.00]** protects training from excessive weights. Map this weight to the model class index and use it in the classification loss. Do not multiply it into `sampler_weight`.

| Mechanism | What it solves | Current implementation |
| --- | --- | --- |
| Full training pool | Retains head-class acoustic diversity | 129,834 original recordings |
| Balanced sampler | Prevents head classes filling batches | `1 / class_count` per recording |
| Mild weighted loss | Gives tail errors more influence | inverse sqrt, clipped 0.25-4.00 |
| Audio augmentation | Adds variation to the smallest classes | only species with <15 training recordings |

Report macro-F1, balanced accuracy, per-SNR performance, and class-frequency-group metrics alongside overall Top-1 accuracy.

## 5. Targeted audio augmentation

Only species with fewer than **15 training examples** are augmented. The target is at least 15 effective examples per such species. Validation and test records remain original audio only.

| Transformation | Parameter values | Intent |
| --- | --- | --- |
| Time stretch | 0.85x, 0.90x, 0.95x, 1.05x, 1.10x | Natural tempo variation |
| Pitch shift | -2, -1, +1, +2 semitones | Individual/geographic pitch variation |
| Noise addition | 5, 10, 15 dB SNR | Recording-condition variation |
| Volume gain | 0.70x, 0.85x, 1.15x, 1.30x | Distance/gain variation |

An original recording can provide up to 16 candidate variants (the sum of each method's parameter options: 5 time-stretch + 4 pitch + 3 noise + 4 volume). New variants are generated only until the species reaches the target, with a **maximum of 50 new variants per species**. A single original recording is capped at **16 variants** to prevent over-representation of any one recording.

**Distribution strategy** (implemented in `expand_train_df`):

| Condition | Strategy |
| --- | --- |
| `needed >= current_count` | Uniform distribution: each original sample gets `⌊needed / N⌋` variants; the first `needed % N` samples get one extra |
| `needed < current_count` | Random selection: exactly `needed` original samples are randomly chosen (seed-controlled, reproducible), each gets 1 variant |

**Key tuning parameters** in `audio_augmentation.py`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `target_per_species` | 15 | Minimum effective examples after augmentation |
| `max_aug_per_species` | 50 | Hard cap on new variants per species (prevents runaway) |
| `seed` | 42 | Controls random selection of samples when `needed < current_count`; per-fold seeds use `cfg.SEED + fold` for reproducibility |

For YAMNet end-to-end training, the recommended approach is online augmentation using `audio_augmentation.py`:

```text
Create one fold's waveform cache
-> identify fold-training species with <15 examples
-> generate training-only waveform variants (via expand_with_augmentation)
-> add variants with their original labels to X_train, y_train_raw
-> train with balanced sampling and mild class-weighted loss
```

Because augmentation happens after the fold split, augmented siblings cannot enter the validation set.

## 6. Formal output files

The formal outputs are in `processed data-129834/`.

| File | Contents |
| --- | --- |
| `01_bird_only_full.csv` | 144,250 cleaned bird records |
| `02_train_full_weighted.csv` | 129,834 training rows with `class_count_train`, `sampler_weight`, `loss_class_weight`, and `cv_fold` |
| `03_test_holdout.csv` | 14,416 fixed holdout rows |
| `class_weights.csv` | One row per species with its count and both weights |
| `cv_fold{1-5}_train.csv` / `cv_fold{1-5}_val.csv` | Five CV splits |
| `summary.csv` | Split row/species totals |

Validation folds contain 25,949-25,981 records. Checks confirmed no `filename` overlap between train/test, nor between training and validation within a fold.

## 7. Reproduction and evaluation

Regenerate the split with:

```bash
cd E:\MLwork\MLwork\data\augmenteddata
python prepare_full_training_data.py
```

The script reads `metadata/ml_full_deduped.csv` and uses seed 42. For strict cross-validation, recompute class weights from each fold's training CSV, so validation data is never used for weight estimation.

Select settings with CV only. Keep `03_test_holdout.csv` untouched until final evaluation, then evaluate clean test audio and the same test recordings with fixed-seed 5 dB, 0 dB, and -5 dB noise. This makes LightGBM, FastAI CNN, and YAMNet directly comparable.
