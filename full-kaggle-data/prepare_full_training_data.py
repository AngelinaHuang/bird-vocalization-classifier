#!/usr/bin/env python3
"""Prepare the full BirdCLEF training pool without per-species caps.

The script keeps a fixed, per-species holdout set and writes training metadata
with two distinct imbalance controls:

* ``sampler_weight``: 1 / class_count.  Pass this to a weighted sampler to
  sample species approximately uniformly.
* ``loss_class_weight``: clipped inverse-square-root frequency.  This is a
  deliberately mild loss weight, so one-recording classes cannot dominate the
  optimisation signal.

No audio is created or modified by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
TEST_RATIO = 0.10
MIN_SAMPLES_FOR_TEST = 5
N_FOLDS = 5
LOSS_WEIGHT_MIN = 0.25
LOSS_WEIGHT_MAX = 4.0


def assign_snr_tier(rating: float) -> str:
    if rating >= 4.0:
        return "High"
    if rating >= 2.0:
        return "Medium"
    return "Heavy"


def filter_birds(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["primary_label"].astype(str)
    ratings = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    birds = df.loc[~labels.str.fullmatch(r"\d+") & (ratings > 0.5)].copy()
    birds["rating"] = ratings.loc[birds.index]
    birds["snr_tier"] = birds["rating"].map(assign_snr_tier)
    return birds.reset_index(drop=True)


def split_holdout(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split within each species, preferentially sampling across SNR tiers.

    Classes with fewer than five recordings are training-only: holding out an
    example from them would make the already scarce class harder to learn.
    """
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, group in df.groupby("primary_label", sort=True):
        n = len(group)
        if n < MIN_SAMPLES_FOR_TEST:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(n * TEST_RATIO)))
        n_test = min(n_test, n - 1)
        # Allocate the species' holdout quota proportionally across its SNR
        # tiers.  Round-robin sampling would severely over-represent minority
        # SNR tiers when a species contributes only one or two test examples.
        tier_rows = {}
        for tier, tier_df in group.groupby("snr_tier", sort=False):
            indices = tier_df.index.to_numpy().copy()
            rng.shuffle(indices)
            tier_rows[tier] = list(indices)
        raw_quota = {tier: n_test * len(indices) / n for tier, indices in tier_rows.items()}
        quota = {tier: int(value) for tier, value in raw_quota.items()}
        remaining = n_test - sum(quota.values())
        for tier in sorted(raw_quota, key=lambda t: raw_quota[t] - quota[t], reverse=True):
            if remaining == 0:
                break
            if quota[tier] < len(tier_rows[tier]):
                quota[tier] += 1
                remaining -= 1
        selected = []
        for tier, take in quota.items():
            selected.extend(tier_rows[tier][:take])
        test_parts.append(df.loc[selected])
        train_parts.append(group.drop(index=selected))

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    # Match the established global 10% SNR holdout size exactly.  The
    # per-species "at least one" rule can otherwise overshoot it slightly.
    eligible_for_holdout = df["primary_label"].map(df["primary_label"].value_counts()) >= MIN_SAMPLES_FOR_TEST
    target_by_tier = {
        tier: int(len(df.loc[eligible_for_holdout & (df["snr_tier"] == tier)]) * TEST_RATIO)
        for tier in ("High", "Medium", "Heavy")
    }
    returned_to_train = []
    retained_test = []
    for tier, tier_test in test.groupby("snr_tier", sort=False):
        target = target_by_tier[tier]
        excess = max(0, len(tier_test) - target)
        if excess:
            returned_indices = rng.choice(tier_test.index.to_numpy(), size=excess, replace=False)
            returned_to_train.append(tier_test.loc[returned_indices])
            tier_test = tier_test.drop(index=returned_indices)
        retained_test.append(tier_test)
    if returned_to_train:
        train = pd.concat([train, *returned_to_train], ignore_index=True)
    test = pd.concat(retained_test, ignore_index=True)
    # A tier can be under target when it occurs only as a small minority within
    # many species. Fill that small deficit from eligible training rows while
    # retaining at least one training example for every species.
    moved_to_test = []
    for tier, target in target_by_tier.items():
        deficit = target - int((test["snr_tier"] == tier).sum())
        if deficit <= 0:
            continue
        train_counts = train["primary_label"].value_counts()
        candidates = train.loc[
            (train["snr_tier"] == tier)
            & (train["primary_label"].map(train_counts) > 1)
        ]
        if len(candidates) < deficit:
            raise RuntimeError(f"Not enough eligible {tier} samples to fill test holdout")
        chosen = rng.choice(candidates.index.to_numpy(), size=deficit, replace=False)
        moved_to_test.append(train.loc[chosen])
        train = train.drop(index=chosen)
    if moved_to_test:
        test = pd.concat([test, *moved_to_test], ignore_index=True)
    train = train.reset_index(drop=True)
    return train, test


def add_imbalance_weights(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = train["primary_label"].value_counts().sort_index()
    # WeightedRandomSampler: total probability mass is identical per class.
    sampler = 1.0 / train["primary_label"].map(counts).astype(float)
    # Gentle loss correction: inverse sqrt frequency, normalised and clipped.
    class_weights = 1.0 / np.sqrt(counts.astype(float))
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.clip(LOSS_WEIGHT_MIN, LOSS_WEIGHT_MAX)
    train = train.copy()
    train["class_count_train"] = train["primary_label"].map(counts).astype(int)
    train["sampler_weight"] = sampler.astype(float)
    train["loss_class_weight"] = train["primary_label"].map(class_weights).astype(float)
    table = pd.DataFrame({
        "primary_label": counts.index,
        "train_count": counts.values,
        "sampler_weight": 1.0 / counts.values,
        "loss_class_weight": class_weights.values,
    })
    return train, table


def add_cv_folds(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign each original training record to one deterministic validation fold.

    Assignment is performed independently within each species and interleaves
    its SNR tiers, so common classes are represented in all folds while tiny
    classes remain train-only in folds where they have no validation example.
    """
    rng = np.random.default_rng(seed)
    fold = pd.Series(index=train.index, dtype="int64")
    for _, group in train.groupby("primary_label", sort=True):
        tier_indices = []
        for _, tier_df in group.groupby("snr_tier", sort=False):
            ids = tier_df.index.to_numpy().copy()
            rng.shuffle(ids)
            tier_indices.append(list(ids))
        interleaved = []
        while any(tier_indices):
            for ids in tier_indices:
                if ids:
                    interleaved.append(ids.pop())
        # Offset each species independently so classes with fewer than five
        # rows do not all accumulate in the first validation fold.
        offset = int(rng.integers(N_FOLDS))
        for position, index in enumerate(interleaved):
            fold.loc[index] = (position + offset) % N_FOLDS + 1
    result = train.copy()
    result["cv_fold"] = fold.astype(int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("metadata/ml_full_deduped.csv"))
    parser.add_argument("--output", type=Path, default=Path("processed data-129834"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input, low_memory=False)
    birds = filter_birds(raw)
    train, test = split_holdout(birds, RANDOM_SEED)
    train, weights = add_imbalance_weights(train)
    train = add_cv_folds(train, RANDOM_SEED)

    birds.to_csv(args.output / "01_bird_only_full.csv", index=False)
    train.to_csv(args.output / "02_train_full_weighted.csv", index=False)
    test.to_csv(args.output / "03_test_holdout.csv", index=False)
    weights.to_csv(args.output / "class_weights.csv", index=False)
    for fold in range(1, N_FOLDS + 1):
        train.loc[train["cv_fold"] != fold].to_csv(
            args.output / f"cv_fold{fold}_train.csv", index=False
        )
        train.loc[train["cv_fold"] == fold].to_csv(
            args.output / f"cv_fold{fold}_val.csv", index=False
        )

    summary = pd.DataFrame([
        {"split": "bird_full", "rows": len(birds), "species": birds["primary_label"].nunique()},
        {"split": "train_full", "rows": len(train), "species": train["primary_label"].nunique()},
        {"split": "test_holdout", "rows": len(test), "species": test["primary_label"].nunique()},
    ])
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
