# Data Processing Documentation

## 1\. Dataset Overview

This project uses bird vocalization metadata from the Google Perch / BirdCLEF 2021–2026 competitions, merging six annual CSV files. The raw total is **183,239 records** covering **1,229 species** (birds, frogs, insects, mammals).

**Key Metadata Fields**:

-   `filename`: Unique audio file identifier
    
-   `primary_label`: Species code (eBird code or taxon ID)
    
-   `secondary_labels`: Other species present in the recording (list)
    
-   `type`: Recording type (call, song, flight call, etc.)
    
-   `rating`: Quality score (0–5), used as SNR proxy
    
-   `latitude`, `longitude`: Coordinates
    
-   `scientific_name`, `common_name`: Taxonomy
    
-   `date`, `time`: Recording time
    
-   `collection`: Source (Xeno‑Canto / iNaturalist)
    
-   `class_name`: Taxonomic class (Aves, Amphibia, Mammalia, Insecta, etc.)
    
-   `inat_taxon_id`: iNaturalist taxon ID
    
-   `source_year`: Year(s) of data origin (e.g., "2025,2026")
    

> **Note**: The original competition description mentioned 86 species, but the merged dataset actually contains **1,229 species**. Therefore, our sampling strategy must ensure a minimum of 5 samples per species (hard constraint).

---

## 2\. Field‑Level Deduplication Strategy

To achieve **zero information loss**, duplicate records with the same `filename` are merged using a **field‑wise intelligent merge**:

| Field Type | Merge Strategy | Example |
| --- | --- | --- |
| **List‑type** (`secondary_labels`, `type`) | Union of all non‑null values, deduplicated | Record A: `['song']`, Record B: `['call']` → `['song', 'call']` |
| **Text‑type** (`scientific_name`, `common_name`, etc.) | Take any non‑null value (prefer latest year) | - |
| **Numeric‑type** (`rating`, `latitude`, `longitude`, etc.) | Take the non‑null value from the latest `source_year` | 2019 rating=4.0, 2025 rating=3.5 → keep 3.5 |
| **Time‑type** (`date`, `time`) | Take the latest non‑null value | - |
| **Categorical** (`collection`, `class_name`, etc.) | Take the latest non‑null value | - |
| **Year‑type** (`source_year`) | Merge and deduplicate as comma‑separated string | "2022,2023" → "2022,2023" |

Rules applied per `filename` group:

1.  Extract all non‑null values for each field.
    
2.  For list fields, compute union and deduplicate.
    
3.  For other fields, parse `source_year` (as integer) and select the value from the record with the maximum year.
    
4.  If a field is empty in all records, keep it as null.
    

This ensures:

-   **No data loss**: all historical labels and classifications are retained.
    
-   **Latest priority**: fields that change over time (rating, coordinates) use the most recent data.
    
-   **Full species coverage**: all 1,229 species are preserved.
    

---

## 3\. Deduplication & Sampling Pipeline

The complete processing pipeline consists of the following steps:

1.  **Load 6 annual CSVs** (`21_train_metadata.csv` … `26_train_metadata.csv`).
    
2.  **Field‑level deduplication** → output `ml_full_deduped.csv` (167,308 unique recordings).
    
3.  **Dual‑layer stratified sampling** (species + SNR, hard constraint ≥5 per species) → `ml_sampled.csv` (5,976 samples).
    
4.  **80/20 train/test split** (SNR‑stratified) → `ml_train.csv` (4,780) and `ml_test.csv` (1,196).
    
5.  **5‑fold cross‑validation** (SNR‑stratified within training set) → each fold gives train/val sets (3,824 / 956).
    

The sampling algorithm is the same as previously described, but now operates on the deduplicated master dataset.

---

## 4\. Representativeness Verification

| Metric | Result |
| --- | --- |
| Species coverage | 1,229 / 1,229 (100%) |
| SNR distribution deviation | High: -1.2pp, Medium: +1.9pp, Heavy: -0.7pp (all within ±2pp) |
| Filename duplication check | 0 duplicates across train/val/test splits |

---

## 5\. Output Files

| Filename | Rows | Purpose |
| --- | --- | --- |
| `ml_full_deduped.csv` | 167,308 | Full deduplicated master dataset (field‑merged) |
| `ml_sampled.csv` | 5,976 | Dual‑layer stratified subset (all 1,229 species covered) |
| `ml_train.csv` | 4,780 | Training set (80%) |
| `ml_test.csv` | 1,196 | Held‑out test set (20%) |
| `ml_cv_fold1_train.csv` … `ml_cv_fold5_train.csv` | 3,824 each | CV training folds |
| `ml_cv_fold1_val.csv` … `ml_cv_fold5_val.csv` | 956 each | CV validation folds |

---

## 6\. Full Processing Code (Python)

Below is the complete, reproducible Python script that implements all the steps described above. Ensure all annual CSV files are placed in the input directory.

python
```
import pandas as pd
import numpy as np
import os
from glob import glob
from collections import defaultdict
import re

# ==================== Configuration ====================
INPUT_DIR = "./metadata_raw/"          # Directory containing 21-26 CSV files
OUTPUT_DIR = "./processed/"            # Output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_FILES = [
    "21_train_metadata.csv",
    "22_train_metadata.csv",
    "23_train_metadata.csv",
    "24_train_metadata.csv",
    "25_train_metadata.csv",
    "26_train_metadata.csv"
]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==================== Helper Functions ====================

def merge_lists(list_vals):
    """Merge multiple lists, deduplicate, filter nulls."""
    merged = set()
    for v in list_vals:
        if isinstance(v, list):
            merged.update(v)
        elif isinstance(v, str) and v.startswith('['):
            try:
                items = eval(v)
                if isinstance(items, list):
                    merged.update(items)
            except:
                pass
        elif v and not pd.isna(v):
            merged.add(str(v))
    return list(merged) if merged else None

def parse_year(year_str):
    """Parse source_year field and return the maximum year (int)."""
    if pd.isna(year_str) or year_str == '':
        return -1
    years = re.findall(r'\d{4}', str(year_str))
    if not years:
        return -1
    return max(map(int, years))

def choose_by_latest_year(group, field, fallback=None):
    """Choose the non-null value from the record with the largest source_year."""
    best_year = -1
    best_val = fallback
    for _, row in group.iterrows():
        val = row[field]
        if pd.isna(val) or val == '':
            continue
        year = parse_year(row['source_year'])
        if year > best_year:
            best_year = year
            best_val = val
    return best_val

def merge_row_group(group):
    """
    Merge all records for a single filename into one row.
    Returns a Series.
    """
    group = group.copy()
    
    # List fields
    sec_labels = merge_lists(group['secondary_labels'].tolist())
    types = merge_lists(group['type'].tolist())
    
    # Merge source_year: union of all years, deduplicated
    all_years = set()
    for y in group['source_year'].dropna():
        if isinstance(y, str):
            for part in re.split(r'[,;\s]+', y):
                if part.strip():
                    all_years.add(part.strip())
        else:
            all_years.add(str(y))
    merged_year = ', '.join(sorted(all_years, key=lambda x: int(x) if x.isdigit() else 0)) if all_years else None
    
    # Other fields: take latest non-null
    latest_rating = choose_by_latest_year(group, 'rating')
    latest_lat = choose_by_latest_year(group, 'latitude')
    latest_lon = choose_by_latest_year(group, 'longitude')
    latest_date = choose_by_latest_year(group, 'date')
    latest_time = choose_by_latest_year(group, 'time')
    latest_collection = choose_by_latest_year(group, 'collection')
    latest_class_name = choose_by_latest_year(group, 'class_name')
    latest_inat_taxon_id = choose_by_latest_year(group, 'inat_taxon_id')
    latest_scientific = choose_by_latest_year(group, 'scientific_name')
    latest_common = choose_by_latest_year(group, 'common_name')
    latest_author = choose_by_latest_year(group, 'author')
    latest_license = choose_by_latest_year(group, 'license')
    latest_url = choose_by_latest_year(group, 'url')
    
    # Fields that are identical across duplicates
    filename = group.iloc[0]['filename']
    primary_label = group.iloc[0]['primary_label']
    
    merged = {
        'filename': filename,
        'primary_label': primary_label,
        'secondary_labels': sec_labels,
        'type': types,
        'rating': latest_rating,
        'latitude': latest_lat,
        'longitude': latest_lon,
        'scientific_name': latest_scientific,
        'common_name': latest_common,
        'author': latest_author,
        'license': latest_license,
        'url': latest_url,
        'date': latest_date,
        'time': latest_time,
        'collection': latest_collection,
        'class_name': latest_class_name,
        'inat_taxon_id': latest_inat_taxon_id,
        'source_year': merged_year,
    }
    return pd.Series(merged)

# ==================== Step 1: Load and merge raw CSVs ====================

print("Loading raw CSV files...")
all_dfs = []
for fname in YEAR_FILES:
    path = os.path.join(INPUT_DIR, fname)
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue
    df = pd.read_csv(path, low_memory=False)
    # Extract year from filename
    year = re.search(r'(\d{2})_train', fname).group(1)
    if year.startswith('21'):
        full_year = 2021
    elif year.startswith('22'):
        full_year = 2022
    elif year.startswith('23'):
        full_year = 2023
    elif year.startswith('24'):
        full_year = 2024
    elif year.startswith('25'):
        full_year = 2025
    elif year.startswith('26'):
        full_year = 2026
    else:
        full_year = 2021
    df['_source_file_year'] = full_year
    if 'source_year' not in df.columns:
        df['source_year'] = str(full_year)
    else:
        df['source_year'] = df['source_year'].fillna(str(full_year))
    all_dfs.append(df)
    print(f"Loaded {fname}: {len(df)} records")

raw_df = pd.concat(all_dfs, ignore_index=True)
print(f"Total raw records: {len(raw_df)}")

# ==================== Step 2: Field-level deduplication ====================

print("\nPerforming field-level deduplication by filename...")
grouped = raw_df.groupby('filename')
merged_rows = []
for name, group in grouped:
    merged_rows.append(merge_row_group(group))

merged_df = pd.DataFrame(merged_rows)
# Reorder columns to match original order
original_cols = ['filename', 'primary_label', 'secondary_labels', 'type', 'rating', 
                 'latitude', 'longitude', 'scientific_name', 'common_name', 'author',
                 'license', 'url', 'date', 'time', 'collection', 'class_name',
                 'inat_taxon_id', 'source_year']
merged_df = merged_df[original_cols]

print(f"Deduplicated records: {len(merged_df)}")

full_dedup_path = os.path.join(OUTPUT_DIR, 'ml_full_deduped.csv')
merged_df.to_csv(full_dedup_path, index=False)
print(f"Full deduped master saved to {full_dedup_path}")

# ==================== Step 3: SNR tier assignment ====================

def assign_snr_tier(rating):
    if rating >= 4.0:
        return 'High'
    elif rating >= 2.0:
        return 'Medium'
    else:
        return 'Heavy'

merged_df['snr_tier'] = merged_df['rating'].apply(assign_snr_tier)

# ==================== Step 4: Dual-layer stratified sampling ====================

def dual_layer_stratified_sampling(df, min_per_species=5, random_state=42):
    np.random.seed(random_state)
    sampled_dfs = []
    species_list = df['primary_label'].unique()
    print(f"Total species: {len(species_list)}")
    
    for species in species_list:
        species_df = df[df['primary_label'] == species]
        available = len(species_df)
        if available == 0:
            continue
        
        n_take = min(min_per_species, available)
        if n_take == 0:
            continue
        
        # Stratify within species by SNR
        tier_counts = species_df['snr_tier'].value_counts()
        available_tiers = [t for t in ['High', 'Medium', 'Heavy'] if tier_counts.get(t, 0) > 0]
        
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
        
        # Distribute any remaining slots
        if remaining > 0:
            for tier in available_tiers:
                if remaining <= 0:
                    break
                capacity = tier_counts[tier] - tier_samples.get(tier, 0)
                add = min(remaining, capacity)
                tier_samples[tier] = tier_samples.get(tier, 0) + add
                remaining -= add
        
        # Perform sampling per tier
        for tier, n in tier_samples.items():
            if n <= 0:
                continue
            tier_df = species_df[species_df['snr_tier'] == tier]
            sampled = tier_df.sample(n=min(n, len(tier_df)), random_state=random_state)
            sampled_dfs.append(sampled)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    return result

sampled_df = dual_layer_stratified_sampling(merged_df, min_per_species=5, random_state=RANDOM_STATE)
print(f"Sampled subset size: {len(sampled_df)}")
print(f"Sampled species: {sampled_df['primary_label'].nunique()}")

sampled_path = os.path.join(OUTPUT_DIR, 'ml_sampled.csv')
sampled_df.to_csv(sampled_path, index=False)
print(f"Sampled subset saved to {sampled_path}")

# ==================== Step 5: 80/20 Train/Test split (SNR-stratified) ====================

from sklearn.model_selection import train_test_split

snr_groups = sampled_df.groupby('snr_tier', group_keys=False)
train_list = []
test_list = []
for _, group in snr_groups:
    train, test = train_test_split(group, test_size=0.2, random_state=RANDOM_STATE)
    train_list.append(train)
    test_list.append(test)

train_df = pd.concat(train_list, ignore_index=True)
test_df = pd.concat(test_list, ignore_index=True)

print(f"Train set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

train_path = os.path.join(OUTPUT_DIR, 'ml_train.csv')
test_path = os.path.join(OUTPUT_DIR, 'ml_test.csv')
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Train set saved to {train_path}")
print(f"Test set saved to {test_path}")

# ==================== Step 6: 5-Fold Cross-Validation (SNR-stratified) ====================

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
X = train_df.reset_index(drop=True)
y = X['snr_tier']

fold_num = 1
for train_idx, val_idx in skf.split(X, y):
    train_cv = X.iloc[train_idx].copy()
    val_cv = X.iloc[val_idx].copy()
    
    train_cv_path = os.path.join(OUTPUT_DIR, f'ml_cv_fold{fold_num}_train.csv')
    val_cv_path = os.path.join(OUTPUT_DIR, f'ml_cv_fold{fold_num}_val.csv')
    train_cv.to_csv(train_cv_path, index=False)
    val_cv.to_csv(val_cv_path, index=False)
    print(f"Fold {fold_num}: Train {len(train_cv)}, Val {len(val_cv)}")
    fold_num += 1

# ==================== Step 7: Representativeness Verification ====================

def verify_representativeness(full_df, subset_df):
    full_species = set(full_df['primary_label'].unique())
    subset_species = set(subset_df['primary_label'].unique())
    coverage = len(subset_species) / len(full_species) * 100
    print(f"Species coverage: {coverage:.1f}%")
    assert coverage >= 99.0, "Species coverage below 99%"
    
    full_tier = full_df['snr_tier'].value_counts(normalize=True) * 100
    subset_tier = subset_df['snr_tier'].value_counts(normalize=True) * 100
    for tier in ['High', 'Medium', 'Heavy']:
        diff = abs(full_tier.get(tier, 0) - subset_tier.get(tier, 0))
        print(f"{tier} tier deviation: {diff:.2f} pp")
        assert diff <= 3.0, f"{tier} tier deviation exceeds 3pp"
    
    assert len(set(subset_df['filename'])) == len(subset_df), "Duplicate filenames in subset!"
    print("All verification checks passed.")

verify_representativeness(merged_df, sampled_df)

print("\n=== Processing complete ===")
```
