"""
=============================================================================
UNIFIED CREDIT RISK PIPELINE - PREPROCESSING
=============================================================================
Train-test splitting, imputation, scaling, and resampling (no leakage).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE, TEST_SIZE, CV_FOLDS, MIN_QUANTILE, MAX_QUANTILE


@dataclass
class PreprocessedData:
    """Container for preprocessed train/test data and metadata."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_resampled: pd.DataFrame
    y_train_resampled: pd.Series
    X_train_original: pd.DataFrame    # NEW: unscaled, post-imputation, pre-SMOTE
    X_train_scaled: pd.DataFrame      # NEW: scaled version
    scaler: StandardScaler
    train_medians: pd.Series
    win_bounds: Dict[str, Tuple[float, float]]
    train_index: pd.Index
    test_index: pd.Index
    feature_names: list
    metadata: dict


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str = "HCRL",
    resample: bool = True,
) -> PreprocessedData:
    """
    Complete preprocessing pipeline with proper train-test separation.
    
    Order of operations (CRITICAL to prevent leakage):
    1. Train-test split
    2. Imputation (fit on train, apply to both)
    3. Winsorization (fit on train, apply to both)
    4. Scaling (fit on train, apply to both)
    5. SMOTE+ENN resampling (on train only)
    
    Args:
        X: Feature matrix
        y: Target vector
        target_name: Name of target for reporting
        resample: Whether to apply SMOTE+ENN resampling
    
    Returns:
        PreprocessedData object with all artifacts
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING: TRAIN-TEST SPLIT & TRANSFORMATIONS")
    print("=" * 70)

    X = X.copy().replace([np.inf, -np.inf], np.nan)
    y = y.copy()

    # =========================================================================
    # STEP 1: TRAIN-TEST SPLIT (CRITICAL - DONE FIRST)
    # =========================================================================
    print(f"\n[1/5] Train-test split (80-20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    train_idx = X_train.index
    test_idx = X_test.index

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Train class distribution: {y_train.value_counts(normalize=True).to_dict()}")

    # =========================================================================
    # STEP 2: IMPUTATION (FIT ON TRAIN ONLY)
    # =========================================================================
    print(f"\n[2/5] Imputation (median, fit on train only)...")
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    print(f"  Nulls before: {int(X.isna().sum().sum())} | After: {int(X_train.isna().sum().sum() + X_test.isna().sum().sum())}")

    # =========================================================================
    # STEP 3: WINSORIZATION (FIT ON TRAIN ONLY)
    # =========================================================================
    print(f"\n[3/5] Winsorization (1%-99% bounds, fit on train)...")
    win_bounds = {}
    for col in X_train.columns:
        lo = X_train[col].quantile(MIN_QUANTILE)
        hi = X_train[col].quantile(MAX_QUANTILE)
        win_bounds[col] = (float(lo), float(hi))
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col] = X_test[col].clip(lo, hi)

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"  Applied winsorization bounds to {len(win_bounds)} features")

    # =========================================================================
    # STEP 4: SCALING (FIT ON TRAIN ONLY)
    # =========================================================================
    print(f"\n[4/5] Standardization (fit on train only)...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"  Scaling artifacts ready for both train & test")

    # Save the original (imputed, winsorized, but unscaled) for SHAP
    X_train_original = X_train.copy()

    # =========================================================================
    # STEP 5: RESAMPLING (ON TRAIN ONLY)
    # =========================================================================
    if resample:
        print(f"\n[5/5] SMOTE+ENN resampling (on train only)...")
        minority_count = int((y_train == 1).sum())
        k_neighbors = max(1, min(5, minority_count - 1)) if minority_count > 1 else 1

        smote_enn = SMOTEENN(
            random_state=RANDOM_STATE,
            smote=SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE),
        )
        X_train_res, y_train_res = smote_enn.fit_resample(X_train_scaled, y_train)
        X_train_res = pd.DataFrame(X_train_res, columns=X_train_scaled.columns)
        y_train_res = pd.Series(y_train_res, name=target_name)
        print(f"  Train shape before: {X_train_scaled.shape} | After: {X_train_res.shape}")
        print(f"  Train class balance after resample: {y_train_res.value_counts(normalize=True).to_dict()}")
    else:
        print(f"\n[5/5] No resampling requested")
        X_train_res = X_train_scaled.copy()
        y_train_res = y_train.copy()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    metadata = {
        "target": target_name,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
        "train_class_balance_before": dict(y_train.value_counts()),
        "train_class_balance_after": dict(y_train_res.value_counts()),
        "test_class_balance": dict(y_test.value_counts()),
        "resampled": resample,
    }

    print(f"\n{'='*70}")
    print(f"✓ Preprocessing complete")
    print(f"{'='*70}\n")

    return PreprocessedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        X_train_resampled=X_train_res,
        y_train_resampled=y_train_res,
        X_train_original=X_train_original,
        X_train_scaled=X_train_scaled,
        scaler=scaler,
        train_medians=train_medians,
        win_bounds=win_bounds,
        train_index=train_idx,
        test_index=test_idx,
        feature_names=list(X_train_scaled.columns),
        metadata=metadata,
    )


def apply_train_transforms_to_full_data(
    X_full: pd.DataFrame,
    prep_data: PreprocessedData,
) -> pd.DataFrame:
    """Apply train-fitted transforms (scaler, medians, bounds) to full dataset for scoring."""
    X = X_full.copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(prep_data.train_medians)

    # Apply winsorization bounds
    for col, (lo, hi) in prep_data.win_bounds.items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)

    # Apply scaling
    X_scaled = pd.DataFrame(
        prep_data.scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X_scaled


def get_cv_splitter(y: pd.Series) -> StratifiedKFold:
    """Get stratified k-fold cross-validation splitter."""
    n_folds = 3 if len(y) < 100 else CV_FOLDS
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
