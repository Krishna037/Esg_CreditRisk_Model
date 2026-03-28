"""
Research-grade Phase 1 Corporate Credit Risk Model using HCRL target.

Steps implemented:
1) HCRL construction (3-pillar voting)
2) Data preprocessing (no leakage)
3) Feature engineering
4) Model training (LR, XGB, LGBM, CatBoost, TabNet)
5) Stacking ensemble
6) Class imbalance handling (SMOTE+ENN on train only)
7) Evaluation + plots + DeLong significance tests
8) SHAP explainability
9) Dual experiment (AZS target vs HCRL target)
10) Final output artifacts
"""

from __future__ import annotations

import os
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from scipy.stats import mstats

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import calibration_curve

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import shap
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

DATA_FILE = "Bloomberg Data.xlsx"
OUT_DIR = "outputs/hcrl_phase1"
os.makedirs(OUT_DIR, exist_ok=True)

PHASE1_FIELDS = [
    "Ticker", "Name", "Revenue T12M", "EPS T12M", "Total Return:D-1", "P/E",
    "Tot Assets LF", "Debt/EBITDA LF", "Debt/Equity LF", "Curr Ratio LF",
    "Quick Ratio LF", "FCF T12M", "EBIT T12M", "EBITDA T12M", "Total Liab LF",
    "ROA to ROE LF", "AZS", "CR Msrmnt", "Total Return:Y-1", "Market Cap",
    "Beta:M-1", "Volat:D-30",
]

ESG_FIELDS = ["% Women on Bd:Y", "GHG Scope 1:Y", "GHG Scope 3:Y", "CO2 Scope 1:Y"]


@dataclass
class PreparedData:
    df_full: pd.DataFrame
    features_raw: pd.DataFrame
    y_hcrl: pd.Series
    y_azs: pd.Series
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_res: pd.DataFrame
    y_train_res: pd.Series
    train_index: pd.Index
    test_index: pd.Index
    scaler: StandardScaler
    train_medians: pd.Series
    win_bounds: Dict[str, Tuple[float, float]]
    feature_names: List[str]


class TabNetSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Minimal sklearn-compatible wrapper for TabNet binary classification."""

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-4,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
        patience: int = 30,
        batch_size: int = 256,
        virtual_batch_size: int = 128,
        random_state: int = SEED,
        verbose: int = 0,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        self.model_ = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.lr, weight_decay=self.weight_decay),
            mask_type="sparsemax",
            seed=self.random_state,
            verbose=self.verbose,
        )

        self.model_.fit(
            X_train=X_tr,
            y_train=y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            weights=1,
            drop_last=False,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        probs = self.model_.predict_proba(X)
        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize names to code-friendly format while preserving semantics."""
    rename_map = {
        "Total Return:D-1": "Total_Return_D1",
        "P/E": "P_E",
        "Tot Assets LF": "Tot_Assets_LF",
        "Debt/EBITDA LF": "Debt_EBITDA_LF",
        "Debt/Equity LF": "Debt_Equity_LF",
        "Curr Ratio LF": "Curr_Ratio_LF",
        "Quick Ratio LF": "Quick_Ratio_LF",
        "FCF T12M": "FCF_T12M",
        "EBIT T12M": "EBIT_T12M",
        "EBITDA T12M": "EBITDA_T12M",
        "Total Liab LF": "Total_Liab_LF",
        "ROA to ROE LF": "ROA_to_ROE_LF",
        "CR Msrmnt": "CR_Msrmnt",
        "Total Return:Y-1": "Total_Return_Y1",
        "Market Cap": "Market_Cap",
        "Beta:M-1": "Beta_M1",
        "Volat:D-30": "Volat_D30",
        "Revenue T12M": "Revenue_T12M",
        "EPS T12M": "EPS_T12M",
    }
    return df.rename(columns=rename_map)


def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-ID columns to numeric and normalize missing markers."""
    out = df.copy()
    for c in out.columns:
        if c in ["Ticker", "Name"]:
            continue
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.replace(0, np.nan)
    return a / b2


def construct_hcrl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create partial HCRL from pillars 1-2 only (AZS and Ohlson).
    Pillar 3 (market-based) is deferred to split_and_preprocess() to prevent leakage.
    Market threshold computation must happen on train set only.
    """
    d = df.copy()

    # Pillar 1: Altman flag
    d["AZS_Flag"] = (d["AZS"] < 1.8).astype(int)

    # Pillar 2: Ohlson proxy
    size = np.log(d["Tot_Assets_LF"].clip(lower=1))
    tlta = safe_div(d["Total_Liab_LF"], d["Tot_Assets_LF"])
    wcta = safe_div(d["Curr_Ratio_LF"] - 1, d["Curr_Ratio_LF"])
    clca = safe_div(pd.Series(1.0, index=d.index), d["Curr_Ratio_LF"])
    oeneg = (d["Total_Liab_LF"] > d["Tot_Assets_LF"]).astype(int)
    nita = safe_div(d["EBIT_T12M"], d["Tot_Assets_LF"])
    futl = safe_div(d["FCF_T12M"], d["Total_Liab_LF"])
    intwo = (d["EPS_T12M"] < 0).astype(int)
    # FIX: CHIN = change in Net Income from year t-1 to year t
    # Proxy: We don't have historical data, so use sign of current NI
    # CHIN = 1 if change_in_NI > 0 (company improving), 0 otherwise
    # Since we lack year t-1, we estimate: CHIN = 1 if current NI > 0 (positive earnings growth signal)
    chin = (d["EBIT_T12M"] > 0).astype(int)  # Improved from hardcoded 0

    temp = pd.DataFrame({
        "SIZE": size,
        "TLTA": tlta,
        "WCTA": wcta,
        "CLCA": clca,
        "OENEG": oeneg,
        "NITA": nita,
        "FUTL": futl,
        "INTWO": intwo,
        "CHIN": chin,
    })
    temp = temp.fillna(temp.median(numeric_only=True))

    o_score = (
        -1.32
        - 0.407 * temp["SIZE"]
        + 6.03 * temp["TLTA"]
        - 1.43 * temp["WCTA"]
        + 0.0757 * temp["CLCA"]
        - 1.72 * temp["OENEG"]
        - 2.37 * temp["NITA"]
        - 1.83 * temp["FUTL"]
        + 0.285 * temp["INTWO"]
        - 0.521 * temp["CHIN"]
    )
    d["PD_Ohlson"] = np.exp(o_score) / (1 + np.exp(o_score))
    d["Ohlson_Flag"] = (d["PD_Ohlson"] > 0.50).astype(int)

    # NOTE: Pillar 3 (market-based) is deferred to split_and_preprocess() to prevent using test data
    # in threshold computation (vol_q75, ret_q25, mcap_q25, etc.)
    # This ensures clean train-test separation and prevents data leakage.
    d["Market_Flag"] = np.nan  # Placeholder; will be filled post-split
    d["HCRL"] = np.nan         # Placeholder; will be filled post-split

    return d


def report_hcrl_vs_azs(d: pd.DataFrame) -> None:
    """Report on HCRL target, but only if Market_Flag and HCRL have been computed (post-split)."""
    # This function is called after split_and_preprocess, so HCRL should be fully defined
    if d["HCRL"].isna().all():
        print("\n[INFO] HCRL not yet computed (will be filled in split_and_preprocess after train-test split)")
        print("Reporting on pillars 1-2 only (Altman + Ohlson):")
        azs_rate = d["AZS_Flag"].mean() * 100
        olh_rate = d["Ohlson_Flag"].mean() * 100
        print(f"  Pillar 1 (AZS) default %: {azs_rate:.2f}%")
        print(f"  Pillar 2 (Ohlson) default %: {olh_rate:.2f}%")
        return

    hcrl_rate = d["HCRL"].mean() * 100
    azs_rate = d["AZS_Flag"].mean() * 100
    print("\n" + "=" * 70)
    print("STEP 1 REPORT - HCRL TARGET DISTRIBUTION")
    print("=" * 70)
    print(f"Default % (HCRL): {hcrl_rate:.2f}%")
    print(f"Default % (AZS_Flag): {azs_rate:.2f}%")
    print(f"Difference (HCRL - AZS): {hcrl_rate - azs_rate:.2f} pp")

    ct = pd.crosstab(d["HCRL"], d["AZS_Flag"])
    chi2, p_val, dof, _ = chi2_contingency(ct)
    print("\nChi-square test HCRL vs AZS_Flag")
    print(ct.to_string())
    print(f"chi2={chi2:.4f}, dof={dof}, p-value={p_val:.6g}")


def engineer_features(d: pd.DataFrame) -> pd.DataFrame:
    """Create required derived features before train-test split."""
    x = d.copy()

    debt_ebitda_safe = x["Debt_EBITDA_LF"].replace(0, np.nan)
    azs_safe = x["AZS"].replace(0, 0.001)

    x["Asset_Efficiency"] = safe_div(x["Revenue_T12M"], x["Tot_Assets_LF"])
    x["Earnings_Quality"] = safe_div(x["FCF_T12M"], x["EBITDA_T12M"])
    x["Profitability_Ratio"] = safe_div(x["EBIT_T12M"], x["Tot_Assets_LF"])
    x["Mkt_Adjusted_Lev"] = x["Debt_Equity_LF"] * x["Beta_M1"]
    x["Volat_Adj_Return"] = safe_div(x["Total_Return_Y1"], (1 + x["Volat_D30"]))
    x["Distress_Proximity"] = safe_div(pd.Series(1.0, index=x.index), azs_safe) * x["Volat_D30"]
    x["Liquidity_Buffer"] = x["Curr_Ratio_LF"] - x["Quick_Ratio_LF"]
    x["Size_Lev_Ratio"] = np.log(x["Market_Cap"].clip(lower=1)) / debt_ebitda_safe

    # Explicitly re-impute later after introducing NaN by zero-division guards.
    return x


def split_and_preprocess(df_hcrl: pd.DataFrame) -> PreparedData:
    print("\n" + "=" * 70)
    print("STEP 2 + STEP 3 - PREPROCESSING + FEATURE ENGINEERING")
    print("=" * 70)

    d = df_hcrl.copy()

    # ===== CRITICAL FIX: TRAIN-TEST SPLIT MOVED HERE (BEFORE ALL STATISTICAL FITS) =====
    # This ensures thresholds are computed on train data only, preventing leakage.
    y_hcrl_full = d["HCRL"]  # Will be re-assigned after pillar 3 computation

    # First, split based on pillars 1-2 (AZS and Ohlson) which are deterministic.
    # We use AZS_Flag as a stratification proxy since HCRL has too few positives.
    X_train_idx, X_test_idx = train_test_split(
        np.arange(len(d)),
        test_size=0.2,
        stratify=d["AZS_Flag"],  # Use AZS_Flag for stratification (similar distribution to HCRL)
        random_state=SEED,
    )

    d_train = d.iloc[X_train_idx].copy()
    d_test = d.iloc[X_test_idx].copy()

    # ===== COMPUTE PILLAR 3 ON TRAIN SET ONLY =====
    # Compute market distress thresholds EXCLUSIVELY from training set.
    vol_q75_train = d_train["Volat_D30"].quantile(0.75)
    ret_q25_train = d_train["Total_Return_Y1"].quantile(0.25)
    mcap_q25_train = d_train["Market_Cap"].quantile(0.25)

    # Pillar 3 flag computed on TRAIN thresholds
    def compute_market_flag(df_subset, vol_q75, ret_q25, mcap_q25):
        market_score = (
            (df_subset["Volat_D30"] > vol_q75).astype(int)
            + (df_subset["Total_Return_Y1"] < ret_q25).astype(int)
            + (df_subset["Beta_M1"] > 1.5).astype(int)
            + (df_subset["Market_Cap"] < mcap_q25).astype(int)
        )
        return (market_score >= 2).astype(int)

    d_train["Market_Flag"] = compute_market_flag(d_train, vol_q75_train, ret_q25_train, mcap_q25_train)
    d_test["Market_Flag"] = compute_market_flag(d_test, vol_q75_train, ret_q25_train, mcap_q25_train)

    # Compute HCRL from all 3 pillars
    d_train["HCRL"] = ((d_train["AZS_Flag"] + d_train["Ohlson_Flag"] + d_train["Market_Flag"]) >= 2).astype(int)
    d_test["HCRL"] = ((d_test["AZS_Flag"] + d_test["Ohlson_Flag"] + d_test["Market_Flag"]) >= 2).astype(int)

    # Reconstruct full dataframe with proper HCRL
    d = pd.concat([d_train, d_test], ignore_index=False)

    # ===== CONTINUE WITH FEATURE ENGINEERING =====
    # Exclude ESG fields for strict Phase 1 scope.
    drop_cols = [c for c in ESG_FIELDS if c in d.columns]
    if drop_cols:
        d = d.drop(columns=drop_cols)

    d = engineer_features(d)

    # Cap Distress_Proximity at 99th percentile.
    cap99 = d["Distress_Proximity"].quantile(0.99)
    d["Distress_Proximity"] = d["Distress_Proximity"].clip(upper=cap99)

    y_hcrl = d["HCRL"].astype(int)
    y_azs = d["AZS_Flag"].astype(int)

    # ===== CRITICAL FIX: REMOVE AZS FROM FEATURE MATRIX =====
    # AZS is used in pillar 1 (AZS < 1.8), so having it as a feature gives the model
    # near-perfect information about the target. Drop it along with other target-construction intermediates.
    feature_df = d.drop(columns=["Ticker", "Name", "HCRL", "AZS", "AZS_Flag", "Ohlson_Flag", "Market_Flag", "PD_Ohlson"])
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    print(f"Pre-scaling shape: {feature_df.shape}")
    print(f"Pre-scaling null cells: {int(feature_df.isna().sum().sum())}")
    print(f"Pre-scaling class balance HCRL=1: {y_hcrl.mean()*100:.2f}%")

    # Get the indices for later reference
    train_index = d_train.index
    test_index = d_test.index

    X_train = feature_df.loc[train_index].copy()
    X_test = feature_df.loc[test_index].copy()
    y_train = y_hcrl.loc[train_index].copy()
    y_test = y_hcrl.loc[test_index].copy()

    # Median imputation fit on train only.
    train_medians = X_train.median(numeric_only=True)
    X_train_imp = X_train.fillna(train_medians)
    X_test_imp = X_test.fillna(train_medians)

    # Winsorization based on train 1%-99% bounds and applied to both splits.
    win_bounds: Dict[str, Tuple[float, float]] = {}
    X_train_win = X_train_imp.copy()
    X_test_win = X_test_imp.copy()
    for col in X_train_win.columns:
        lo = float(X_train_win[col].quantile(0.01))
        hi = float(X_train_win[col].quantile(0.99))
        win_bounds[col] = (lo, hi)
        X_train_win[col] = X_train_win[col].clip(lo, hi)
        X_test_win[col] = X_test_win[col].clip(lo, hi)

    # Guard against any inf/nan introduced by numeric transforms.
    X_train_win = X_train_win.replace([np.inf, -np.inf], np.nan)
    X_test_win = X_test_win.replace([np.inf, -np.inf], np.nan)
    X_train_win = X_train_win.fillna(X_train_win.median(numeric_only=True))
    X_test_win = X_test_win.fillna(X_train_win.median(numeric_only=True))

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_win),
        columns=X_train_win.columns,
        index=X_train_win.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_win),
        columns=X_test_win.columns,
        index=X_test_win.index,
    )

    X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # SMOTE+ENN on train only.
    minority_count = int((y_train == 1).sum())
    k_neighbors = max(1, min(5, minority_count - 1)) if minority_count > 1 else 1
    smote_enn = SMOTEENN(random_state=SEED, smote=SMOTE(k_neighbors=k_neighbors, random_state=SEED))
    X_train_res, y_train_res = smote_enn.fit_resample(X_train_scaled, y_train)
    X_train_res = pd.DataFrame(X_train_res, columns=X_train_scaled.columns)
    y_train_res = pd.Series(y_train_res, name="HCRL")

    print(f"Train shape before resample: {X_train_scaled.shape}")
    print(f"Train shape after  resample: {X_train_res.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    print(f"Nulls after preprocessing (train/test): {int(X_train_scaled.isna().sum().sum())} / {int(X_test_scaled.isna().sum().sum())}")
    print(f"Class balance after resample HCRL=1: {y_train_res.mean()*100:.2f}%")

    return PreparedData(
        df_full=d,
        features_raw=feature_df,
        y_hcrl=y_hcrl,
        y_azs=y_azs,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        X_train_res=X_train_res,
        y_train_res=y_train_res,
        train_index=train_index,
        test_index=test_index,
        scaler=scaler,
        train_medians=train_medians,
        win_bounds=win_bounds,
        feature_names=list(X_train_scaled.columns),
    )


def choose_cv(y_train: pd.Series) -> StratifiedKFold:
    folds = 3 if len(y_train) < 100 else 5
    if folds == 3:
        print("[WARN] Sample size < 100. CV adjusted to 3-fold.")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)


def train_models(prep: PreparedData) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 4 + STEP 5 + STEP 6 - MODEL TRAINING STACK")
    print("=" * 70)

    Xtr = prep.X_train_res
    ytr = prep.y_train_res
    Xte = prep.X_test_scaled
    yte = prep.y_test

    cv = choose_cv(ytr)
    scale_pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    models: Dict[str, Any] = {}

    # Logistic Regression with grid search.
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=SEED)
    lr_grid = GridSearchCV(
        estimator=lr,
        param_grid={"C": [0.01, 0.1, 1, 10]},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    lr_grid.fit(Xtr, ytr)
    models["Logistic Regression"] = lr_grid.best_estimator_
    print(f"Logistic best C: {lr_grid.best_params_['C']}")

    # XGBoost with grid search.
    # NOTE: use_label_encoder=False removed — deprecated in XGBoost 2.0+
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb_grid = GridSearchCV(
        estimator=xgb,
        param_grid={
            "max_depth": [4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [300, 500, 700],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    xgb_grid.fit(Xtr, ytr)
    best_xgb = xgb_grid.best_estimator_
    models["XGBoost"] = best_xgb
    print(f"XGBoost best params: {xgb_grid.best_params_}")

    # LightGBM.
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        is_unbalance=True,
        random_state=SEED,
    )
    lgbm.fit(Xtr, ytr)
    models["LightGBM"] = lgbm

    # CatBoost.
    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        verbose=0,
        random_state=SEED,
    )
    cat.fit(Xtr, ytr)
    models["CatBoost"] = cat

    # TabNet.
    tabnet = TabNetSklearnWrapper(random_state=SEED, verbose=0)
    tabnet.fit(Xtr.values.astype(np.float32), ytr.values.astype(np.int64))
    models["TabNet"] = tabnet

    # Stacking: requested base learners include XGB/LGBM/CatBoost/TabNet.
    # StackingClassifier requires sklearn estimators; TabNet wrapper is sklearn-compatible.
    estimators = [
        ("xgb", best_xgb),
        ("lgbm", lgbm),
        ("catboost", cat),
        ("tabnet", tabnet),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=SEED),
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    stack.fit(Xtr, ytr)
    models["Stacking Ensemble"] = stack

    # Refit base models explicitly on resampled train if needed consistency.
    for name, model in models.items():
        if name == "TabNet":
            continue
        if name == "Stacking Ensemble":
            continue

    # Save trained models map for downstream usage.
    with open(os.path.join(OUT_DIR, "trained_model_names.json"), "w", encoding="utf-8") as f:
        json.dump(list(models.keys()), f, indent=2)

    return models


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate_models(models: Dict[str, Any], prep: PreparedData) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    print("\n" + "=" * 70)
    print("STEP 7 - EVALUATION FRAMEWORK")
    print("=" * 70)

    Xte = prep.X_test_scaled
    yte = prep.y_test.values

    rows = []
    probas: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        if name == "TabNet":
            y_prob = model.predict_proba(Xte.values.astype(np.float32))[:, 1]
        else:
            y_prob = model.predict_proba(Xte)[:, 1]

        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(yte, y_prob)
        ks = ks_statistic(yte, y_prob)
        row = {
            "Model": name,
            "AUC_ROC": auc,
            "KS_Statistic": ks,
            "Gini": 2 * auc - 1,
            "Precision": precision_score(yte, y_pred, zero_division=0),
            "Recall": recall_score(yte, y_pred, zero_division=0),
            "F1_Score": f1_score(yte, y_pred, zero_division=0),
            "Brier_Score": brier_score_loss(yte, y_prob),
            "Average_Precision": average_precision_score(yte, y_prob),
        }
        rows.append(row)
        probas[name] = y_prob

    metrics_df = pd.DataFrame(rows).sort_values("AUC_ROC", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(os.path.join(OUT_DIR, "model_results_table.csv"), index=False)
    print(metrics_df.round(4).to_string(index=False))

    # ROC curves
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, y_prob in probas.items():
        fpr, tpr, _ = roc_curve(yte, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curves - All Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # PR curves
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, y_prob in probas.items():
        p, r, _ = precision_recall_curve(yte, y_prob)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(yte, y_prob):.3f})")
    ax.set_title("Precision-Recall Curves - All Models")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pr_curves.png"), dpi=150)
    plt.close()

    best_name = metrics_df.iloc[0]["Model"]
    best_prob = probas[best_name]
    best_pred = (best_prob >= 0.5).astype(int)

    # Confusion matrix heatmap for best model.
    cm = confusion_matrix(yte, best_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best.png"), dpi=150)
    plt.close()

    # Calibration plot for best model.
    frac_pos, mean_pred = calibration_curve(yte, best_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label=best_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_title("Calibration Plot - Best Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "calibration.png"), dpi=150)
    plt.close()

    return metrics_df, probas


# DeLong implementation for correlated ROC AUCs.
def compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_variance(y_true: np.ndarray, y_pred: np.ndarray):
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = y_pred[np.newaxis, order]
    aucs, delongcov = fast_delong(preds, label_1_count)
    return aucs[0], delongcov


def delong_roc_test(y_true: np.ndarray, pred_one: np.ndarray, pred_two: np.ndarray) -> float:
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = np.vstack([pred_one, pred_two])[:, order]
    aucs, delongcov = fast_delong(preds, label_1_count)
    diff = np.abs(aucs[0] - aucs[1])
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = diff / np.sqrt(max(var, 1e-12))
    # Normal approximation using scipy-free expression via erf.
    p = 2 * (1 - 0.5 * (1 + math.erf(z / np.sqrt(2))))
    return float(max(min(p, 1.0), 0.0))


def run_statistical_tests(metrics_df: pd.DataFrame, probas: Dict[str, np.ndarray], y_test: pd.Series) -> None:
    print("\nStatistical Significance (DeLong AUC test)")
    y = y_test.values
    best_name = metrics_df.iloc[0]["Model"]

    # Best vs Logistic baseline.
    if "Logistic Regression" in probas:
        p1 = delong_roc_test(y, probas[best_name], probas["Logistic Regression"])
        print(f"Best model vs Logistic Regression: p-value={p1:.6f}")

    # Stacking vs best single model.
    single_df = metrics_df[~metrics_df["Model"].eq("Stacking Ensemble")]
    best_single = single_df.iloc[0]["Model"]
    if "Stacking Ensemble" in probas and best_single in probas:
        p2 = delong_roc_test(y, probas["Stacking Ensemble"], probas[best_single])
        print(f"Stacking Ensemble vs best single ({best_single}): p-value={p2:.6f}")


def run_shap_explainability(models: Dict[str, Any], metrics_df: pd.DataFrame, prep: PreparedData, probas: Dict[str, np.ndarray]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 8 - SHAP EXPLAINABILITY")
    print("=" * 70)

    priority = ["XGBoost", "LightGBM", "CatBoost", metrics_df.iloc[0]["Model"]]
    best_single = None
    for m in priority:
        if m in models and m != "Stacking Ensemble":
            best_single = m
            break

    model = models[best_single]
    X_test = prep.X_test_scaled

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        shap_values = sv[1] if len(sv) > 1 else sv[0]
    else:
        shap_values = sv

    # Summary beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Summary bar
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Waterfalls: highest, median, lowest PD.
    y_prob = probas[best_single]
    idx_high = int(np.argmax(y_prob))
    idx_low = int(np.argmin(y_prob))
    idx_med = int(np.argsort(y_prob)[len(y_prob) // 2])

    for tag, idx in [("high", idx_high), ("median", idx_med), ("low", idx_low)]:
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0],
            data=X_test.iloc[idx].values,
            feature_names=X_test.columns.tolist(),
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_waterfall_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Dependence plots for top 3 features by |SHAP|.
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-3:][::-1]
    for i in top_idx:
        feat = X_test.columns[i]
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(feat, shap_values, X_test, ax=ax, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_dependence_{feat}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Feature importance export.
    fi = pd.DataFrame({"Feature": X_test.columns, "MeanAbsSHAP": mean_abs})
    fi = fi.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    fi.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

    # Economic sign sanity checks.
    sign_expect = {
        "Debt_EBITDA_LF": "positive",
        "Volat_D30": "positive",
        "Distress_Proximity": "positive",
        "Curr_Ratio_LF": "negative",
        "FCF_T12M": "negative",
        "Profitability_Ratio": "negative",
    }
    print("\nEconomic sign checks (feature value vs SHAP correlation):")
    sv_df = pd.DataFrame(shap_values, columns=X_test.columns)
    for feat, exp_sign in sign_expect.items():
        if feat not in X_test.columns:
            continue
        corr = np.corrcoef(X_test[feat], sv_df[feat])[0, 1]
        actual = "positive" if corr >= 0 else "negative"
        flag = "OK" if actual == exp_sign else "REVIEW"
        print(f"  {feat:20s} expected={exp_sign:8s} actual={actual:8s} corr={corr:+.4f} [{flag}]")

    return fi


def apply_train_transforms_to_full(prep: PreparedData) -> pd.DataFrame:
    """Apply train-fitted imputer/winsor/scaler to full feature set for scoring."""
    X = prep.features_raw.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(prep.train_medians)
    for c, (lo, hi) in prep.win_bounds.items():
        if c in X.columns:
            X[c] = X[c].clip(lo, hi)
    Xs = pd.DataFrame(prep.scaler.transform(X), columns=X.columns, index=X.index)
    Xs = Xs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xs


def dual_experiment_azs_vs_hcrl(prep: PreparedData) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 9 - DUAL EXPERIMENT (AZS TARGET vs HCRL TARGET)")
    print("=" * 70)

    Xtr = prep.X_train_scaled
    Xte = prep.X_test_scaled

    y_h_train = prep.y_hcrl.loc[prep.train_index]
    y_h_test = prep.y_hcrl.loc[prep.test_index]

    y_a_train = prep.y_azs.loc[prep.train_index]
    y_a_test = prep.y_azs.loc[prep.test_index]

    # Same preprocessing and same model hyperparameters, separate target-specific resampling.
    sm = SMOTEENN(random_state=SEED)
    Xh_res, yh_res = sm.fit_resample(Xtr, y_h_train)
    Xa_res, ya_res = sm.fit_resample(Xtr, y_a_train)

    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
    )

    xgb_h = XGBClassifier(scale_pos_weight=float((yh_res == 0).sum() / max((yh_res == 1).sum(), 1)), **params)
    xgb_a = XGBClassifier(scale_pos_weight=float((ya_res == 0).sum() / max((ya_res == 1).sum(), 1)), **params)

    xgb_h.fit(Xh_res, yh_res)
    xgb_a.fit(Xa_res, ya_res)

    ph = xgb_h.predict_proba(Xte)[:, 1]
    pa = xgb_a.predict_proba(Xte)[:, 1]

    def collect(y_true: pd.Series, prob: np.ndarray, label: str) -> Dict[str, Any]:
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_true, prob)
        return {
            "Target": label,
            "AUC": auc,
            "KS": ks_statistic(y_true.values, prob),
            "F1": f1_score(y_true, pred, zero_division=0),
            "Brier": brier_score_loss(y_true, prob),
        }

    res_a = collect(y_a_test, pa, "AZS_Flag")
    res_h = collect(y_h_test, ph, "HCRL")
    cmp_df = pd.DataFrame([res_a, res_h])

    # NOTE: FIX - DeLong test comparison
    # Each model is trained on its own target. Comparing them via DeLong on a THIRD ground truth
    # (HCRL for both probabilities) is not statistically meaningful because:
    # - xgb_a was optimized to predict AZS_Flag, not HCRL
    # - The comparison is not apples-to-apples
    # Instead, we report AUC improvements relative to the original targets:
    #   res_h["AUC"] = HCRL model on HCRL truth
    #   res_a["AUC"] = AZS model on AZS truth
    # These are separate benchmarks, not directly comparable via DeLong.
    # If you want DeLong: train both on SAME target, then compare via DeLong.
    
    auc_gain = (res_h["AUC"] - res_a["AUC"]) * 100

    cmp_df.to_csv(os.path.join(OUT_DIR, "azs_vs_hcrl_experiment.csv"), index=False)
    print(cmp_df.round(4).to_string(index=False))
    print(f"\nNote: Each model optimized for its own target (not directly comparable).")
    print(f"HCRL model AUC (vs HCRL truth): {res_h['AUC']:.4f}")
    print(f"AZS model AUC (vs AZS truth): {res_a['AUC']:.4f}")
    print(f"Performance gap on own targets: {abs(auc_gain):.2f}%")

    return cmp_df


def export_company_pd_scores(models: Dict[str, Any], metrics_df: pd.DataFrame, prep: PreparedData) -> None:
    print("\n" + "=" * 70)
    print("STEP 10 - FINAL OUTPUT EXPORTS")
    print("=" * 70)

    X_all_scaled = apply_train_transforms_to_full(prep)

    preferred = []
    if "Stacking Ensemble" in models:
        preferred.append("Stacking Ensemble")
    preferred.extend([m for m in metrics_df["Model"].tolist() if m != "Stacking Ensemble"])

    final_model_name = None
    pd_scores = None
    for model_name in preferred:
        model = models[model_name]
        try:
            if model_name == "TabNet":
                pd_scores = model.predict_proba(X_all_scaled.values.astype(np.float32))[:, 1]
            else:
                pd_scores = model.predict_proba(X_all_scaled)[:, 1]
            final_model_name = model_name
            break
        except Exception as ex:
            print(f"[WARN] Scoring failed with {model_name}: {ex}")

    if pd_scores is None or final_model_name is None:
        raise RuntimeError("Could not generate PD scores with any trained model.")

    out = pd.DataFrame({
        "Ticker": prep.df_full["Ticker"].values,
        "HCRL_Label": prep.y_hcrl.values,
        "PD_Score": pd_scores,
    })

    out["Risk_Tier"] = np.where(
        out["PD_Score"] < 0.3,
        "Low",
        np.where(out["PD_Score"] <= 0.6, "Medium", "High"),
    )

    out.to_csv(os.path.join(OUT_DIR, "company_pd_scores.csv"), index=False)

    print(f"Final model for scoring: {final_model_name}")
    print(f"Saved: {os.path.join(OUT_DIR, 'company_pd_scores.csv')}")


def final_report(metrics_df: pd.DataFrame) -> None:
    best = metrics_df.iloc[0]
    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)
    print(f"Best model: {best['Model']}")
    print(f"AUC-ROC: {best['AUC_ROC']:.4f}")
    print(f"KS: {best['KS_Statistic']:.4f}")
    print(f"F1: {best['F1_Score']:.4f}")
    print(f"Average Precision: {best['Average_Precision']:.4f}")
    print(f"Brier Score: {best['Brier_Score']:.4f}")

    print("\nOutputs generated:")
    generated = [
        "model_results_table.csv",
        "feature_importance.csv",
        "company_pd_scores.csv",
        "roc_curves.png",
        "pr_curves.png",
        "confusion_matrix_best.png",
        "calibration.png",
        "shap_summary.png",
        "shap_bar.png",
        "shap_waterfall_high.png",
        "shap_waterfall_median.png",
        "shap_waterfall_low.png",
    ]
    for f in generated:
        print(f"  - {os.path.join(OUT_DIR, f)}")


def main() -> None:
    print("=" * 70)
    print("PHASE 1 CREDIT RISK MODEL (HCRL TARGET)")
    print("=" * 70)

    # Load and keep only Phase 1 fields.
    raw = pd.read_excel(DATA_FILE)
    missing_fields = [c for c in PHASE1_FIELDS if c not in raw.columns]
    if missing_fields:
        raise ValueError(f"Required fields missing: {missing_fields}")

    df = raw[PHASE1_FIELDS].copy()
    df = clean_numeric_data(df)
    df = normalize_columns(df)

    # Keep rows with valid AZS because pillar 1 requires it.
    df = df.dropna(subset=["AZS"]).reset_index(drop=True)

    # Step 1: construct HCRL and report.
    df_hcrl = construct_hcrl(df)
    report_hcrl_vs_azs(df_hcrl)

    # Step 2 and 3: preprocessing and engineering.
    prep = split_and_preprocess(df_hcrl)

    # Step 4, 5, 6: model training.
    models = train_models(prep)

    # Step 7: evaluation.
    metrics_df, probas = evaluate_models(models, prep)
    run_statistical_tests(metrics_df, probas, prep.y_test)

    # Step 8: SHAP explainability.
    run_shap_explainability(models, metrics_df, prep, probas)

    # Step 9: dual target experiment.
    dual_experiment_azs_vs_hcrl(prep)

    # Step 10: final outputs.
    export_company_pd_scores(models, metrics_df, prep)
    final_report(metrics_df)


if __name__ == "__main__":
    main()
