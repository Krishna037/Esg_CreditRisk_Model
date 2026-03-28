"""
=============================================================================
CREDIT RISK MODEL — PHASE 1: DATA PREPROCESSING & FEATURE ENGINEERING
=============================================================================
Bloomberg Data: 502 companies x 26 fields
Target: Binary default variable derived from Altman Z-Score (AZS < 1.8 = Default)

Steps:
  1. Load data & exploratory analysis
  2. Drop identifiers, handle missing values
  3. Construct binary target variable from AZS
  4. Feature engineering (8 derived ratios)
  5. Winsorization (1st-99th percentile)
  6. Train/Test split (80/20, stratified)
  7. StandardScaler (fit on train only)
  8. Class imbalance handling (SMOTE+ENN)
  9. Save processed datasets
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving figures
import seaborn as sns
from scipy.stats import mstats
import os

# ── optional imports (install if missing) ─────────────────────────────────
try:
    from imblearn.combine import SMOTEENN
    SMOTE_AVAILABLE = True
except ImportError:
    print("[WARN] imbalanced-learn not found. Run: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── output directory ───────────────────────────────────────────────────────
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── data cleaning helper ──────────────────────────────────────────────────
def clean_dataframe(df):
    """
    Bloomberg Excel often stores missing values as spaces (' ') and
    encodes numeric columns as object dtype.
    This function:
      1. Strips string whitespace in every cell
      2. Replaces blank / whitespace-only strings with NaN
      3. Coerces all non-identifier columns to numeric
    """
    ID_COLS = {"Ticker", "Name"}
    for col in df.columns:
        if col in ID_COLS:
            continue
        # Strip strings and replace blanks with NaN
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        # Coerce to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("STEP 1 — LOADING BLOOMBERG DATA")
print("=" * 60)

df_raw = pd.read_excel("Bloomberg Data.xlsx")
print(f"Raw shape  : {df_raw.shape}")
print(f"Columns    : {df_raw.columns.tolist()}\n")

# Clean: convert space-strings → NaN, coerce object cols → numeric
df_raw = clean_dataframe(df_raw)
print("Data cleaning applied (space-strings → NaN, objects → numeric).")

# Save a copy to work on
df = df_raw.copy()

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("=" * 60)
print("STEP 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\n── Missing Values ──")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
miss_df = miss_df[miss_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
print(miss_df.to_string())

print("\n── Descriptive Statistics ──")
print(df.describe().round(3).to_string())

# Plot missing values heatmap
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(df.isnull().T, cbar=True, yticklabels=True, ax=ax,
            cmap="YlOrRd", linewidths=0.3)
ax.set_title("Missing Value Heatmap (Bloomberg Dataset)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_missing_heatmap.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR}/01_missing_heatmap.png")

# AZS distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["AZS"].dropna().hist(bins=40, color="#4e79a7", edgecolor="black", ax=axes[0])
axes[0].axvline(1.8, color="red", linestyle="--", linewidth=2, label="Default threshold (1.8)")
axes[0].axvline(2.99, color="orange", linestyle="--", linewidth=2, label="Safe threshold (2.99)")
axes[0].set_title("Altman Z-Score Distribution")
axes[0].set_xlabel("AZS")
axes[0].legend()
axes[1].boxplot(df["AZS"].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor="#4e79a7"))
axes[1].set_title("AZS Box Plot")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_azs_distribution.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/02_azs_distribution.png")

# =============================================================================
# 3. DROP IDENTIFIERS & HANDLE MISSING TARGET
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — TARGET VARIABLE CONSTRUCTION")
print("=" * 60)

# Drop rows where AZS is missing (never impute target)
df.dropna(subset=["AZS"], inplace=True)
print(f"Rows after dropping missing AZS: {len(df)}")

# Binary target: AZS < 1.8 → Default (1), else → Safe (0)
df["Default"] = (df["AZS"] < 1.8).astype(int)

# Class distribution
counts = df["Default"].value_counts()
print(f"\nClass Distribution:")
print(f"  Safe    (0): {counts.get(0, 0)}  ({counts.get(0,0)/len(df)*100:.1f}%)")
print(f"  Default (1): {counts.get(1, 0)}  ({counts.get(1,0)/len(df)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Safe (0)", "Default (1)"], [counts.get(0,0), counts.get(1,0)],
       color=["#59a14f", "#e15759"], edgecolor="black")
ax.set_title("Class Distribution after AZS Binary Encoding")
ax.set_ylabel("Count")
for i, v in enumerate([counts.get(0,0), counts.get(1,0)]):
    ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_class_distribution.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/03_class_distribution.png")

# Zone breakdown (for reference)
df["AZS_Zone"] = pd.cut(
    df["AZS"],
    bins=[-np.inf, 1.8, 2.99, np.inf],
    labels=["Distress", "Grey", "Safe"]
)
print("\nAZS Zone Breakdown:")
print(df["AZS_Zone"].value_counts().to_string())

# ── Save a metadata frame with Ticker, Name, raw AZS, Zone, Default ──────
# We'll join this back to predictions in Phase 2.
df_meta = df[["Ticker", "Name", "AZS", "AZS_Zone", "Default"]].copy() if "Ticker" in df.columns \
    else df[["AZS", "AZS_Zone", "Default"]].copy()
print(f"\nMetadata frame shape: {df_meta.shape}")
print(df_meta.head(3).to_string())

# Drop original AZS and zone column before modelling
df.drop(columns=["AZS", "AZS_Zone"], inplace=True)

# Drop identifiers (kept in df_meta)
df.drop(columns=["Ticker", "Name"], inplace=True, errors="ignore")

# =============================================================================
# 4. MISSING VALUE TREATMENT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4 — MISSING VALUE TREATMENT")
print("=" * 60)

# Separate target
y = df["Default"]
X = df.drop(columns=["Default"])

# Drop columns with >30% missing
thresh = 0.30
high_missing = [c for c in X.columns if X[c].isnull().mean() > thresh]
if high_missing:
    print(f"Dropping columns with >30% missing: {high_missing}")
    X.drop(columns=high_missing, inplace=True)
else:
    print("No columns exceed 30% missing threshold.")

# Impute remaining missing with column median
for col in X.columns:
    if X[col].isnull().any():
        med = X[col].median()
        X[col].fillna(med, inplace=True)
        print(f"  Imputed '{col}' with median={med:.4f}")

print(f"\nAfter imputation — any missing: {X.isnull().any().any()}")

# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5 — FEATURE ENGINEERING (8 derived ratios)")
print("=" * 60)

ASSUMED_RATE = 0.05   # 5% assumed interest rate for proxy

def safe_div(a, b, fallback=0.0):
    """Division with zero-guard."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fallback)
    return result

# 1. Interest Coverage Proxy  — Campbell et al. (2008)
if "EBIT T12M" in X.columns and "Total Liab LF" in X.columns:
    X["Interest_Coverage_Proxy"] = safe_div(
        X["EBIT T12M"].values,
        (X["Total Liab LF"] * ASSUMED_RATE).values
    )
    print("  ✓ Interest_Coverage_Proxy = EBIT / (Total Liab * 0.05)")

# 2. Asset Utilisation — Altman (1968)
if "Revenue T12M" in X.columns and "Tot Assets LF" in X.columns:
    X["Asset_Utilisation"] = safe_div(
        X["Revenue T12M"].values, X["Tot Assets LF"].values
    )
    print("  ✓ Asset_Utilisation = Revenue / Total Assets")

# 3. Earnings Quality — Khandani et al. (2010)
if "FCF T12M" in X.columns and "EBITDA T12M" in X.columns:
    X["Earnings_Quality"] = safe_div(
        X["FCF T12M"].values, X["EBITDA T12M"].values
    )
    print("  ✓ Earnings_Quality = FCF / EBITDA")

# 4. Market-to-Book Proxy — Campbell et al. (2008)
if "Market Cap" in X.columns and "Tot Assets LF" in X.columns:
    X["Market_to_Book"] = safe_div(
        X["Market Cap"].values, X["Tot Assets LF"].values
    )
    print("  ✓ Market_to_Book = Market Cap / Total Assets")

# 5. Profitability Ratio (ROA) — Ohlson (1980)
if "EBIT T12M" in X.columns and "Tot Assets LF" in X.columns:
    X["Profitability_ROA"] = safe_div(
        X["EBIT T12M"].values, X["Tot Assets LF"].values
    )
    print("  ✓ Profitability_ROA = EBIT / Total Assets")

# 6. Liquidity Spread — Hand & Henley (1997)
if "Curr Ratio LF" in X.columns and "Quick Ratio LF" in X.columns:
    X["Liquidity_Spread"] = X["Curr Ratio LF"] - X["Quick Ratio LF"]
    print("  ✓ Liquidity_Spread = Current Ratio - Quick Ratio")

# 7. Momentum Signal — Bharath & Shumway (2008)
if "Total Return:Y-1" in X.columns and "Volat:D-30" in X.columns:
    X["Momentum_Signal"] = X["Total Return:Y-1"] - X["Volat:D-30"]
    print("  ✓ Momentum_Signal = Total Return Y-1 - Volatility D-30")

# 8. Debt Burden Ratio
if "Total Liab LF" in X.columns and "Tot Assets LF" in X.columns:
    X["Debt_Burden"] = safe_div(
        X["Total Liab LF"].values, X["Tot Assets LF"].values
    )
    print("  ✓ Debt_Burden = Total Liabilities / Total Assets")

print(f"\nTotal features after engineering: {X.shape[1]}")
print(f"Feature list: {X.columns.tolist()}")

# =============================================================================
# 6. WINSORIZATION (1st–99th percentile)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6 — WINSORIZATION (1st–99th percentile)")
print("=" * 60)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    lower = X[col].quantile(0.01)
    upper = X[col].quantile(0.99)
    X[col] = X[col].clip(lower=lower, upper=upper)

print(f"Winsorized {len(numeric_cols)} numeric columns.")

# =============================================================================
# 7. TRAIN / TEST SPLIT (80/20, stratified)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7 — TRAIN/TEST SPLIT (80/20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train defaults: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"Test  defaults: {y_test.sum()}  ({y_test.mean()*100:.1f}%)")

# =============================================================================
# 8. FEATURE SCALING (StandardScaler — fit on train, transform both)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8 — STANDARDSCALER (fit on train only)")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
print("Scaling complete.")

# =============================================================================
# 9. CLASS IMBALANCE HANDLING (SMOTE+ENN on training set only)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9 — CLASS IMBALANCE (SMOTE+ENN)")
print("=" * 60)

if SMOTE_AVAILABLE:
    smote_enn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smote_enn.fit_resample(X_train_scaled, y_train)
    X_train_res = pd.DataFrame(X_train_res, columns=X_train_scaled.columns)
    y_train_res = pd.Series(y_train_res, name="Default")
    print(f"After SMOTE+ENN — Train shape: {X_train_res.shape}")
    print(f"  Default (1): {y_train_res.sum()}  Safe (0): {(y_train_res==0).sum()}")
else:
    # Fall back: use scaled train set and rely on class_weight='balanced' in models
    X_train_res = X_train_scaled.copy()
    y_train_res = y_train.copy()
    print("SMOTE+ENN skipped — using class_weight='balanced' in models instead.")

# =============================================================================
# 10. SAVE PROCESSED DATA
# =============================================================================
print("\n" + "=" * 60)
# ── helper function to save with retry for open files ───────────────
import time
def safe_save_csv(df, filepath, **kwargs):
    for attempt in range(5):
        try:
            df.to_csv(filepath, **kwargs)
            return True
        except PermissionError:
            print(f"  [WARN] {filepath} is open in another program. Retrying in 3s... ({attempt+1}/5)")
            time.sleep(3)
    print(f"  [ERROR] Could not save {filepath}. Please close the file and retry.")
    return False

# ── model-ready feature sets ──────────────────────────────────────────────
safe_save_csv(X_train_res, f"{OUT_DIR}/X_train.csv", index=False)
safe_save_csv(y_train_res, f"{OUT_DIR}/y_train.csv", index=False)
safe_save_csv(X_test_scaled, f"{OUT_DIR}/X_test.csv", index=False)
safe_save_csv(y_test, f"{OUT_DIR}/y_test.csv", index=False)

# ── unscaled versions for tree models ─────────────────────────────────────
safe_save_csv(X_train, f"{OUT_DIR}/X_train_raw.csv", index=False)
safe_save_csv(y_train, f"{OUT_DIR}/y_train_raw.csv", index=False)
safe_save_csv(X_test, f"{OUT_DIR}/X_test_raw.csv", index=False)

# ── AZS metadata for the ENTIRE dataset (used to attach to predictions) ───
# Ensure df_meta index exactly aligns with X and y before any resets
df_meta["AZS_Zone"] = df_meta["AZS_Zone"].astype(str)
safe_save_csv(df_meta, f"{OUT_DIR}/az_scores_all.csv", index=True, index_label="row_id")

# Train/Test split was done on X and y. y_test.index contains the exact indices.
test_idx = y_test.index
az_test = df_meta.loc[test_idx].copy()
safe_save_csv(az_test, f"{OUT_DIR}/az_scores_test.csv", index=False)

print(f"\nSaved AZS metadata for all {len(df_meta)} companies  → {OUT_DIR}/az_scores_all.csv")
print(f"Saved AZS metadata for test {len(az_test)} companies → {OUT_DIR}/az_scores_test.csv")

# ── feature list ──────────────────────────────────────────────────────────
import json
with open(f"{OUT_DIR}/feature_list.json", "w") as f:
    json.dump(X.columns.tolist(), f, indent=2)

print(f"\nSaved to '{OUT_DIR}/' folder:")
print(f"  X_train.csv / y_train.csv      (scaled + SMOTE resampled)")
print(f"  X_test.csv  / y_test.csv        (scaled, NO SMOTE)")
print(f"  X_train_raw.csv / y_train_raw.csv / X_test_raw.csv (unscaled + raw labels)")
print(f"  az_scores_all.csv               (Ticker | Name | AZS | Zone | Default)")
print(f"  az_scores_test.csv              (same, test rows only)")
print(f"  feature_list.json")
print("\n✅ PHASE 1 PREPROCESSING COMPLETE — Run credit_risk_phase2_modeling.py next")

print(f"\nSaved to '{OUT_DIR}/' folder:")
print(f"  X_train.csv / y_train.csv  (scaled + SMOTE resampled)")
print(f"  X_test.csv  / y_test.csv   (scaled, NO SMOTE)")
print(f"  X_train_raw.csv / y_train_raw.csv / X_test_raw.csv  (unscaled + raw labels)")
print(f"  feature_list.json")
print("\n✅ PHASE 1 PREPROCESSING COMPLETE — Run credit_risk_phase2_modeling.py next")
