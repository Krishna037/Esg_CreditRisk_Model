"""
=============================================================================
CREDIT RISK MODEL — PHASE 3: ESG AUGMENTATION
=============================================================================
Adds Phase 2 ESG features:
  - % Women on Bd:Y     (Governance)
  - GHG Scope 1:Y       (Environment — direct emissions)
  - GHG Scope 3:Y       (Environment — value chain)
  - CO2 Scope 1:Y       (Environment — carbon footprint)

Compares Phase 1 (financial only) vs Phase 2 (financial + ESG) performance.

Run AFTER: credit_risk_phase1_preprocessing.py & credit_risk_phase2_modeling.py
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score,
                              precision_score, recall_score, brier_score_loss,
                             )
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier; XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from imblearn.combine import SMOTEENN; SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ESG_COLS = ["% Women on Bd:Y", "GHG Scope 1:Y", "GHG Scope 3:Y", "CO2 Scope 1:Y"]

def clean_dataframe(df):
    """Convert space-strings → NaN and coerce object columns → numeric."""
    ID_COLS = {"Ticker", "Name"}
    for col in df.columns:
        if col in ID_COLS:
            continue
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# =============================================================================
# 1. RELOAD RAW DATA & REBUILD WITH ESG FEATURES
# =============================================================================
print("=" * 60)
print("PHASE 3 — ESG AUGMENTATION")
print("=" * 60)

df = pd.read_excel("Bloomberg Data.xlsx")
df = clean_dataframe(df)    # convert space-strings → NaN, objects → numeric

# Drop rows where AZS is missing
df = df.dropna(subset=["AZS"])

# Binary target
df["Default"] = (df["AZS"] < 1.8).astype(int)
df = df.drop(columns=["AZS", "Ticker", "Name"], errors="ignore")

y = df["Default"]
X_all = df.drop(columns=["Default"])

# Check which ESG columns exist
esg_present = [c for c in ESG_COLS if c in X_all.columns]
print(f"\nESG columns found: {esg_present}")
missing_esg = [c for c in ESG_COLS if c not in X_all.columns]
if missing_esg:
    print(f"[WARN] Missing ESG columns: {missing_esg}")

# =============================================================================
# 2. BUILD TWO FEATURE SETS
# =============================================================================
# Feature engineering helper
def engineer_features(X):
    X = X.copy()
    ASSUMED_RATE = 0.05
    def sd(a, b, fb=0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b != 0, a / b, fb)

    if "EBIT T12M" in X.columns and "Total Liab LF" in X.columns:
        X["Interest_Coverage_Proxy"] = sd(X["EBIT T12M"].values, (X["Total Liab LF"]*ASSUMED_RATE).values)
    if "Revenue T12M" in X.columns and "Tot Assets LF" in X.columns:
        X["Asset_Utilisation"] = sd(X["Revenue T12M"].values, X["Tot Assets LF"].values)
    if "FCF T12M" in X.columns and "EBITDA T12M" in X.columns:
        X["Earnings_Quality"] = sd(X["FCF T12M"].values, X["EBITDA T12M"].values)
    if "Market Cap" in X.columns and "Tot Assets LF" in X.columns:
        X["Market_to_Book"] = sd(X["Market Cap"].values, X["Tot Assets LF"].values)
    if "EBIT T12M" in X.columns and "Tot Assets LF" in X.columns:
        X["Profitability_ROA"] = sd(X["EBIT T12M"].values, X["Tot Assets LF"].values)
    if "Curr Ratio LF" in X.columns and "Quick Ratio LF" in X.columns:
        X["Liquidity_Spread"] = X["Curr Ratio LF"] - X["Quick Ratio LF"]
    if "Total Return:Y-1" in X.columns and "Volat:D-30" in X.columns:
        X["Momentum_Signal"] = X["Total Return:Y-1"] - X["Volat:D-30"]
    if "Total Liab LF" in X.columns and "Tot Assets LF" in X.columns:
        X["Debt_Burden"] = sd(X["Total Liab LF"].values, X["Tot Assets LF"].values)
    return X

def preprocess(X, y, esg=False):
    """
    Full preprocessing pipeline with CORRECT train-test ordering to prevent leakage.
    Returns X_tr, X_te, y_tr, y_te (scaled).
    
    KEY FIX: train-test split happens BEFORE imputation/winsorization to ensure
    statistics (median, quantiles) are computed on train data only.
    """
    if not esg:
        X = X.drop(columns=esg_present, errors="ignore")

    # Feature engineering
    X = engineer_features(X)

    # =========================
    # CRITICAL: SPLIT FIRST
    # =========================
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                                random_state=42, stratify=y)

    # Drop >30% missing (compute on train only)
    high = [c for c in X_tr.columns if X_tr[c].isnull().mean() > 0.30]
    X_tr = X_tr.drop(columns=high, errors="ignore")
    X_te = X_te.drop(columns=high, errors="ignore")

    # Median imputation (fit on train only, apply to both)
    train_medians = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(train_medians)
    X_te = X_te.fillna(train_medians)

    # Winsorization (fit bounds on train only, apply to both)
    for col in X_tr.select_dtypes(include=[np.number]).columns:
        lo = X_tr[col].quantile(0.01)
        hi = X_tr[col].quantile(0.99)
        X_tr[col] = X_tr[col].clip(lower=lo, upper=hi)
        X_te[col] = X_te[col].clip(lower=lo, upper=hi)

    # Scale (fit on train only, apply to both)
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_te_s = pd.DataFrame(scaler.transform(X_te),   columns=X_te.columns)

    import re
    def clean_cols(cols):
        return [re.sub(r'[^\w\s]', '_', c).replace(' ', '_') for c in cols]
    
    X_tr_s.columns = clean_cols(X_tr_s.columns)
    X_te_s.columns = clean_cols(X_te_s.columns)

    # SMOTE (on train only)
    if SMOTE_OK:
        from imblearn.combine import SMOTEENN
        sm = SMOTEENN(random_state=42)
        X_tr_r, y_tr_r = sm.fit_resample(X_tr_s, y_tr)
        X_tr_r = pd.DataFrame(X_tr_r, columns=X_tr_s.columns)
        y_tr_r = pd.Series(y_tr_r, name="Default")
    else:
        X_tr_r, y_tr_r = X_tr_s, y_tr

    return X_tr_r, X_te_s, y_tr_r, y_te

def quick_eval(name, model, X_te, y_te):
    yp = model.predict_proba(X_te)[:, 1]
    yc = (yp >= 0.5).astype(int)
    auc = roc_auc_score(y_te, yp)
    ks  = ks_2samp(yp[y_te==1], yp[y_te==0]).statistic
    f1  = f1_score(y_te, yc, zero_division=0)
    brier = brier_score_loss(y_te, yp)
    print(f"  {name:35s}  AUC={auc:.4f}  KS={ks:.4f}  F1={f1:.4f}  Brier={brier:.4f}")
    return {"Model": name, "AUC": auc, "KS": ks, "F1": f1, "Brier": brier, "y_prob": yp}

# =============================================================================
# 3. TRAIN & COMPARE (Financial Only vs Financial + ESG)
# =============================================================================
comparison = []

for include_esg in [False, True]:
    label = "Phase 2 (+ ESG)" if include_esg else "Phase 1 (Financial)"
    print(f"\n── {label} ──")

    X_tr, X_te, y_tr, y_te = preprocess(X_all.copy(), y, esg=include_esg)
    neg, pos = (y_tr==0).sum(), (y_tr==1).sum()
    spw = neg/pos if pos>0 else 1

    print(f"  Features: {X_tr.shape[1]}  |  Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")

    # XGBoost (primary model)
    if XGB_OK:
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        r = quick_eval(f"XGBoost  [{label}]", model, X_te, y_te)
        r["Phase"] = label
        comparison.append(r)
        
        # Save ESG model and predictions if this is the ESG phase
        if include_esg:
            import joblib
            model_path = f"{OUT_DIR}/models/xgboost_esg_augmented.joblib"
            joblib.dump(model, model_path)
            
            # Save predictions
            az_test = pd.read_csv(f"{OUT_DIR}/az_scores_test.csv")
            y_prob = model.predict_proba(X_te)[:, 1]
            az_test["Predicted_Default_ESG"] = (y_prob >= 0.5).astype(int)
            az_test["P_Default_ESG"] = y_prob.round(6)
            az_test.to_csv(f"{OUT_DIR}/predictions/xgboost_esg_predictions.csv", index=False)
            print(f"  💾 Saved ESG model → {model_path}")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=7, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    r2 = quick_eval(f"RandomForest [{label}]", rf, X_te, y_te)
    r2["Phase"] = label
    comparison.append(r2)

# =============================================================================
# 4. PLOT: PHASE 1 vs PHASE 2 COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1 vs PHASE 2 COMPARISON")
print("=" * 60)

comp_df = pd.DataFrame([{k:v for k,v in r.items() if k != "y_prob"}
                         for r in comparison]).set_index("Model").round(4)
print(comp_df.to_string())
comp_df.to_csv(f"{OUT_DIR}/phase_comparison.csv")

# Bar chart: AUC improvement
fig, ax = plt.subplots(figsize=(10, 5))
models_esg = [(r["Model"].replace("[Phase 1 (Financial)]","").replace("[Phase 2 (+ ESG)]","").strip(),
               r["Phase"], r["AUC"]) for r in comparison]
bar_labels = [f"{m}\n{p}" for m, p, _ in models_esg]
bar_vals   = [v for _, _, v in models_esg]
bar_colors = ["#4e79a7" if "Phase 1" in p else "#e15759" for _, p, _ in models_esg]
bars = ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="black")
ax.set_ylim(0, 1.05)
ax.set_ylabel("AUC-ROC")
ax.set_title("Phase 1 (Financial Only) vs Phase 2 (+ ESG) — AUC Comparison")
for bar, v in zip(bars, bar_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4e79a7", label="Phase 1 (Financial)"),
                   Patch(color="#e15759", label="Phase 2 (+ ESG)")],
          loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/13_phase1_vs_phase2_auc.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR}/13_phase1_vs_phase2_auc.png")

# ESG feature distributions (for context)
fig, axes = plt.subplots(1, min(len(esg_present), 4), figsize=(14, 4))
if len(esg_present) == 1:
    axes = [axes]
for ax, col in zip(axes, esg_present):
    data = df_raw_esg = pd.read_excel("Bloomberg Data.xlsx")[col].dropna()
    ax.hist(data, bins=30, color="#9467bd", edgecolor="black", alpha=0.8)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("Value")
plt.suptitle("ESG Feature Distributions", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/14_esg_distributions.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/14_esg_distributions.png")

print("\n🎉 PHASE 3 ESG AUGMENTATION COMPLETE")
print("=" * 60)
print("\nAll output files in ./outputs/:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(f"{OUT_DIR}/{f}")
    print(f"  {f:50s}  {size:>8,d} bytes")
