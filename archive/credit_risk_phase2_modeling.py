"""
=============================================================================
CREDIT RISK MODEL — PHASE 2 & 3: MODELING, EVALUATION & EXPLAINABILITY
=============================================================================
Models trained (each saved to outputs/models/):
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost              (primary recommended)
  4. LightGBM
  5. Stacking Ensemble    (best final model)

Per-model outputs (outputs/model_summaries/ & outputs/predictions/):
  - <model>.joblib          — saved model object
  - <model>_summary.json    — metrics + feature importance / coefficients
  - <model>_predictions.csv — Ticker | Name | AZS | Zone | Default | P(Default)

Evaluation outputs (outputs/):
  - model_comparison.csv    — all metrics side-by-side
  - ROC curves, metrics bar chart

SHAP outputs (outputs/shap/):
  - shap_beeswarm.png, shap_bar.png, shap_waterfall.png, shap_dependence.png

LIME output (outputs/shap/lime_explanation.png)

Run AFTER: credit_risk_phase1_preprocessing.py
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score, brier_score_loss,
                             average_precision_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold, cross_validate
from scipy.stats import ks_2samp

try:
    from xgboost import XGBClassifier;   XGB_OK = True
except ImportError:
    print("[WARN] xgboost not found. pip install xgboost"); XGB_OK = False

try:
    from lightgbm import LGBMClassifier; LGB_OK = True
except ImportError:
    print("[WARN] lightgbm not found. pip install lightgbm"); LGB_OK = False

try:
    import shap;                          SHAP_OK = True
except ImportError:
    print("[WARN] shap not found. pip install shap"); SHAP_OK = False

try:
    import lime, lime.lime_tabular;      LIME_OK = True
except ImportError:
    print("[WARN] lime not found. pip install lime"); LIME_OK = False

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
OUT_DIR   = "outputs"
MODEL_DIR = os.path.join(OUT_DIR, "models")        # saved .joblib files
SUMM_DIR  = os.path.join(OUT_DIR, "model_summaries")  # per-model JSON summaries
PRED_DIR  = os.path.join(OUT_DIR, "predictions")   # per-model prediction CSVs
SHAP_DIR  = os.path.join(OUT_DIR, "shap")          # SHAP/LIME plots
for d in [MODEL_DIR, SUMM_DIR, PRED_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# LOAD PREPROCESSED DATA
# =============================================================================
print("=" * 65)
print("LOADING PREPROCESSED DATA")
print("=" * 65)

X_train     = pd.read_csv(f"{OUT_DIR}/X_train.csv")
y_train     = pd.read_csv(f"{OUT_DIR}/y_train.csv").squeeze()
X_test      = pd.read_csv(f"{OUT_DIR}/X_test.csv")
y_test      = pd.read_csv(f"{OUT_DIR}/y_test.csv").squeeze()
X_train_raw = pd.read_csv(f"{OUT_DIR}/X_train_raw.csv")
X_test_raw  = pd.read_csv(f"{OUT_DIR}/X_test_raw.csv")

try:
    y_train_raw = pd.read_csv(f"{OUT_DIR}/y_train_raw.csv").squeeze()
except FileNotFoundError:
    y_train_raw = y_train.copy()
    print("[WARN] y_train_raw.csv not found. Using y_train.csv as fallback.")

# AZS metadata (Ticker, Name, AZS, Zone, Default) for test companies
az_test = pd.read_csv(f"{OUT_DIR}/az_scores_test.csv")

# LightGBM requires JSON-safe feature names (no ':', '%', ' ', '[', ']')
import re
def clean_cols(cols):
    return [re.sub(r'[^\w\s]', '_', c).replace(' ', '_') for c in cols]

X_train.columns = clean_cols(X_train.columns)
X_test.columns = clean_cols(X_test.columns)
X_train_raw.columns = clean_cols(X_train_raw.columns)
X_test_raw.columns = clean_cols(X_test_raw.columns)

with open(f"{OUT_DIR}/feature_list.json") as f:
    features = clean_cols(json.load(f))

print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  AZS test metadata: {az_test.shape}")
print(f"  Default rate (test): {y_test.mean()*100:.1f}%\n")

if len(X_train) != len(y_train):
    raise ValueError(f"Length mismatch: X_train={len(X_train)} but y_train={len(y_train)}")
if len(X_train_raw) != len(y_train_raw):
    print(f"[WARN] Length mismatch in raw set (X_train_raw={len(X_train_raw)}, y_train_raw={len(y_train_raw)}).")
    print("[WARN] Raw feature set will be ignored for model fitting to avoid label misalignment.")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pw = round(float(neg / pos), 4) if pos > 0 else 1.0
print(f"  XGBoost scale_pos_weight: {scale_pw}")

RUN_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Features that are near-proxies of Altman Z-score inputs can leak target signal
# because target Default is directly defined from AZS < 1.8.
KNOWN_AZS_PROXY_FEATURES = {
    "Asset_Utilisation",
    "Profitability_ROA",
    "Interest_Coverage_Proxy",
    "Debt_Burden",
    "Market_to_Book",
}


def prune_leakage_features(X_tr, X_te, y_tr):
    """Drop known AZS proxy features and highly suspicious single-feature separators."""
    drop_cols = set(c for c in X_tr.columns if c in KNOWN_AZS_PROXY_FEATURES)

    suspicious = []
    for col in X_tr.columns:
        vals = X_tr[col]
        if vals.nunique(dropna=True) < 3:
            continue
        try:
            single_auc = roc_auc_score(y_tr, vals)
            single_auc = max(single_auc, 1 - single_auc)
            if single_auc >= 0.98:
                suspicious.append(col)
        except Exception:
            continue

    drop_cols.update(suspicious)
    drop_cols = sorted(drop_cols)

    if drop_cols:
        print(f"\n[Leakage Guard] Dropping {len(drop_cols)} features: {drop_cols}")
        X_tr = X_tr.drop(columns=drop_cols, errors="ignore")
        X_te = X_te.drop(columns=drop_cols, errors="ignore")
    else:
        print("\n[Leakage Guard] No suspicious leakage features detected.")

    return X_tr, X_te, drop_cols


def run_cv_summary(model, X_tr, y_tr, model_name):
    """5-fold stratified CV on train data for robust generalization estimate."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    cv_out = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)

    summary = {
        "cv_auc_mean": float(np.mean(cv_out["test_auc"])),
        "cv_auc_std": float(np.std(cv_out["test_auc"])),
        "cv_f1_mean": float(np.mean(cv_out["test_f1"])),
        "cv_precision_mean": float(np.mean(cv_out["test_precision"])),
        "cv_recall_mean": float(np.mean(cv_out["test_recall"])),
    }
    print(
        f"  CV(5) {model_name}: "
        f"AUC={summary['cv_auc_mean']:.4f}±{summary['cv_auc_std']:.4f}  "
        f"F1={summary['cv_f1_mean']:.4f}"
    )
    return summary

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
PALETTE = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b", "#9467bd"]

def save_model(model, name_slug):
    """Save model as joblib file; return path."""
    path = os.path.join(MODEL_DIR, f"{name_slug}.joblib")
    joblib.dump(model, path)
    print(f"  💾 Model saved → {path}")
    return path


def save_summary(summary_dict, name_slug):
    """Save a model summary JSON file."""
    path = os.path.join(SUMM_DIR, f"{name_slug}_summary.json")
    with open(path, "w") as f:
        json.dump(summary_dict, f, indent=2, default=str)
    print(f"  📋 Summary saved → {path}")


def save_predictions(model, X_te, y_te, az_df, name_slug, name_label):
    """Save per-row predictions with AZS scores attached."""
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pred_df = az_df.reset_index(drop=True).copy()
    pred_df["Predicted_Default"]     = y_pred
    pred_df["P_Default"]             = y_prob.round(6)
    pred_df["Correct"]               = (y_pred == y_te.values).astype(int)
    pred_df["Model"]                 = name_label

    path = os.path.join(PRED_DIR, f"{name_slug}_predictions.csv")
    pred_df.to_csv(path, index=False)
    print(f"  📄 Predictions saved → {path}")
    return y_prob, pred_df


def evaluate(name_label, name_slug, model, X_tr, y_tr, X_te, y_te, az_df, color="#4e79a7", cv_summary=None):
    """Full evaluation: metrics, model save, summary, predictions."""
    print(f"\n{'─'*65}")
    print(f"  EVALUATING: {name_label}")
    print(f"{'─'*65}")

    # ── Save model ──────────────────────────────────────────────────────
    model_path = save_model(model, name_slug)

    # ── Predictions & probabilities ─────────────────────────────────────
    y_prob, pred_df = save_predictions(model, X_te, y_te, az_df, name_slug, name_label)
    y_pred = (y_prob >= 0.5).astype(int)
    y_prob_train = model.predict_proba(X_tr)[:, 1]

    # ── Metrics ────────────────────────────────────────────────────────
    auc_tr = roc_auc_score(y_tr, y_prob_train)
    auc   = roc_auc_score(y_te, y_prob)
    gini  = 2 * auc - 1
    brier = brier_score_loss(y_te, y_prob)
    pr_auc = average_precision_score(y_te, y_prob)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    prec  = precision_score(y_te, y_pred, zero_division=0)
    rec   = recall_score(y_te, y_pred, zero_division=0)
    f1    = f1_score(y_te, y_pred, zero_division=0)
    overfit_gap = auc_tr - auc
    overfit_flag = overfit_gap > 0.10

    defaulters  = y_prob[y_te == 1]
    non_default = y_prob[y_te == 0]
    ks = ks_2samp(defaulters, non_default).statistic if len(defaulters) > 0 else 0.0

    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    print(f"\n  Train AUC : {auc_tr:.4f}")
    print(f"  Test AUC  : {auc:.4f}    Gini: {gini:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}    Balanced Acc: {bal_acc:.4f}")
    print(f"  Overfit Gap (Train-Test AUC): {overfit_gap:.4f}  |  Flag={overfit_flag}")
    print(f"  KS Stat   : {ks:.4f}")
    print(f"  Precision : {prec:.4f}    Recall: {rec:.4f}    F1: {f1:.4f}")
    print(f"  Brier     : {brier:.4f}")
    print(f"  Confusion Matrix  →  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(classification_report(y_te, y_pred, target_names=["Safe","Default"],
                                zero_division=0))

    # ── Feature importance / coefficients ──────────────────────────────
    feat_imp = {}
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        feat_imp = dict(zip(X_te.columns, [round(float(c), 6) for c in coef]))
    elif hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feat_imp = dict(zip(X_te.columns, [round(float(v), 6) for v in fi]))
    elif hasattr(model, "final_estimator_") and hasattr(model.final_estimator_, "coef_"):
        coef = model.final_estimator_.coef_[0]
        # stacking uses transform_output of base models; use generic index
        feat_imp = {f"meta_feat_{i}": round(float(c), 6)
                    for i, c in enumerate(coef)}

    # ── Summary JSON ───────────────────────────────────────────────────
    summary = {
        "model_name":        name_label,
        "model_slug":        name_slug,
        "model_file":        model_path,
        "run_timestamp":     RUN_TIME,
        "test_size":         int(len(y_te)),
        "metrics": {
            "Train_AUC":     round(auc_tr, 6),
            "AUC_ROC":       round(auc, 6),
            "Gini":          round(gini, 6),
            "KS_Statistic":  round(ks, 6),
            "PR_AUC":        round(pr_auc, 6),
            "Balanced_Accuracy": round(bal_acc, 6),
            "Precision":     round(prec, 6),
            "Recall":        round(rec, 6),
            "F1_Score":      round(f1, 6),
            "Brier_Score":   round(brier, 6),
            "Overfit_AUC_Gap": round(overfit_gap, 6),
            "Overfit_Flag":  bool(overfit_flag),
        },
        "cross_validation": cv_summary or {},
        "confusion_matrix":  {"TN": int(tn), "FP": int(fp),
                              "FN": int(fn), "TP": int(tp)},
        "feature_importance_or_coefficients": dict(
            sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
        ),
        "az_score_range_in_test": {
            "min": round(float(az_df["AZS"].min()), 4) if "AZS" in az_df.columns else None,
            "max": round(float(az_df["AZS"].max()), 4) if "AZS" in az_df.columns else None,
        },
        "highest_risk_companies": (
            pred_df.sort_values("P_Default", ascending=False)
            [["Ticker", "Name", "AZS", "AZS_Zone", "P_Default", "Default"]]
            .head(5).to_dict(orient="records")
            if "Ticker" in pred_df.columns else []
        )
    }
    save_summary(summary, name_slug)

    return {
        "Name": name_label, "Slug": name_slug,
        "AUC": auc, "Train_AUC": auc_tr, "Overfit_Gap": overfit_gap,
        "PR_AUC": pr_auc, "Balanced_Acc": bal_acc,
        "Gini": gini, "KS": ks,
        "Precision": prec, "Recall": rec, "F1": f1, "Brier": brier,
        "y_prob": y_prob, "color": color
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
X_train_model, X_test_model, dropped_leakage_cols = prune_leakage_features(
    X_train.copy(), X_test.copy(), y_train
)
print(f"Modeling features after leakage guard: {X_train_model.shape[1]}")

print("\n" + "=" * 65)
print("MODEL 1 — LOGISTIC REGRESSION (Baseline)")
print("=" * 65)

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=2500,
    random_state=42,
    C=0.2,
    penalty="elasticnet",
    l1_ratio=0.2,
    solver="saga",
)
lr_cv = run_cv_summary(lr, X_train_model, y_train, "Logistic Regression")
lr.fit(X_train_model, y_train)
lr_res = evaluate("Logistic Regression", "logistic_regression",
                  lr, X_train_model, y_train, X_test_model, y_test, az_test, PALETTE[0], lr_cv)

# Coefficient plot
coef_df = pd.DataFrame({"Feature": X_test_model.columns,
                         "Coefficient": lr.coef_[0]})
coef_df = coef_df.sort_values("Coefficient")
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#e15759" if c > 0 else "#4e79a7" for c in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression — Coefficients\n(Red=↑default risk, Blue=↓default risk)", fontsize=12)
ax.set_xlabel("Coefficient")
plt.tight_layout()
plt.savefig(f"{SUMM_DIR}/logistic_regression_coefficients.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("MODEL 2 — RANDOM FOREST")
print("=" * 65)

rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=10,
    min_samples_leaf=3,
    min_samples_split=8,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_cv = run_cv_summary(rf, X_train_model, y_train, "Random Forest")
rf.fit(X_train_model, y_train)
rf_res = evaluate("Random Forest", "random_forest",
                  rf, X_train_model, y_train, X_test_model, y_test, az_test, PALETTE[1], rf_cv)

# Feature importance plot
fi_df = pd.DataFrame({"Feature": X_test_model.columns,
                       "Importance": rf.feature_importances_})
fi_df = fi_df.sort_values("Importance", ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(fi_df["Feature"], fi_df["Importance"],
        color="#76b7b2", edgecolor="black")
ax.set_title("Random Forest — Feature Importances (MDI)", fontsize=12)
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig(f"{SUMM_DIR}/random_forest_feature_importance.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — XGBOOST
# ══════════════════════════════════════════════════════════════════════════════
results = [lr_res, rf_res]

if XGB_OK:
    print("\n" + "=" * 65)
    print("MODEL 3 — XGBOOST (Primary Recommended)")
    print("=" * 65)

    xgb = XGBClassifier(
        learning_rate=0.03,
        n_estimators=700,
        max_depth=4,
        min_child_weight=3,
        gamma=0.10,
        reg_alpha=0.20,
        reg_lambda=2.0,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pw,
        eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb_cv = run_cv_summary(xgb, X_train_model, y_train, "XGBoost")
    xgb.fit(X_train_model, y_train)
    xgb_res = evaluate("XGBoost", "xgboost",
                        xgb, X_train_model, y_train, X_test_model, y_test, az_test, PALETTE[2], xgb_cv)
    results.append(xgb_res)

    # XGBoost feature importance plot
    fi_xgb = pd.DataFrame({"Feature": X_test_model.columns,
                             "Importance": xgb.feature_importances_})
    fi_df = fi_df.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fi_xgb["Feature"], fi_xgb["Importance"],
            color="#e15759", edgecolor="black")
    ax.set_title("XGBoost — Feature Importances (F-score)", fontsize=12)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{SUMM_DIR}/xgboost_feature_importance.png", dpi=150)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════
if LGB_OK:
    print("\n" + "=" * 65)
    print("MODEL 4 — LightGBM")
    print("=" * 65)

    lgb = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=25,
        reg_alpha=0.2,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_cv = run_cv_summary(lgb, X_train_model, y_train, "LightGBM")
    lgb.fit(X_train_model, y_train)
    lgb_res = evaluate("LightGBM", "lightgbm",
                        lgb, X_train_model, y_train, X_test_model, y_test, az_test, PALETTE[3], lgb_cv)
    results.append(lgb_res)

    fi_lgb = pd.DataFrame({"Feature": X_test_model.columns,
                             "Importance": lgb.feature_importances_})
    fi_lgb = fi_lgb.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fi_lgb["Feature"], fi_lgb["Importance"],
            color="#f28e2b", edgecolor="black")
    ax.set_title("LightGBM — Feature Importances", fontsize=12)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{SUMM_DIR}/lightgbm_feature_importance.png", dpi=150)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("MODEL 5 — STACKING ENSEMBLE (Best Final Model)")
print("=" * 65)

base_estimators = [
    ("lr",  LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1)),
    ("rf",  RandomForestClassifier(n_estimators=200, max_depth=7,
                                    class_weight="balanced", random_state=42))
]
if XGB_OK:
    base_estimators.append(("xgb", XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pw,
        eval_metric="logloss", random_state=42, n_jobs=-1)))
if LGB_OK:
    base_estimators.append(("lgb", LGBMClassifier(
        n_estimators=200, num_leaves=31, class_weight="balanced",
        random_state=42, verbose=-1, n_jobs=-1)))

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
    cv=5, passthrough=False, n_jobs=-1
)

stack_cv = run_cv_summary(stack, X_train_model, y_train, "Stacking Ensemble")
stack.fit(X_train_model, y_train)
stack_res = evaluate("Stacking Ensemble", "stacking_ensemble",
                      stack, X_train_model, y_train, X_test_model, y_test, az_test, PALETTE[4], stack_cv)
results.append(stack_res)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE & PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

metrics_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("y_prob", "color", "Slug")}
    for r in results
]).set_index("Name").round(4)
print(metrics_df.to_string())
metrics_df.to_csv(f"{OUT_DIR}/model_comparison.csv")
print(f"\nSaved → {OUT_DIR}/model_comparison.csv")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for r in results:
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    ax.plot(fpr, tpr, label=f"{r['Name']} (AUC={r['AUC']:.3f})",
            linewidth=2, color=r["color"])
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models"); ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
plt.close()
print(f"Saved → {OUT_DIR}/roc_curves.png")

# Metrics bar chart
METRIC_COLS = ["AUC", "Train_AUC", "PR_AUC", "Balanced_Acc", "F1", "Brier"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, metric in zip(axes.flat, METRIC_COLS):
    vals = metrics_df[metric]
    bars = ax.bar(vals.index, vals.values,
                  color=[r["color"] for r in results], edgecolor="black")
    ax.set_title(metric, fontsize=12)
    if metric == "Brier":
        ax.set_ylim(0, max(0.35, float(vals.max()) + 0.05))
    else:
        ax.set_ylim(0, 1.0)
    ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.suptitle("Model Comparison — Key Metrics", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/metrics_comparison.png", dpi=150)
plt.close()
print(f"Saved → {OUT_DIR}/metrics_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
if SHAP_OK:
    print("\n" + "=" * 65)
    print("SHAP EXPLAINABILITY (XGBoost)")
    print("=" * 65)

    best_model  = xgb if XGB_OK else stack
    best_slug   = "xgboost" if XGB_OK else "stacking_ensemble"
    X_explain   = X_test_model

    explainer    = shap.TreeExplainer(best_model)
    shap_values  = explainer(X_explain)

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, plot_type="dot",
                      show=False, max_display=20)
    plt.title("SHAP Summary (Beeswarm) — XGBoost", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {SHAP_DIR}/shap_beeswarm.png")

    # Bar
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_explain, plot_type="bar",
                      show=False, max_display=20)
    plt.title("SHAP Feature Importance (Bar) — XGBoost", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {SHAP_DIR}/shap_bar.png")

    # Waterfall — highest-risk company
    probs = best_model.predict_proba(X_explain)[:, 1]
    idx_hr = int(np.argmax(probs))
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[idx_hr], show=False, max_display=15)

    ticker_info = ""
    if "Ticker" in az_test.columns:
        row = az_test.iloc[idx_hr] if idx_hr < len(az_test) else pd.Series()
        ticker_info = f"  {row.get('Ticker','?')} — {row.get('Name','?')[:30]}"

    plt.title(f"SHAP Waterfall — Highest-Risk Company{ticker_info}\n"
              f"P(Default)={probs[idx_hr]:.2%}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {SHAP_DIR}/shap_waterfall.png")

    # Dependence — top feature
    top_feat = X_explain.columns[np.abs(shap_values.values).mean(axis=0).argmax()]
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(top_feat, shap_values.values, X_explain,
                         ax=ax, show=False)
    ax.set_title(f"SHAP Dependence Plot — {top_feat}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/shap_dependence_{top_feat.replace('/','_').replace(':','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {SHAP_DIR}/shap_dependence_{top_feat}.png")

else:
    print("\n[SKIP] SHAP not installed.")

# ══════════════════════════════════════════════════════════════════════════════
# LIME
# ══════════════════════════════════════════════════════════════════════════════
if LIME_OK:
    print("\n" + "=" * 65)
    print("LIME (secondary validation)")
    print("=" * 65)

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_model.values,
        feature_names=list(X_test_model.columns),
        class_names=["Safe", "Default"],
        mode="classification", random_state=42
    )
    probs_l = (xgb if XGB_OK else stack).predict_proba(X_test_model)[:, 1]
    idx_lime = int(np.argmax(probs_l))
    exp = lime_exp.explain_instance(
        X_test_model.values[idx_lime],
        (xgb if XGB_OK else stack).predict_proba,
        num_features=10, top_labels=1
    )
    fig_lime = exp.as_pyplot_figure()
    fig_lime.suptitle(f"LIME Explanation — Highest-Risk Company (row {idx_lime})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/lime_explanation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {SHAP_DIR}/lime_explanation.png")

else:
    print("\n[SKIP] LIME not installed.")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

best_name = metrics_df["AUC"].idxmax()
best_row  = metrics_df.loc[best_name]
print(f"\n✅ Best Model : {best_name}")
print(f"   AUC-ROC   : {best_row['AUC']:.4f}")
print(f"   Gini      : {best_row['Gini']:.4f}")
print(f"   KS        : {best_row['KS']:.4f}")
print(f"   F1        : {best_row['F1']:.4f}")

print(f"\n📂 Output Directory Structure:")
print(f"  {OUT_DIR}/")
print(f"    models/              — {len(os.listdir(MODEL_DIR))} saved .joblib model files")
print(f"    model_summaries/     — {len(os.listdir(SUMM_DIR))} JSON summaries + feature importance plots")
print(f"    predictions/         — {len(os.listdir(PRED_DIR))} per-model prediction CSVs (with AZS scores)")
print(f"    shap/                — {len(os.listdir(SHAP_DIR))} SHAP/LIME plots")
print(f"    model_comparison.csv — full metrics table")
print(f"    roc_curves.png")
print(f"    metrics_comparison.png")

print("\n🎉 PHASE 2 & 3 COMPLETE — Run credit_risk_phase3_esg.py for ESG comparison")
