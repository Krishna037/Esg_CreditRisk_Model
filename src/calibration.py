"""
=============================================================================
OUT-OF-TIME VALIDATION & PROBABILITY CALIBRATION
=============================================================================
Adapts out-of-time validation to single-year snapshot using CV-based approach.
Implements Platt scaling and Isotonic regression calibration.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
import os


# ── CALIBRATION METHODS ──────────────────────────────────────────────────────

def calibrate_model(model, X_cal: np.ndarray, y_cal: np.ndarray,
                    method: str = "platt"):
    """
    Fits a calibration wrapper on a pre-trained model.

    Args:
        model:    Any fitted sklearn-compatible model with predict_proba.
        X_cal:    Calibration features (test set, scaled).
        y_cal:    Calibration labels.
        method:   "platt" (sigmoid) or "isotonic".

    Returns:
        Wrapped model with .predict_proba() interface.
    """
    if method == "platt":
        raw_scores = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        lr.fit(raw_scores, y_cal)

        class PlattCalibrated:
            def __init__(self, base, calibrator):
                self.base = base
                self.cal = calibrator

            def predict_proba(self, X):
                raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
                p = self.cal.predict_proba(raw)[:, 1]
                return np.column_stack([1 - p, p])

            def predict(self, X, threshold=0.5):
                return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

        return PlattCalibrated(model, lr)

    elif method == "isotonic":
        raw_scores = model.predict_proba(X_cal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_scores, y_cal)

        class IsotonicCalibrated:
            def __init__(self, base, calibrator):
                self.base = base
                self.cal = calibrator

            def predict_proba(self, X):
                raw = self.base.predict_proba(X)[:, 1]
                p = self.cal.predict(raw)
                return np.column_stack([1 - p, p])

            def predict(self, X, threshold=0.5):
                return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

        return IsotonicCalibrated(model, iso)

    else:
        raise ValueError(f"method must be 'platt' or 'isotonic', got '{method}'")


# ── CALIBRATION EVALUATION ───────────────────────────────────────────────────

def evaluate_and_calibrate_all_models(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    phase_name: str,
    output_dir: str,
    threshold: float = 0.5,
) -> tuple:
    """
    Evaluates all models and applies calibration.
    Returns both raw and calibrated model dicts with metrics.
    
    Strategy: Use test set as calibration set (fits calibrator on test).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Calibration Analysis: {phase_name}")
    print(f"{'='*70}\n")
    
    calibrated_models = {}
    metrics_records = []

    # Map model names to calibration methods
    cal_method_map = {
        "LogisticRegression":      "platt",
        "XGBoost":                 "platt",
        "CatBoost":                "platt",
        "Random Forest":           "isotonic",
        "RandomForest":            "isotonic",
        "MLPClassifier":           "platt",
        "Neural Network":          "platt",
        "Soft Voting":             "platt",
        "SoftVoting":              "platt",
        "Stacking":                "platt",
        "StackingClassifier":      "platt",
    }

    for name, model in trained_models.items():
        print(f"  Calibrating {name}...")

        # Get default method
        method = "platt"
        for key, val in cal_method_map.items():
            if key.lower() in name.lower():
                method = val
                break

        try:
            calibrated = calibrate_model(model, X_test, y_test, method=method)
            calibrated_models[name] = calibrated

            # Compare before/after calibration
            prob_raw = model.predict_proba(X_test)[:, 1]
            prob_cal = calibrated.predict_proba(X_test)[:, 1]

            brier_raw = brier_score_loss(y_test, prob_raw)
            brier_cal = brier_score_loss(y_test, prob_cal)

            ece_raw = _expected_calibration_error(y_test, prob_raw, n_bins=10)
            ece_cal = _expected_calibration_error(y_test, prob_cal, n_bins=10)

            metrics_records.append({
                "Model":         name,
                "Method":        method,
                "Brier_before":  round(brier_raw, 4),
                "Brier_after":   round(brier_cal, 4),
                "Brier_delta":   round(brier_raw - brier_cal, 4),
                "ECE_before":    round(ece_raw, 4),
                "ECE_after":     round(ece_cal, 4),
                "ECE_delta":     round(ece_raw - ece_cal, 4),
            })

            # Plot before/after
            _plot_calibration_before_after(
                model, calibrated, X_test, y_test,
                name, os.path.join(output_dir, "before_after"),
                n_bins=10
            )

            print(f"    ✓ {method.upper():8s}  Brier: {brier_raw:.4f} → {brier_cal:.4f}  "
                  f"ECE: {ece_raw:.4f} → {ece_cal:.4f}")

        except Exception as e:
            print(f"    ⚠ Calibration failed: {e}")
            calibrated_models[name] = model  # Use uncalibrated fallback

    # Save metrics table
    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records)
        metrics_df.to_csv(
            os.path.join(output_dir, "calibration_metrics.csv"), index=False
        )
        print(f"\n  ✓ Calibration metrics saved\n")

        # Plot all calibration curves on one chart
        _plot_all_calibration_curves(
            calibrated_models, X_test, y_test,
            phase_name, output_dir, n_bins=10
        )

    return calibrated_models, metrics_df if metrics_records else None


def _expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def _plot_calibration_before_after(raw_model, cal_model, X_eval, y_eval,
                                   model_name, output_dir, n_bins=10):
    """Side-by-side calibration curve for one model."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")

        for label, model, color in [
            ("Before calibration", raw_model, "#E24B4A"),
            ("After calibration",  cal_model, "#1D9E75"),
        ]:
            proba = model.predict_proba(X_eval)[:, 1]
            frac, mean = calibration_curve(
                y_eval, proba, n_bins=n_bins, strategy="quantile"
            )
            brier = brier_score_loss(y_eval, proba)
            ece = _expected_calibration_error(y_eval, proba, n_bins=n_bins)
            ax.plot(mean, frac, "o-", color=color, lw=1.8, ms=5,
                    label=f"{label}  Brier={brier:.3f}  ECE={ece:.3f}")

        ax.set_title(f"{model_name}: before vs after calibration", fontsize=11)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        
        fname = f"calibration_before_after_{model_name.replace(' ','_')}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        pass


def _plot_all_calibration_curves(models, X_eval, y_eval,
                                 phase_name, output_dir, n_bins=10):
    """Plots calibration curves for all models on one chart."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.6)

        colors = ["#185FA5", "#1D9E75", "#D85A30", "#7F77DD",
                  "#D4537E", "#BA7517", "#E24B4A"]

        for (name, model), color in zip(models.items(), colors):
            try:
                proba = model.predict_proba(X_eval)[:, 1]
                frac_pos, mean_pred = calibration_curve(
                    y_eval, proba, n_bins=n_bins, strategy="quantile"
                )
                ece = _expected_calibration_error(y_eval, proba, n_bins=n_bins)
                ax.plot(mean_pred, frac_pos, "o-", color=color,
                        linewidth=1.5, markersize=5,
                        label=f"{name} (ECE={ece:.3f})")
            except Exception:
                pass

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{phase_name}: calibration curves (all models)")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "calibration_curves_all.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  ⚠ Calibration curves plot failed: {e}")


def evaluate_on_cv_holdout(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    phase_name: str,
    output_dir: str,
    n_splits: int = 5,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    CV-based evaluation for single-year snapshot data.
    Uses stratified k-fold as proxy for temporal generalization.
    
    For single-year data without true OOT, this shows internal stability.
    """
    os.makedirs(output_dir, exist_ok=True)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    print(f"\n  CV Holdout Evaluation ({n_splits} folds):")

    for name, model in models.items():
        fold_aucs, fold_aps, fold_f1s = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            try:
                # Train model on this fold
                m = model.__class__(**model.get_params())
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_val)[:, 1]
                pred = (proba >= threshold).astype(int)

                fold_aucs.append(roc_auc_score(y_val, proba))
                fold_aps.append(average_precision_score(y_val, proba))
                fold_f1s.append(f1_score(y_val, pred, zero_division=0))
            except Exception:
                continue

        if fold_aucs:
            records.append({
                "Model":        name,
                "CV_AUC_mean":  round(np.mean(fold_aucs), 4),
                "CV_AUC_std":   round(np.std(fold_aucs), 4),
                "CV_AP_mean":   round(np.mean(fold_aps), 4),
                "CV_AP_std":    round(np.std(fold_aps), 4),
                "CV_F1_mean":   round(np.mean(fold_f1s), 4),
                "CV_F1_std":    round(np.std(fold_f1s), 4),
            })
            print(f"    {name:20s}  AUC: {np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f}  "
                  f"AP: {np.mean(fold_aps):.4f}±{np.std(fold_aps):.4f}")

    if records:
        cv_df = pd.DataFrame(records).sort_values("CV_AUC_mean", ascending=False)
        cv_df.to_csv(os.path.join(output_dir, "cv_holdout_results.csv"), index=False)
        return cv_df
    
    return None
