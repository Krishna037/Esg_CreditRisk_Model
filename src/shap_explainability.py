"""
=============================================================================
SHAP EXPLAINABILITY ANALYSIS
=============================================================================
Comprehensive SHAP-based model interpretation for credit risk models.
Generates beeswarm plots, waterfall charts, and feature importance comparisons.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _get_explainer(model, X_train_sample, model_name):
    """
    Returns the right SHAP explainer for each model type.
    TreeExplainer is exact and fast for XGB/CatBoost/RF.
    KernelExplainer is approximate — use a small background sample.
    """
    tree_models = ["XGBoost", "CatBoost", "RandomForest", "Soft Voting", "Stacking"]
    if any(tm in model_name for tm in tree_models):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            pass

    # KernelExplainer for LR, MLP, or fallback
    try:
        background = shap.kmeans(X_train_sample, min(50, len(X_train_sample)))
        if hasattr(model, 'predict_proba'):
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            predict_fn = lambda x: model.predict(x)
        return shap.KernelExplainer(predict_fn, background)
    except Exception as e:
        print(f"    Warning: KernelExplainer failed: {e}")
        return None


def _shap_values_for_model(explainer, X_test, model_name):
    """
    Extracts SHAP values as a plain 2D numpy array (samples × features).
    Handles both old-style (list of arrays) and new Explanation objects.
    """
    if explainer is None:
        return None
        
    try:
        raw = explainer.shap_values(X_test)

        # Binary classifiers sometimes return [neg_class, pos_class]
        if isinstance(raw, list):
            return raw[1] if len(raw) > 1 else raw[0]

        # New shap API returns an Explanation object
        if hasattr(raw, "values"):
            vals = raw.values
            if vals.ndim == 3:     # (samples, features, classes)
                return vals[:, :, 1] if vals.shape[2] > 1 else vals[:, :, 0]
            return vals

        return raw
    except Exception as e:
        print(f"    Error extracting SHAP values: {e}")
        return None


# ── MAIN SHAP ANALYSIS FUNCTION ──────────────────────────────────────────────

def run_shap_analysis(
    models: dict,           # {"XGBoost": fitted_model, "CatBoost": fitted_model, ...}
    X_train: pd.DataFrame,  # original (pre-SMOTE) training data, scaled
    X_test: pd.DataFrame,   # test data, scaled
    feature_names: list,    # column names in order
    output_dir: str,        # e.g. "outputs/unified_pipeline/phase1/shap"
    phase_name: str = "Phase 1",
    top_n: int = 15,        # how many features to show in plots
):
    """
    Comprehensive SHAP analysis for all models in a phase.
    Generates beeswarm, bar chart, waterfall, and cross-model heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_test_arr  = np.array(X_test)
    X_train_arr = np.array(X_train)

    all_mean_shap = {}   # model_name -> Series of mean |SHAP|

    print(f"\n{'='*70}")
    print(f"SHAP Analysis: {phase_name}")
    print(f"{'='*70}")

    for model_name, model in models.items():
        print(f"\n  Processing {model_name}...")
        model_dir = os.path.join(output_dir, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)

        try:
            explainer   = _get_explainer(model, X_train_arr, model_name)
            if explainer is None:
                print(f"    Skipped — explainer creation failed.")
                continue
                
            shap_vals   = _shap_values_for_model(explainer, X_test_arr, model_name)
            if shap_vals is None:
                print(f"    Skipped — SHAP value extraction failed.")
                continue
        except Exception as e:
            print(f"    Skipped — SHAP failed: {e}")
            continue

        # Convert to DataFrame and save
        shap_df = pd.DataFrame(shap_vals, columns=feature_names)
        shap_df.to_csv(os.path.join(model_dir, "shap_values.csv"), index=False)

        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        all_mean_shap[model_name] = mean_abs

        # ── Plot 1: Beeswarm (global importance + directionality) ────────────
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_vals,
                X_test_arr,
                feature_names=feature_names,
                max_display=top_n,
                show=False,
                plot_type="dot",    # beeswarm
            )
            plt.title(f"{phase_name} — {model_name}: SHAP beeswarm (top {top_n})",
                      fontsize=12, pad=12)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "shap_beeswarm.png"), 
                       dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✓ Beeswarm plot saved")
        except Exception as e:
            print(f"    ⚠ Beeswarm plot failed: {e}")

        # ── Plot 2: Bar chart (mean |SHAP|) ──────────────────────────────────
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            shap.summary_plot(
                shap_vals,
                X_test_arr,
                feature_names=feature_names,
                max_display=top_n,
                show=False,
                plot_type="bar",
            )
            plt.title(f"{phase_name} — {model_name}: mean |SHAP| (top {top_n})", 
                     fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "shap_bar.png"), 
                       dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✓ Bar chart saved  |  Top: {mean_abs.index[0]} ({mean_abs.iloc[0]:.4f})")
        except Exception as e:
            print(f"    ⚠ Bar chart failed: {e}")

        # ── Plot 3: Waterfall for 3 individual companies ──────────────────────
        try:
            _plot_individual_waterfalls(
                explainer, X_test_arr, feature_names, model_dir, model_name, phase_name
            )
            print(f"    ✓ Waterfall charts saved")
        except Exception as e:
            print(f"    ⚠ Waterfall charts failed: {e}")

    # ── Cross-model summary ───────────────────────────────────────────────────
    if all_mean_shap:
        try:
            summary_df = pd.DataFrame(all_mean_shap).fillna(0)
            summary_df.to_csv(os.path.join(output_dir, "shap_cross_model_summary.csv"))
            _plot_cross_model_heatmap(summary_df, output_dir, phase_name, top_n)
            print(f"\n  ✓ Cross-model SHAP summary saved")
        except Exception as e:
            print(f"  ⚠ Cross-model summary failed: {e}")

    print(f"\nSHAP analysis complete → {output_dir}\n")
    return all_mean_shap


def _plot_individual_waterfalls(explainer, X_test_arr, feature_names,
                                 model_dir, model_name, phase_name):
    """
    Saves waterfall plots for 3 representative test companies.
    Uses shap.Explanation objects for compatibility.
    """
    try:
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]

        raw = explainer.shap_values(X_test_arr)
        if isinstance(raw, list):
            vals = raw[1] if len(raw) > 1 else raw[0]
        elif hasattr(raw, "values"):
            vals = raw.values
            if vals.ndim == 3:
                vals = vals[:, :, 1] if vals.shape[2] > 1 else vals[:, :, 0]
        else:
            vals = raw

        n = len(vals)
        risk_score = vals.sum(axis=1)
        
        # Pick indices: high-risk, low-risk, mid-risk by total SHAP sum
        idx_high = int(np.argmax(risk_score))
        idx_low = int(np.argmin(risk_score))
        idx_mid = int(np.argsort(np.abs(risk_score - np.median(risk_score)))[0])

        for label, idx in [("high_risk", idx_high),
                            ("low_risk",  idx_low),
                            ("borderline", idx_mid)]:
            try:
                exp = shap.Explanation(
                    values        = vals[idx],
                    base_values   = base_val,
                    data          = X_test_arr[idx],
                    feature_names = feature_names,
                )
                fig, ax = plt.subplots(figsize=(9, 6))
                shap.waterfall_plot(exp, max_display=12, show=False)
                plt.title(f"{phase_name} — {model_name}: {label} company", fontsize=11)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(model_dir, f"shap_waterfall_{label}.png"),
                    dpi=150, bbox_inches="tight"
                )
                plt.close()
            except Exception as e:
                pass

    except Exception as e:
        print(f"    Waterfall detail plot failed: {e}")


def _plot_cross_model_heatmap(summary_df, output_dir, phase_name, top_n):
    """
    Heatmap of mean |SHAP| per feature × model.
    Rows = top features by average importance, cols = models.
    """
    try:
        import seaborn as sns
        
        top_features = summary_df.mean(axis=1).sort_values(ascending=False).head(top_n).index
        plot_df = summary_df.loc[top_features]

        fig, ax = plt.subplots(figsize=(max(8, len(summary_df.columns) * 1.5), 7))
        sns.heatmap(
            plot_df,
            ax=ax,
            cmap="YlOrRd",
            annot=True,
            fmt=".3f",
            linewidths=0.4,
            cbar_kws={"label": "mean |SHAP|"},
        )
        ax.set_title(f"{phase_name}: feature importance heatmap across models", fontsize=12)
        ax.set_xlabel("Model")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_cross_model_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Heatmap failed: {e}")


def plot_esg_shap_delta(phase1_shap_csv, phase3_shap_csv, output_dir):
    """
    Shows which ESG features entered the top-N after Phase 3 augmentation.
    Call after both phases complete.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        p1 = pd.read_csv(phase1_shap_csv, index_col=0)
        p3 = pd.read_csv(phase3_shap_csv, index_col=0)

        # Use mean across models
        p1_mean = p1.mean(axis=1).sort_values(ascending=False)
        p3_mean = p3.mean(axis=1).sort_values(ascending=False)

        # Rank delta: positive = rose in importance with ESG added
        all_feats = p3_mean.index.union(p1_mean.index)
        delta = p3_mean.reindex(all_feats).fillna(0) - p1_mean.reindex(all_feats).fillna(0)
        delta = delta.sort_values()

        esg_keywords = ["esg", "environmental", "social", "governance",
                        "talk", "walk", "brsr", "emission", "scope", "ghg", "women"]
        is_esg = delta.index.str.lower().str.contains("|".join(esg_keywords))

        colors = ["#1D9E75" if e else "#888780" for e in is_esg]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(delta.index, delta.values, color=colors, height=0.6)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Change in mean |SHAP| (Phase 3 − Phase 1)")
        ax.set_title("ESG augmentation impact on feature importance")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#1D9E75", label="ESG feature"),
            Patch(color="#888780", label="Financial feature"),
        ])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_esg_delta.png"), 
                   dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓ ESG SHAP delta plot saved → {output_dir}")
    except Exception as e:
        print(f"⚠ ESG delta plot failed: {e}")
