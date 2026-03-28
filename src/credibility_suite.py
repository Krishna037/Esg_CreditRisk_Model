"""
=============================================================================
CREDIBILITY SUITE ORCHESTRATOR
=============================================================================
Master function that runs all XAI, reliability, and research credibility
enhancements after model training and evaluation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def run_complete_credibility_suite(
    trained_models: dict,           # {"ModelName": fitted_model}
    X_train_original: pd.DataFrame, # Pre-SMOTE training data
    X_train: np.ndarray,            # Preprocessed training data
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_full: pd.DataFrame,          # Full dataframe for gap analysis
    feature_names: list,
    config,
    output_dir: str,
    phase_name: str = "Phase 1",
):
    """
    Complete research credibility upgrade:
    1. SHAP explainability (XAI)
    2. Probability calibration (reliability)
    3. Hypothesis testing (statistical rigor)
    4. Leakage audit (governance)
    """
    
    credibility_dir = os.path.join(output_dir, "credibility")
    os.makedirs(credibility_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"CREDIBILITY SUITE: {phase_name}")
    print(f"{'='*80}")

    # ── 1. SHAP EXPLAINABILITY ───────────────────────────────────────────────
    if config.RUN_SHAP_ANALYSIS:
        print(f"\n[1/4] SHAP Explainability Analysis...")
        try:
            from shap_explainability import run_shap_analysis

            run_shap_analysis(
                models        = trained_models,
                X_train       = X_train_original,         # Pre-SMOTE real data
                X_test        = pd.DataFrame(X_test, columns=feature_names),
                feature_names = feature_names,
                output_dir    = os.path.join(credibility_dir, "shap"),
                phase_name    = phase_name,
                top_n         = config.SHAP_TOP_N_FEATURES,
            )
        except Exception as e:
            print(f"  ⚠ SHAP analysis failed: {e}")

    # ── 2. PROBABILITY CALIBRATION ───────────────────────────────────────────
    if config.RUN_CALIBRATION:
        print(f"\n[2/4] Probability Calibration & CV Holdout...")
        try:
            from calibration import (
                evaluate_and_calibrate_all_models,
                evaluate_on_cv_holdout,
            )

            # Calibrate on test set
            calibrated_models, cal_metrics = evaluate_and_calibrate_all_models(
                trained_models = trained_models,
                X_test         = X_test,
                y_test         = y_test,
                phase_name     = phase_name,
                output_dir     = os.path.join(credibility_dir, "calibration"),
            )

            # CV holdout evaluation (internal stability for single-year data)
            cv_results = evaluate_on_cv_holdout(
                models      = trained_models,
                X_train     = X_train,
                y_train     = y_train,
                phase_name  = phase_name,
                output_dir  = os.path.join(credibility_dir, "cv_holdout"),
                n_splits    = config.CV_HOLDOUT_FOLDS,
            )
        except Exception as e:
            print(f"  ⚠ Calibration failed: {e}")
            calibrated_models = trained_models

    else:
        calibrated_models = trained_models

    # ── 3. HYPOTHESIS TESTING (DeLong, McNemar, Permutation) ─────────────────
    if any([config.RUN_DELONG_TESTS, config.RUN_MCNEMAR_TESTS,
            config.RUN_PERMUTATION_TEST]):
        print(f"\n[3/4] Hypothesis Testing...")
        try:
            from hypothesis_testing import (
                run_all_delong_comparisons,
                run_all_mcnemar_comparisons,
                permutation_baseline_test,
                generate_reproducibility_table,
            )

            # Collect probas and predictions
            model_probas = {
                name: m.predict_proba(X_test)[:, 1]
                for name, m in calibrated_models.items()
            }
            model_preds = {
                name: m.predict(X_test)
                for name, m in calibrated_models.items()
            }

            hyp_dir = os.path.join(credibility_dir, "hypothesis_testing")

            # DeLong AUC comparison
            if config.RUN_DELONG_TESTS:
                run_all_delong_comparisons(
                    y_test, model_probas,
                    output_dir=os.path.join(hyp_dir, "delong"),
                    phase_name=phase_name,
                )

            # McNemar error comparison
            if config.RUN_MCNEMAR_TESTS:
                run_all_mcnemar_comparisons(
                    y_test, model_preds,
                    output_dir=os.path.join(hyp_dir, "mcnemar"),
                    phase_name=phase_name,
                )

            # Permutation test on best model
            if config.RUN_PERMUTATION_TEST:
                best_name = max(
                    model_probas,
                    key=lambda n: roc_auc_score(y_test, model_probas[n])
                )
                best_model = calibrated_models[best_name]
                permutation_baseline_test(
                    X_train, y_train, X_test, y_test,
                    model           = best_model,
                    model_name      = best_name,
                    output_dir      = os.path.join(hyp_dir, "permutation"),
                    n_permutations  = config.PERMUTATION_N_ITERATIONS,
                )

            # Reproducibility table
            if config.REPRODUCIBILITY_TABLE:
                generate_reproducibility_table(
                    config,
                    output_dir=os.path.join(hyp_dir, "reproducibility"),
                )

        except Exception as e:
            print(f"  ⚠ Hypothesis testing failed: {e}")

    # ── 4. LEAKAGE AUDIT ────────────────────────────────────────────────────
    if config.RUN_LEAKAGE_SCAN or config.RUN_HCRL_AUDIT:
        print(f"\n[4/4] Leakage Audit & Robustness Checks...")
        try:
            from hypothesis_testing import scan_feature_auc

            # Single-feature AUC scan
            if config.RUN_LEAKAGE_SCAN:
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
                X_test_df = pd.DataFrame(X_test, columns=feature_names)

                scan_feature_auc(
                    X_train_df, y_train, X_test_df, y_test,
                    output_dir=os.path.join(credibility_dir, "leakage_audit"),
                    leakage_threshold=config.LEAKAGE_SCAN_THRESHOLD,
                )

        except Exception as e:
            print(f"  ⚠ Leakage audit failed: {e}")

    print(f"\n{'='*80}")
    print(f"✓ Credibility suite complete → {credibility_dir}")
    print(f"{'='*80}\n")

    return calibrated_models


def run_esg_gap_deep_analysis(
    df_full: pd.DataFrame,
    config,
    output_dir: str,
):
    """
    ESG Talk-Walk gap analysis (Phase 3 only).
    Derives gap features, tests standalone predictiveness, default rates by quartile.
    """
    
    esg_dir = os.path.join(output_dir, "esg_gap_analysis")
    os.makedirs(esg_dir, exist_ok=True)

    if config.TALK_SCORE_COL not in df_full.columns or \
       config.WALK_SCORE_COL not in df_full.columns:
        print(f"  ⚠ Talk/Walk columns not found in data. Skipping ESG gap analysis.")
        return None

    print(f"\n{'='*80}")
    print(f"ESG TALK-WALK GAP ANALYSIS")
    print(f"{'='*80}\n")

    try:
        from esg_gap_analysis import (
            engineer_gap_features,
            test_gap_as_standalone_predictor,
            compute_gap_quartile_default_rates,
            plot_talk_walk_scatter,
            plot_gap_correlation_heatmap,
        )

        # ── 1. Engineer gap features ─────────────────────────────────────────
        print(f"[1/5] Engineering gap features...")
        df_gap = engineer_gap_features(
            df_full,
            talk_col    = config.TALK_SCORE_COL,
            walk_col    = config.WALK_SCORE_COL,
            company_col = "Name",
            year_col    = "fiscal_year",
        )

        gap_feature_cols = [c for c in config.GAP_FEATURES if c in df_gap.columns]

        # ── 2. Test gap as standalone predictor ─────────────────────────────
        print(f"\n[2/5] Testing gap features as standalone predictors...")
        test_gap_as_standalone_predictor(
            df          = df_gap,
            gap_features= gap_feature_cols,
            target_col  = config.HCRL_CONFIG.get("pillar1_name", "HCRL"),
            output_dir  = os.path.join(esg_dir, "standalone_tests"),
        )

        # ── 3. Default rates by gap quartile ───────────────────────────────
        print(f"\n[3/5] Computing default rates by gap quartile...")
        compute_gap_quartile_default_rates(
            df          = df_gap,
            gap_col     = "gap_raw",
            target_col  = config.HCRL_CONFIG.get("pillar1_name", "HCRL"),
            output_dir  = os.path.join(esg_dir, "quartile_analysis"),
        )

        # ── 4. Talk-Walk scatter ──────────────────────────────────────────
        print(f"\n[4/5] Creating Talk-Walk scatter plot...")
        plot_talk_walk_scatter(
            df          = df_gap,
            talk_col    = config.TALK_SCORE_COL,
            walk_col    = config.WALK_SCORE_COL,
            target_col  = config.HCRL_CONFIG.get("pillar1_name", "HCRL"),
            output_dir  = os.path.join(esg_dir, "scatter"),
            company_col = "Name",
        )

        # ── 5. Gap correlation heatmap ────────────────────────────────────
        print(f"\n[5/5] Creating gap-financial correlation heatmap...")
        # Use a subset of financial features
        financial_features = [c for c in df_gap.columns 
                            if any(x in c for x in 
                            ["Revenue", "EPS", "Assets", "Debt", "Ratio", "ROA"])][:8]
        plot_gap_correlation_heatmap(
            df                 = df_gap,
            gap_features       = gap_feature_cols,
            financial_features = financial_features,
            target_col         = config.HCRL_CONFIG.get("pillar1_name", "HCRL"),
            output_dir         = os.path.join(esg_dir, "correlations"),
        )

        print(f"\n{'='*80}")
        print(f"✓ ESG gap analysis complete → {esg_dir}")
        print(f"{'='*80}\n")

        return df_gap

    except Exception as e:
        print(f"  ⚠ ESG gap analysis failed: {e}")
        return None
