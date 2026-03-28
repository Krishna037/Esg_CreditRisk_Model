# Final Documentation - Capstone Credit Risk Pipeline

## Date
March 28, 2026

## Scope Completed
- Removed LightGBM from the active production pipeline.
- Added Neural Network (MLP) model with early stopping and L2 regularization.
- Added stronger regularization settings for XGBoost and Random Forest.
- Added anti-target-proxy feature controls to reduce deterministic HCRL reconstruction.
- Added robustness metrics in evaluation: ECE, CV_AUC_Mean, CV_AUC_Std, CV_F1_Mean, CV_F1_Std.
- Added additional with-vs-without ESG comparison visualizations.
- Added dedicated Capstone requirements file.

## Why AUC Was Reaching 1.00
AUC values near 1.00 happened mainly because HCRL target is constructed from AZS/Ohlson/market-distress logic, while model features still retained strong proxy variables linked to the same construction process.

This is a form of target-proxy leakage, where the model does not necessarily generalize better, but can reconstruct the engineered label very accurately.

## What Was Changed to Improve Trustworthiness
1. Removed direct and derived proxy features for HCRL phases:
   - Distress_Proximity and Ohlson-construction-related proxies.
2. Added regularization:
   - XGBoost: lower depth, lower learning rate, stronger L1/L2 penalties.
   - Random Forest: stronger leaf and split constraints.
   - MLP: L2 regularization and early stopping.
3. Added stability and calibration checks:
   - Cross-validation mean/std metrics.
   - Expected Calibration Error (ECE).

## Active Model Set
- Logistic Regression
- XGBoost
- CatBoost
- Random Forest
- Neural Network (MLP)
- Soft Voting Ensemble
- Stacking Ensemble

## Updated Output Artifacts
- Phase-specific model metrics and predictions:
  - outputs/unified_pipeline/phase_1_hcrl_construction_&_baseline_models/
  - outputs/unified_pipeline/phase_2_azs_baseline_comparison/
  - outputs/unified_pipeline/phase_3_esg_augmentation/
- Cross-phase comparison visuals:
  - outputs/unified_pipeline/comparison_visualizations/
- ESG integration visuals:
  - outputs/unified_pipeline/esg_integration_visualizations/

## Latest Training Results (Post-Refactor)

### Phase 1: HCRL Construction & Baseline Models
- Best model: CatBoost
- AUC: 0.9946
- F1: 0.5263

### Phase 2: AZS Baseline Comparison
- Best model: Logistic Regression
- AUC: 0.9932
- F1: 0.7059

### Phase 3: ESG Augmentation
- Best model: Logistic Regression
- AUC: 0.9973
- F1: 0.7143

### Interpretation
- Metrics remain high because labels are still engineered from structured financial risk logic.
- Results are now less artificially perfect than previous 1.0000 outcomes, and robustness diagnostics are available in each phase output.

## ESG vs Non-ESG Comparison Pack
- outputs/unified_pipeline/comparison_visualizations/best_model_with_vs_without_esg.png
- outputs/unified_pipeline/comparison_visualizations/model_metrics_heatmap_with_vs_without_esg.png
- outputs/unified_pipeline/comparison_visualizations/esg_impact_delta_by_model.png
- outputs/unified_pipeline/comparison_visualizations/with_vs_without_esg_model_metrics.csv

## Recommended Next Hardening Steps
- Add strict out-of-time validation split (time-based holdout).
- Add model calibration (Platt or isotonic) before production scoring.
- Add PSI/CSI drift monitoring for deployed scoring.
- Add confidence intervals via bootstrap for all headline metrics.
