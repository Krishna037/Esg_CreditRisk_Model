# Credit Risk Prediction Unified Pipeline v2.0

Detailed technical summary of the Capstone credit risk project, including data, methods, model training, ESG integration, outputs, and reliability controls.

## 1. Project Purpose

This project builds a production-style credit risk pipeline for corporate default prediction using a unified architecture and three phases:

1. Phase 1: HCRL Construction and Baseline Models
2. Phase 2: AZS Baseline Comparison
3. Phase 3: ESG Augmentation

The objective is to compare financial-only and ESG-augmented risk modeling with reproducible training, robust evaluation, and complete output artifacts for analysis and prediction.

## 2. Key Concepts

### 2.1 HCRL (Hybrid Credit Risk Label)

HCRL is a 3-pillar voting target:

1. AZS distress flag (AZS < 1.8)
2. Ohlson PD distress flag (PD_Ohlson > 0.5)
3. Market distress flag (derived from distress-proximity behavior)

Final HCRL label is 1 when at least 2 of the 3 pillars are distressed.

### 2.2 AZS Baseline

Phase 2 predicts AZS_Flag directly as a baseline benchmark for comparison against HCRL pipelines.

### 2.3 ESG Augmentation

Phase 3 keeps the HCRL target but adds ESG features to input signals for potential incremental performance.

## 3. Active Model Stack

The active pipeline currently trains these 7 models:

1. Logistic Regression
2. XGBoost
3. CatBoost
4. Random Forest
5. Neural Network (MLP)
6. Soft Voting Ensemble
7. Stacking Ensemble

Note: LightGBM is removed from active training flow.

## 4. Project Structure

```text
Capstone/
|- src/
|  |- config.py
|  |- data_helpers.py
|  |- preprocessing.py
|  |- model_training.py
|  |- evaluation.py
|  |- comparison_visualization.py
|  |- esg_integration.py
|  |- esg_visualization.py
|  |- pipeline.py
|  |- run_pipeline.py
|- data/
|- outputs/
|  |- unified_pipeline/
|- docs/
|- archive/
|- requirements.txt
|- README.md
```

## 5. Module-by-Module Method Summary

### 5.1 src/config.py

Centralized configuration for:

1. Paths
2. Feature lists and column mappings
3. Hyperparameters
4. Target thresholds
5. Leakage-proxy controls
6. Plot and output settings

### 5.2 src/data_helpers.py

Main methods:

1. load_raw_data(): reads Bloomberg file and applies numeric cleanup.
2. normalize_column_names(): standardizes raw field names.
3. construct_ohlson_score(): computes Ohlson probability and distress flag.
4. construct_hcrl_targets(): initializes AZS/Ohlson pillars and target stats.
5. engineer_features(): creates derived financial risk features.
6. compute_hcrl_target(): final 3-pillar vote for HCRL.
7. prepare_dataset_for_pipeline(): end-to-end feature/target preparation per phase.
8. build_azs_hcrl_audit_table(): transparent company-level target audit export.

### 5.3 src/preprocessing.py

Main methods:

1. preprocess_data():
   1. train-test split (stratified)
   2. median imputation (fit on train only)
   3. winsorization bounds (fit on train only)
   4. standard scaling (fit on train only)
   5. SMOTE+ENN resampling (train only)
2. apply_train_transforms_to_full_data(): reuses train-fitted transforms for scoring.
3. get_cv_splitter(): returns stratified cross-validation splitter.

### 5.4 src/model_training.py

Main methods:

1. train_logistic_regression()
2. train_xgboost()
3. train_catboost()
4. train_random_forest()
5. train_mlp()
6. train_soft_voting_ensemble()
7. train_stacking_ensemble()
8. train_all()

Includes stronger regularization and MLP early stopping for better generalization control.

### 5.5 src/evaluation.py

Main methods:

1. compute_metrics()
2. _optimal_f1_metrics()
3. _cross_validated_metrics()
4. _expected_calibration_error()
5. evaluate_all_models()
6. generate_roc_curves()
7. generate_pr_curves()
8. generate_confusion_matrices()
9. save_metrics_table()

Reported metrics include AUC, KS, Precision, Recall, F1, Brier, AP, ECE, and CV mean/std metrics.

### 5.6 src/pipeline.py and src/run_pipeline.py

Orchestration methods:

1. run_phase(): full train/evaluate/save for one phase.
2. run_all_phases(): executes Phase 1 and Phase 2 by default, Phase 3 with flag.
3. _generate_summary_report(): phase-level comparison and summary charts.

### 5.7 ESG Integration Modules

1. src/esg_integration.py: Talk-Walk ESG score integration pipeline.
2. src/esg_visualization.py: ESG-specific analytics charts.

## 6. Data Description

### 6.1 Primary Financial Data

Source: data/Bloomberg Data.xlsx

Contains core financial and market features such as:

1. Revenue, EPS, Returns
2. Assets, liabilities, debt ratios
3. Liquidity metrics
4. Volatility, beta, market cap
5. ESG columns used in Phase 3

### 6.2 ESG Data Assets

Relevant ESG files include:

1. data/company_esg_scores.csv
2. data/walk_data_esg_score.xlsx
3. data/brsr_esg_scores_optimized.csv
4. data/esg_talk_walk_integrated.csv

Phase 3 in unified pipeline uses configured ESG columns from Bloomberg data unless explicitly integrated with Talk-Walk outputs.

### 6.3 Target Rates

In latest robust run, phase datasets load approximately 395 rows after filtering with class imbalance in distress class (about 6.5% to 7.5% positives depending on target).

## 7. End-to-End Process Flow

```text
Raw Data
  -> Column normalization and cleaning
  -> Feature engineering
  -> Target construction (HCRL or AZS_Flag)
  -> Leakage/proxy feature controls
  -> Train-test split
  -> Train-fitted imputation/winsorization/scaling
  -> Train-only SMOTE+ENN
  -> Train 7 models
  -> Evaluate on untouched test split
  -> Save models, metrics, predictions, plots, metadata
  -> Build phase and ESG comparison visualizations
```

## 8. Why Earlier AUC Values Were Very High

Earlier near-perfect AUC behavior was primarily due to target-proxy leakage risk for engineered labels (especially HCRL) where strong proxy variables can help reconstruct labels too easily.

Current mitigations:

1. Explicit target-proxy feature dropping for HCRL phases.
2. Stronger model regularization.
3. Additional reliability diagnostics (ECE and CV stability metrics).

This improves trustworthiness even when AUC remains high.

## 9. Latest Output Summary

From latest full run (with ESG enabled):

1. Phase 1 best model: CatBoost, AUC 0.9946
2. Phase 2 best model: Logistic Regression, AUC 0.9932
3. Phase 3 best model: Logistic Regression, AUC 0.9973

See outputs/unified_pipeline/summary_comparison.csv for exact values.

## 10. Output Artifacts and What They Mean

### 10.1 Phase Folders

Each phase folder under outputs/unified_pipeline contains:

1. model_comparison.csv: full model ranking and metrics.
2. metrics.csv: phase metrics snapshot.
3. predictions_best_model.csv: best-model probabilities and class predictions.
4. metadata.json: data and preprocessing metadata.
5. best_model_why.txt: rationale for selected best model.
6. *.joblib: saved models.
7. plots/: ROC, PR, confusion matrices.

### 10.2 Cross-Phase and ESG Comparison

outputs/unified_pipeline/comparison_visualizations:

1. best_model_with_vs_without_esg.png
2. model_metrics_heatmap_with_vs_without_esg.png
3. esg_impact_delta_by_model.png
4. with_vs_without_esg_model_metrics.csv

### 10.3 ESG Integration Visuals

outputs/unified_pipeline/esg_integration_visualizations:

1. Score distributions
2. Confidence vs score
3. Risk flag frequencies
4. Talk vs Walk alignment
5. Gap analysis plots
6. Top/bottom ESG rankings
7. Correlation heatmap
8. Quantile band chart

## 11. How to Run

### 11.1 Install

```bash
cd Capstone
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 11.2 Run Core Pipeline (No ESG phase)

```bash
python src/run_pipeline.py
```

### 11.3 Run Full Pipeline (With ESG phase)

```bash
python src/run_pipeline.py --with-esg
```

### 11.4 Run ESG Talk-Walk Integration Only

```bash
python src/esg_integration.py
```

## 12. How to Use Saved Models for Prediction

Typical workflow:

1. Load a saved model (*.joblib) from a phase folder.
2. Prepare new data with same preprocessing pipeline assumptions.
3. Call predict_proba() for default risk score.
4. Apply operational threshold (0.5 or optimized threshold from metrics.csv).

Notes:

1. Use the same feature schema and ordering as training.
2. For production scoring, persist and apply the train-fitted transforms.

## 13. Reliability and Governance Guidance

For model governance decisions, use not only test AUC but also:

1. CV_AUC_Mean and CV_AUC_Std
2. CV_F1_Mean and CV_F1_Std
3. Brier score and ECE
4. Phase-to-phase ESG deltas

Recommended next upgrades:

1. Out-of-time validation split
2. Probability calibration (Platt/Isotonic)
3. Drift monitoring (PSI/CSI)
4. Bootstrapped confidence intervals for headline metrics

## 14. Dependencies

See Capstone/requirements.txt.

Core libraries:

1. numpy, pandas
2. scikit-learn
3. imbalanced-learn
4. xgboost
5. catboost
6. matplotlib, seaborn
7. openpyxl
8. joblib

## 15. Documentation Map

1. docs/README_PIPELINE.md: deeper pipeline design details.
2. docs/ADVANCED_TECHNIQUES.md: advanced modeling ideas.
3. docs/FINAL_DOCUMENTATION.md: latest implementation and result notes.
4. ESG_INTEGRATION_README.md: ESG Talk-Walk integration details.

## 16. Maintenance Notes

1. Update hyperparameters only in src/config.py.
2. Keep outputs/unified_pipeline as the single source of latest run artifacts.
3. Keep archive scripts as reference only, not active training flow.

---

Project Status: Production-oriented research pipeline with robust evaluation and ESG comparison capabilities.
