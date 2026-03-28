# Unified Credit Risk Model Pipeline

**Version:** 2.0 (Refactored & Consolidated)  
**Status:** Production-Ready  
**Last Updated:** March 2026

---

## Overview

A professional, modular machine learning pipeline for corporate credit risk modeling combining:
- **Phase 1: HCRL Construction** - 3-pillar default target with baseline models
- **Phase 2: AZS Baseline** - Traditional Altman-based comparison
- **Phase 3: ESG Augmentation** - Credit impact of ESG factors

All phases run through a **unified architecture** with shared utilities, proper train-test separation (zero leakage), and consistent reporting.

---

## Architecture

```
pipeline.py                 ← Main orchestrator (entry point)
    ↓
├── data_helpers.py        ← Data loading, cleaning, target construction
├── preprocessing.py       ← Train-test split, imputation, scaling, resampling
├── model_training.py      ← Unified model training (7 algorithms)
├── evaluation.py          ← Metrics, plots, comparison reports
└── config.py              ← All constants, hyperparameters, file paths
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| **config.py** | Centralized configuration: paths, hyperparameters, constants |
| **data_helpers.py** | Load Bloomberg data, construct HCRL/AZS targets, feature engineering |
| **preprocessing.py** | Train-test split → imputation → scaling → SMOTE resampling (preventing leakage) |
| **model_training.py** | Unified model trainer: LR, XGB, CatBoost, RF, MLP, Soft Voting, Stacking |
| **evaluation.py** | Metrics computation, plots (ROC/PR/confusion), comparison tables |
| **pipeline.py** | Orchestrator: chains all modules, runs phases 1-3 end-to-end |

---

## Key Improvements Over Legacy Scripts

### 1. **Zero Code Duplication**
- ✅ Single data loading → used by all phases
- ✅ Single preprocessing → used by all phases
- ✅ Single model training → same hyperparameters across phases
- ✅ Single evaluation → consistent metrics everywhere

### 2. **Proper Data Leakage Prevention**
- ✅ Train-test split happens **first** (before any statistics)
- ✅ Imputation medians fit on **train only** → applied to test
- ✅ Winsorization bounds fit on **train only** → applied to test
- ✅ Scaler fit on **train only** → applied to test
- ✅ SMOTE+ENN resampling on **train only** (test untouched)
- ✅ Market distress thresholds computed on **train only**
- ✅ Raw AZS removed from features (was leaking target)

### 3. **Modular & Extensible**
- Easy to add new algorithms (just add method to `UnifiedModelTrainer`)
- Easy to add new metrics (just add to `compute_metrics()`)
- Easy to add new phases (just define in `PHASES` config)
- Configuration-driven (change hyperparameters in one place)

### 4. **Professional Structure**
- Consistent error handling and logging
- Type hints throughout
- Dataclass-based data containers
- Clear progress reporting
- Automatic artifact saving (models, predictions, plots, metadata)

---

## Usage

### Quick Start

```python
# Run all 3 phases end-to-end
from pipeline import main

results = main()
```

### Run Single Phase

```python
from pipeline import CreditRiskPipeline

pipeline = CreditRiskPipeline(verbose=True)
phase1_result = pipeline.run_phase("PHASE_1_HCRL")
phase2_result = pipeline.run_phase("PHASE_2_AZS")
phase3_result = pipeline.run_phase("PHASE_3_ESG")
```

### Access Results

```python
# Get best model for Phase 1
best_model = phase1_result["models"]["XGBoost"]
metrics_df = phase1_result["metrics_df"]

# Get predictions evaluator
evaluator = phase1_result["evaluator"]
predictions = evaluator.predictions

# Plot data
prep_data = phase1_result["prep_data"]
X_train = prep_data.X_train
y_train = prep_data.y_train
```

---

## Output Structure

```
outputs/unified_pipeline/
├── summary_comparison.csv                 # High-level phase comparison
├── model_comparison.csv                   # (Overall) model metrics
├── phase_1_hcrl_construction_baseline_models/
│   ├── logistic_regression.joblib
│   ├── xgboost.joblib
│   ├── catboost.joblib
│   ├── random_forest.joblib
│   ├── neural_network_(mlp).joblib
│   ├── soft_voting_ensemble.joblib
│   ├── stacking_ensemble.joblib
│   ├── metrics.csv
│   ├── metadata.json
│   └── predictions_best_model.csv
├── phase_2_azs_baseline_comparison/
│   ├── [same structure]
├── phase_3_esg_augmentation/
│   ├── [same structure]
├── plots/
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── confusion_matrix_*.png
└── models/
    └── [all trained .joblib files]
```

---

## Configuration

Edit `config.py` to customize:

```python
# Data
DATA_FILE = "Bloomberg Data.xlsx"
PHASE1_FIELDS = [...]  # Required columns
ESG_FIELDS = [...]     # ESG features

# Models
XGBOOST_PARAMS = {...}       # Hyperparameters
MLP_PARAMS = {...}
# ... etc

# Pipeline
TEST_SIZE = 0.20             # Train-test split ratio
CV_FOLDS = 5                 # Cross-validation folds
RANDOM_STATE = 42            # Reproducibility seed

# Output
OUT_DIR = "outputs/unified_pipeline"
VERBOSE = True               # Progress reporting
SAVE_MODELS = True           # Save trained models
SAVE_PREDICTIONS = True      # Save test set predictions
```

---

## Data Flow

### Phase 1: HCRL

```
[Bloomberg Data] 
    ↓
[Clean & Engineer Features]
    ↓
[Construct HCRL Target: 3-pillar voting]
    - Pillar 1: AZS < 1.8
    - Pillar 2: Ohlson PD > 0.50
    - Pillar 3: Market distress (computed post-split to prevent leakage)
    ↓
[Train-Test Split 80-20 stratified]
    ↓
[Imputation (fit train → apply to both)]
    ↓
[Winsorization (fit train → apply to both)]
    ↓
[Scaling (fit train → apply to both)]
    ↓
[SMOTE+ENN Resampling (train only)]
    ↓
[Train 7 Models: LR, XGB, Cat, RF, MLP, Soft Voting, Stacking]
    ↓
[Evaluate on Test Set → Metrics & Plots]
```

### Phases 2 & 3

Same pipeline, different targets:
- **Phase 2**: Target = `AZS_Flag` (traditional benchmark)
- **Phase 3**: Same as Phase 1 but with ESG features included

---

## Leakage Prevention Checklist

✅ Train-test split **before** any statistics  
✅ Imputation medians fit on **train only**  
✅ Winsorization bounds fit on **train only**  
✅ Scaler fit on **train only**  
✅ SMOTE resampling on **train only**  
✅ Market distress thresholds computed on **train only**  
✅ Raw AZS removed from features  
✅ No test data seen during fit operations  

---

## Models Included

1. **Logistic Regression** - Baseline, interpretable
2. **XGBoost** - Boosting with grid search
3. **CatBoost** - Auto class weights
4. **Random Forest** - Ensemble baseline
5. **Neural Network (MLP)** - Early stopping + L2 regularization
6. **Soft Voting Ensemble** - Weighted probabilistic ensemble
7. **Stacking Ensemble** - Meta-learner combining base models

---

## Metrics Reported

- **AUC-ROC** - Discrimination threshold-independent
- **KS Statistic** - Maximum separation between default/non-default
- **Precision** - % predicted defaults that truly default
- **Recall** - % of true defaults caught
- **F1** - Harmonic mean of precision & recall
- **Brier Score** - Calibration quality
- **AP (Average Precision)** - PR curve area
- **ECE (Expected Calibration Error)** - Probability trustworthiness
- **CV_AUC_Mean / CV_AUC_Std** - Stability across folds

---

## Reproducibility

- All operations use `RANDOM_STATE=42`
- Stratified train-test splits maintain class balance
- Cross-validation is deterministic
- Results are reproducible across runs on same hardware

---

## Performance Expectations

**Important:** After deduplication and leakage fixes, expect:
- ✅ More **realistic** metrics (not all 1.0)
- ✅ Stable performance across CV folds
- ✅ Meaningful differences between models
- ✅ ESG features genuinely improve (if they add signal)

---

## Troubleshooting

### Out of Memory
- Reduce `CV_FOLDS` in config
- Reduce `n_estimators` in model params
- Use subset of data for testing

### Slow Training
- Set `do_grid_search=False` in `train_xgboost()`
- Reduce hyperparameter grid
- Use fewer CV folds

### Model Not Saving
- Check folder permissions: `outputs/unified_pipeline/`
- Verify `SAVE_MODELS=True` in config

### Low Metrics
- Check target distribution (too imbalanced?)
- Review feature engineering (meaningful predictors?)
- Ensure data quality (missing values?)

---

## Next Steps

1. ✅ Run full pipeline: `python pipeline.py`
2. ✅ Review `outputs/unified_pipeline/summary_comparison.csv`
3. ✅ Load best model: `joblib.load("model.joblib")`
4. ✅ Score new data: `best_model.predict_proba(X_new)`
5. ✅ Deploy to production (via containerization/API)

---

## Citation

If using this pipeline, cite:
> Unified Credit Risk Pipeline v2.0 (2026)
> 3-Pillar HCRL Target with Leakage-Proof Preprocessing

---

## License

Internal Use Only  
Bloomberg Data Subject to License Terms

