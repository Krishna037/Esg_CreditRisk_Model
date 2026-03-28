"""
=============================================================================
UNIFIED CREDIT RISK MODEL PIPELINE - CONFIGURATION
=============================================================================
Centralized configuration for all phases.
"""

import os
from typing import List, Dict, Tuple

# ============================================================================
# DIRECTORIES
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "unified_pipeline")
MODELS_DIR = os.path.join(OUT_DIR, "models")
PREDS_DIR = os.path.join(OUT_DIR, "predictions")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
SUMMARIES_DIR = os.path.join(OUT_DIR, "summaries")

for directory in [OUT_DIR, MODELS_DIR, PREDS_DIR, PLOTS_DIR, SUMMARIES_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_FILE = "Bloomberg Data.xlsx"

PHASE1_FIELDS = [
    "Ticker", "Name", "Revenue_T12M", "EPS_T12M", "Total_Return_D1", "P_E",
    "Tot_Assets_LF", "Debt_EBITDA_LF", "Debt_Equity_LF", "Curr_Ratio_LF",
    "Quick_Ratio_LF", "FCF_T12M", "EBIT_T12M", "EBITDA_T12M", "Total_Liab_LF",
    "ROA_to_ROE_LF", "AZS", "CR_Msrmnt", "Total_Return_Y1", "Market_Cap",
    "Beta_M1", "Volat_D30",
]

ESG_FIELDS = ["Women_Board_Pct", "GHG_Scope1", "GHG_Scope3", "CO2_Scope1"]

COLUMN_RENAME_MAP = {
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
    "% Women on Bd:Y": "Women_Board_Pct",
    "GHG Scope 1:Y": "GHG_Scope1",
    "GHG Scope 3:Y": "GHG_Scope3",
    "CO2 Scope 1:Y": "CO2_Scope1",
}

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

LOGISTIC_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
    "C": 0.1,
}

XGBOOST_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "min_child_weight": 3,
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "auto_class_weights": "Balanced",
    "verbose": 0,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 450,
    "max_depth": 6,
    "min_samples_leaf": 4,
    "min_samples_split": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.01,
    "learning_rate_init": 0.001,
    "max_iter": 600,
    "early_stopping": True,
    "validation_fraction": 0.2,
    "n_iter_no_change": 25,
    "random_state": RANDOM_STATE,
}

# ============================================================================
# OHLSON MODEL CONSTANTS
# ============================================================================
OHLSON_COEFFICIENTS = {
    "INTERCEPT": -1.32,
    "SIZE": -0.407,
    "TLTA": 6.03,
    "WCTA": -1.43,
    "CLCA": 0.0757,
    "OENEG": -1.72,
    "NITA": -2.37,
    "FUTL": -1.83,
    "INTWO": 0.285,
    "CHIN": -0.521,
}

OHLSON_THRESHOLDS = {
    "PD": 0.50,  # Pillar 2 flag if PD_Ohlson > 0.50
}

# ============================================================================
# AZS THRESHOLDS
# ============================================================================
AZS_THRESHOLDS = {
    "SAFE": 2.99,
    "GREY": 1.81,
    "DISTRESS": 1.80,
}

# ============================================================================
# HCRL CONFIGURATION (3-PILLAR VOTING)
# ============================================================================
HCRL_CONFIG = {
    "pillar1_name": "AZS",
    "pillar1_threshold": 1.8,  # AZS < 1.8 → flag
    "pillar2_name": "Ohlson PD",
    "pillar2_threshold": 0.50,  # PD_Ohlson > 0.50 → flag
    "pillar3_name": "Market Distress",
    "pillar3_conditions": {
        "volatility_q": 0.75,
        "return_q": 0.25,
        "beta_threshold": 1.5,
        "market_cap_q": 0.25,
        "min_conditions_met": 2,
    },
    "voting_threshold": 2,  # Need ≥2 pillars to flag as default
}

# ============================================================================
# FEATURE ENGINEERING DERIVED FEATURES
# ============================================================================
DERIVED_FEATURES = [
    "Asset_Efficiency",
    "Earnings_Quality",
    "Profitability_Ratio",
    "Mkt_Adjusted_Lev",
    "Volat_Adj_Return",
    "Distress_Proximity",
    "Liquidity_Buffer",
    "Size_Lev_Ratio",
]

# ============================================================================
# VISUALIZATION
# ============================================================================
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#bab0ac"]

PLOT_STYLE = {
    "figsize": (10, 6),
    "dpi": 150,
    "fontsize": 11,
}

# ============================================================================
# PIPELINE PHASES
# ============================================================================
PHASES = {
    "PHASE_1_HCRL": {
        "name": "Phase 1: HCRL Construction & Baseline Models",
        "target": "HCRL",
        "include_esg": False,
        "description": "3-pillar HCRL target with financial features only",
    },
    "PHASE_2_AZS": {
        "name": "Phase 2: AZS Baseline Comparison",
        "target": "AZS_Flag",
        "include_esg": False,
        "description": "Baseline AZS-only target for comparison",
    },
    "PHASE_3_ESG": {
        "name": "Phase 3: ESG Augmentation",
        "target": "HCRL",
        "include_esg": True,
        "description": "Same HCRL but with ESG features included",
    },
}

# By default, run non-ESG phases only unless explicitly enabled.
ENABLE_ESG_PHASE = False

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================
VERBOSE = True
SAVE_INTERMEDIATE = True
SAVE_MODELS = True
SAVE_PREDICTIONS = True
SAVE_PLOTS = True

# ============================================================================
# QUALITY CHECKS
# ============================================================================
LEAKAGE_CHECK_FEATURES = {
    "Asset_Utilisation",
    "Profitability_ROA",
    "Interest_Coverage_Proxy",
    "Debt_Burden",
    "Market_to_Book",
    "Distress_Proximity",
}

# Additional target-proxy controls to reduce deterministic learning for HCRL.
HCRL_TARGET_PROXY_FEATURES = {
    "Tot_Assets_LF",
    "Total_Liab_LF",
    "Curr_Ratio_LF",
    "EBIT_T12M",
    "FCF_T12M",
    "EPS_T12M",
    "Distress_Proximity",
}

MIN_POSITIVE_SAMPLES_FOR_SMOTE = 5
MAX_QUANTILE = 0.99
MIN_QUANTILE = 0.01

# ============================================================================
# ESG TALK-WALK GAP ANALYSIS (NEW)
# ============================================================================
# ESG data file and column names for gap analysis
ESG_DATA_FILE = "esg_talk_walk_integrated.csv"

TALK_SCORE_COL = "Talk_Score"          # Disclosure/claim ESG score
WALK_SCORE_COL = "Walk_Score"          # Outcome/delivery ESG score
ESG_TOTAL_SCORE_COL = "Total_ESG_Score"

# Gap features to compute and include in Phase 3
GAP_FEATURES = [
    "gap_raw",
    "gap_abs",
    "gap_direction",
    "greenwashing_flag",
    "under_reporter_flag",
    "esg_consistency",
]

# ============================================================================
# CREDIBILITY & HYPOTHESIS TESTING (NEW)
# ============================================================================
# SHAP explainability parameters
RUN_SHAP_ANALYSIS = True
SHAP_TOP_N_FEATURES = 15
SHAP_BACKGROUND_SAMPLES = 50

# Calibration parameters
RUN_CALIBRATION = True
CALIBRATION_METHOD = "platt"  # or "isotonic"

# Hypothesis testing parameters
RUN_DELONG_TESTS = True
RUN_MCNEMAR_TESTS = True
RUN_PERMUTATION_TEST = True
PERMUTATION_N_ITERATIONS = 200

# Leakage audit parameters
LEAKAGE_SCAN_THRESHOLD = 0.85  # Single-feature AUC > threshold = suspect
RUN_LEAKAGE_SCAN = True
RUN_HCRL_AUDIT = True

# CV holdout evaluation (for single-year snapshot data)
RUN_CV_HOLDOUT = True
CV_HOLDOUT_FOLDS = 5

# Reproducibility
REPRODUCIBILITY_TABLE = True
