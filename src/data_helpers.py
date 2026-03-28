"""
=============================================================================
UNIFIED CREDIT RISK PIPELINE - DATA HELPERS
=============================================================================
Shared data loading, cleaning, and feature engineering utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from config import COLUMN_RENAME_MAP, DATA_FILE, DATA_DIR, PHASE1_FIELDS, ESG_FIELDS, DERIVED_FEATURES
from config import OHLSON_COEFFICIENTS, OHLSON_THRESHOLDS, AZS_THRESHOLDS, HCRL_CONFIG

# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================

def load_raw_data() -> pd.DataFrame:
    """Load and clean raw Bloomberg data."""
    file_path = f"{DATA_DIR}/{DATA_FILE}"
    df = pd.read_excel(file_path)
    return clean_numeric_data(df)


def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-ID columns to numeric and normalize missing markers."""
    out = df.copy()
    for col in out.columns:
        if col in ["Ticker", "Name"]:
            continue
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip().replace(
                {"": np.nan, "nan": np.nan, "None": np.nan}
            )
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column naming across all scripts."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def filter_by_fields(df: pd.DataFrame, fields: list) -> pd.DataFrame:
    """Select only required fields."""
    available = [f for f in fields if f in df.columns]
    return df[available].copy()


# ============================================================================
# TARGET CONSTRUCTION - HCRL (3-PILLAR VOTING)
# ============================================================================

def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    """Safe division with zero-division handling."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.fillna(default)


def construct_ohlson_score(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Ohlson (1980) model bankruptcy probability score.
    Returns (PD_Ohlson, Ohlson_Flag).
    """
    d = df.copy()

    # Ohlson components
    size = np.log(d["Tot_Assets_LF"].clip(lower=1))
    tlta = safe_divide(d["Total_Liab_LF"], d["Tot_Assets_LF"])
    wcta = safe_divide(d["Curr_Ratio_LF"] - 1, d["Curr_Ratio_LF"])
    clca = safe_divide(pd.Series(1.0, index=d.index), d["Curr_Ratio_LF"])
    oeneg = (d["Total_Liab_LF"] > d["Tot_Assets_LF"]).astype(int)
    nita = safe_divide(d["EBIT_T12M"], d["Tot_Assets_LF"])
    futl = safe_divide(d["FCF_T12M"], d["Total_Liab_LF"])
    intwo = (d["EPS_T12M"] < 0).astype(int)
    chin = (d["EBIT_T12M"] > 0).astype(int)  # Proxy for earnings growth

    components = pd.DataFrame({
        "SIZE": size, "TLTA": tlta, "WCTA": wcta, "CLCA": clca,
        "OENEG": oeneg, "NITA": nita, "FUTL": futl, "INTWO": intwo, "CHIN": chin,
    })
    components = components.fillna(components.median(numeric_only=True))

    # Apply Ohlson coefficients
    o_score = (
        OHLSON_COEFFICIENTS["INTERCEPT"]
        + OHLSON_COEFFICIENTS["SIZE"] * components["SIZE"]
        + OHLSON_COEFFICIENTS["TLTA"] * components["TLTA"]
        + OHLSON_COEFFICIENTS["WCTA"] * components["WCTA"]
        + OHLSON_COEFFICIENTS["CLCA"] * components["CLCA"]
        + OHLSON_COEFFICIENTS["OENEG"] * components["OENEG"]
        + OHLSON_COEFFICIENTS["NITA"] * components["NITA"]
        + OHLSON_COEFFICIENTS["FUTL"] * components["FUTL"]
        + OHLSON_COEFFICIENTS["INTWO"] * components["INTWO"]
        + OHLSON_COEFFICIENTS["CHIN"] * components["CHIN"]
    )

    pd_ohlson = np.exp(o_score) / (1 + np.exp(o_score))
    ohlson_flag = (pd_ohlson > OHLSON_THRESHOLDS["PD"]).astype(int)

    return pd_ohlson, ohlson_flag


def construct_hcrl_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Construct HCRL (3-pillar) and AZS targets.
    
    Returns:
        (df_with_targets, target_counts_dict)
    """
    d = df.copy()

    # Pillar 1: AZS flag
    d["AZS_Flag"] = (d["AZS"] < AZS_THRESHOLDS["DISTRESS"]).astype(int)

    # Pillar 2: Ohlson flag
    d["PD_Ohlson"], d["Ohlson_Flag"] = construct_ohlson_score(d)

    # Pillar 3: Market distress (computed post-split to prevent leakage)
    # Placeholder; will be filled in preprocessing
    d["Market_Flag"] = np.nan
    d["HCRL"] = np.nan

    # Additional target for Phase 2 comparison
    d["Default_AZS"] = d["AZS_Flag"]  # Alias

    target_stats = {
        "AZS_Flag_positives": int(d["AZS_Flag"].sum()),
        "AZS_Flag_rate": float(d["AZS_Flag"].mean() * 100),
        "Ohlson_Flag_positives": int(d["Ohlson_Flag"].sum()),
        "Ohlson_Flag_rate": float(d["Ohlson_Flag"].mean() * 100),
    }

    return d, target_stats


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for credit risk modeling."""
    d = df.copy()

    debt_ebitda_safe = d["Debt_EBITDA_LF"].replace(0, np.nan)
    azs_safe = d["AZS"].replace(0, 0.001)

    d["Asset_Efficiency"] = safe_divide(d["Revenue_T12M"], d["Tot_Assets_LF"])
    d["Earnings_Quality"] = safe_divide(d["FCF_T12M"], d["EBITDA_T12M"])
    d["Profitability_Ratio"] = safe_divide(d["EBIT_T12M"], d["Tot_Assets_LF"])
    d["Mkt_Adjusted_Lev"] = d["Debt_Equity_LF"] * d["Beta_M1"]
    d["Volat_Adj_Return"] = safe_divide(d["Total_Return_Y1"], (1 + d["Volat_D30"]))
    d["Distress_Proximity"] = safe_divide(pd.Series(1.0, index=d.index), azs_safe) * d["Volat_D30"]
    d["Liquidity_Buffer"] = d["Curr_Ratio_LF"] - d["Quick_Ratio_LF"]
    d["Size_Lev_Ratio"] = np.log(d["Market_Cap"].clip(lower=1)) / debt_ebitda_safe

    # Cap Distress_Proximity at 99th percentile
    cap99 = d["Distress_Proximity"].quantile(0.99)
    d["Distress_Proximity"] = d["Distress_Proximity"].clip(upper=cap99)

    # Guard against inf/nan
    d = d.replace([np.inf, -np.inf], np.nan)

    return d


def compute_hcrl_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HCRL target using 3-pillar majority voting.
    Pillar 1: AZS < 1.8 → flag
    Pillar 2: Ohlson PD > 0.50 → flag  
    Pillar 3: Market distress (based on volatility/return proxy) → flag
    
    Returns df with HCRL and Market_Flag computed.
    """
    d = df.copy()
    
    # Pillar 3: Market distress flag (using Distress_Proximity as proxy)
    # Higher distress proximity = higher distress flag
    market_distress_threshold = d["Distress_Proximity"].quantile(0.66)
    d["Market_Flag"] = (d["Distress_Proximity"] > market_distress_threshold).astype(int)
    
    # HCRL: Majority voting (2 out of 3 pillars)
    pillar_sum = d["AZS_Flag"] + d["Ohlson_Flag"] + d["Market_Flag"]
    d["HCRL"] = (pillar_sum >= 2).astype(int)
    
    return d


def build_azs_hcrl_audit_table(include_esg: bool = False) -> pd.DataFrame:
    """Create per-company AZS/HCRL audit table with zones and pillar flags."""
    df = load_raw_data()
    df = normalize_column_names(df)
    df = filter_by_fields(df, PHASE1_FIELDS + ESG_FIELDS)
    df = df.dropna(subset=["AZS"]).reset_index(drop=True)
    df, _ = construct_hcrl_targets(df)
    df = engineer_features(df)
    df = compute_hcrl_target(df)

    df["AZS_Zone"] = pd.cut(
        df["AZS"],
        bins=[-np.inf, AZS_THRESHOLDS["DISTRESS"], AZS_THRESHOLDS["SAFE"], np.inf],
        labels=["Distress", "Grey", "Safe"],
        right=False,
    ).astype(str)

    cols = [
        "Ticker", "Name", "AZS", "AZS_Zone", "AZS_Flag",
        "PD_Ohlson", "Ohlson_Flag", "Distress_Proximity", "Market_Flag", "HCRL",
    ]

    if include_esg:
        cols.extend([c for c in ESG_FIELDS if c in df.columns])

    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols].sort_values("AZS", ascending=True).reset_index(drop=True)



# ============================================================================
# TARGET FILTERING & PREPARATION
# ============================================================================

def prepare_dataset_for_pipeline(
    phase: str,
    include_esg: bool = False,
    drop_leakage_features: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Complete data preparation for a specific phase.
    
    Args:
        phase: "PHASE_1_HCRL", "PHASE_2_AZS", or "PHASE_3_ESG"
        include_esg: Whether to include ESG features
        drop_leakage_features: Whether to remove known target-proxy features
    
    Returns:
        (X, y, metadata_dict)
    """
    # Load and clean
    df = load_raw_data()
    df = normalize_column_names(df)
    df = filter_by_fields(df, PHASE1_FIELDS + ESG_FIELDS)
    df = df.dropna(subset=["AZS"]).reset_index(drop=True)

    # Construct targets
    df, target_stats = construct_hcrl_targets(df)

    # Feature engineering
    df = engineer_features(df)
    
    # Compute HCRL target using 3-pillar voting
    df = compute_hcrl_target(df)

    # Select target and features based on phase
    if phase == "PHASE_2_AZS":
        y = df["AZS_Flag"].astype(int)
        target_name = "AZS_Flag"
    else:  # PHASE_1_HCRL or PHASE_3_ESG
        y = df["HCRL"].astype(int)
        target_name = "HCRL"

    # Remove ESG if not needed
    esg_cols = [c for c in ESG_FIELDS if c in df.columns]
    if not include_esg:
        df = df.drop(columns=esg_cols, errors="ignore")

    # Remove AZS from features (it's a target-proxy via pillar 1)
    X = df.drop(
        columns=[
            "Ticker", "Name", "HCRL", "AZS", "AZS_Flag",
            "Ohlson_Flag", "Market_Flag", "PD_Ohlson", "Default_AZS"
        ],
        errors="ignore"
    )

    # Remove known leakage features if requested
    if drop_leakage_features:
        from config import LEAKAGE_CHECK_FEATURES, HCRL_TARGET_PROXY_FEATURES
        leakage_cols = [c for c in LEAKAGE_CHECK_FEATURES if c in X.columns]
        if leakage_cols:
            X = X.drop(columns=leakage_cols)

        # Extra guard for HCRL phases: drop direct proxies used in target construction.
        if phase in {"PHASE_1_HCRL", "PHASE_3_ESG"}:
            proxy_cols = [c for c in HCRL_TARGET_PROXY_FEATURES if c in X.columns]
            if proxy_cols:
                X = X.drop(columns=proxy_cols)

    # Guard against inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)

    metadata = {
        "phase": phase,
        "target": target_name,
        "n_samples": len(df),
        "n_features": X.shape[1],
        "target_positives": int(y.sum()),
        "target_rate": float(y.mean() * 100),
        "n_missing": int(X.isna().sum().sum()),
        "include_esg": include_esg,
        "feature_names": X.columns.tolist(),     # NEW: for SHAP and other analyses
        "df_full": df,                            # NEW: full dataframe for ESG gap analysis
        **target_stats,
    }

    return X, y, metadata
