"""
ESG Talk–Walk Integration Module
================================
Integrates BRSR disclosure (Talk) and behavioral news (Walk) data
into a unified Final ESG Score for use in Probability of Default (PD) models.

Input: BRSR + pre-aggregated News datasets
Output: Integrated ESG scores (CSV + JSON) with risk flags and confidence levels [0-100 scale]
Date: March 2026
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import warnings

from esg_visualization import generate_esg_visualizations

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

CAPSTONE_DIR = Path(__file__).parent.parent
DATA_DIR = CAPSTONE_DIR / "data"

TALK_FILE = DATA_DIR / "brsr_esg_scores_optimized.csv"
WALK_FILE = DATA_DIR / "walk_data_esg_score.xlsx"
OUTPUT_CSV = DATA_DIR / "esg_talk_walk_integrated.csv"
OUTPUT_JSON = DATA_DIR / "esg_talk_walk_integrated.json"
VISUALS_DIR = CAPSTONE_DIR / "outputs" / "unified_pipeline" / "esg_integration_visualizations"

# Formula weights
ALPHA_TALK = 0.4
ALPHA_WALK = 0.6
BETA_PENALTY = 0.25
VOLATILITY_PENALTY = 0.1

# ESG Weights (should sum to 1.0)
ESG_WEIGHTS = {"E": 0.44, "S": 0.31, "G": 0.25}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 0. LOAD DATA
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Talk (BRSR) and Walk (pre-aggregated) datasets."""
    logger.info(f"Loading Talk dataset from {TALK_FILE}...")
    talk_df = pd.read_csv(TALK_FILE)
    logger.info(f"  ✓ Loaded {len(talk_df)} companies")
    
    logger.info(f"Loading Walk dataset from {WALK_FILE}...")
    walk_df = pd.read_excel(WALK_FILE)
    logger.info(f"  ✓ Loaded {len(walk_df)} records")
    
    return talk_df, walk_df


# ============================================================================
# 1. PREPROCESS DATA
# ============================================================================

def preprocess_data(talk_df: pd.DataFrame, walk_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize columns and normalize scores."""
    logger.info("Preprocessing data...")
    
    talk_df = talk_df.copy()
    walk_df = walk_df.copy()
    
    # Map company column
    if "company" in talk_df.columns:
        talk_df["Company_Name"] = talk_df["company"]
    
    # Extract Talk scores (already computed from BRSR keywords)
    for col_name, talk_key in [("Talk_E", "env_score"), ("Talk_S", "soc_score"), ("Talk_G", "gov_score")]:
        if talk_key in talk_df.columns:
            talk_df[col_name] = pd.to_numeric(talk_df[talk_key], errors="coerce")
        else:
            talk_df[col_name] = 0.1  # Transparency penalty
        
        # Normalize to [0,1] if needed
        if talk_df[col_name].max() > 1.0:
            talk_df[col_name] = talk_df[col_name] / 100.0
        talk_df[col_name] = talk_df[col_name].clip(0, 1)
    
    # Keep required Talk columns
    talk_df = talk_df[["Company_Name", "Talk_E", "Talk_S", "Talk_G"]].drop_duplicates(subset=["Company_Name"])
    logger.info(f"  ✓ Talk: {len(talk_df)} companies")
    
    # Map Walk company column
    if "company" in walk_df.columns:
        walk_df["Company_Name"] = walk_df["company"]
    
    # Extract pre-computed Walk scores
    walk_df["Walk_E"] = pd.to_numeric(walk_df.get("environmental_weighted_score", 0), errors="coerce").fillna(0).clip(-1, 1)
    walk_df["Walk_S"] = pd.to_numeric(walk_df.get("social_weighted_score", 0), errors="coerce").fillna(0).clip(-1, 1)
    walk_df["Walk_G"] = pd.to_numeric(walk_df.get("governance_weighted_score", 0), errors="coerce").fillna(0).clip(-1, 1)
    walk_df["article_count"] = pd.to_numeric(walk_df.get("total_articles", 0), errors="coerce").fillna(0)
    
    walk_df = walk_df[["Company_Name", "Walk_E", "Walk_S", "Walk_G", "article_count"]].drop_duplicates(subset=["Company_Name"])
    logger.info(f"  ✓ Walk: {len(walk_df)} companies")
    
    return talk_df, walk_df


# ============================================================================
# 2. COMPUTE TALK SCORES
# ============================================================================

def compute_talk_scores(talk_df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing Talk data with transparency penalty."""
    logger.info("Computing Talk scores...")
    result = talk_df.copy()
    
    for col in ["Talk_E", "Talk_S", "Talk_G"]:
        result[col] = result[col].fillna(0.1).clip(0, 1)
    
    result["missing_talk"] = talk_df[["Talk_E", "Talk_S", "Talk_G"]].isna().any(axis=1)
    logger.info(f"  ✓ Talk scores for {len(result)} companies")
    
    return result


# ============================================================================
# 3. EXTRACT WALK SCORES (Pre-aggregated)
# ============================================================================

def extract_walk_scores(walk_df: pd.DataFrame) -> pd.DataFrame:
    """Extract pre-computed Walk scores."""
    logger.info("Extracting Walk scores...")
    result = walk_df.copy()
    result["missing_walk"] = result["article_count"] <= 0
    logger.info(f"  ✓ Walk scores: {len(result)} companies ({(~result['missing_walk']).sum()} with articles)")
    return result


# ============================================================================
# 4. COMPUTE GAP (Greenwashing Detection)
# ============================================================================

def compute_gap(talk: pd.DataFrame, walk: pd.DataFrame) -> pd.DataFrame:
    """Compute Gap_i = Talk_i - Walk_i for greenwashing risk detection."""
    logger.info("Computing Gap (Talk - Walk)...")
    
    result = talk.merge(walk, on="Company_Name", how="outer")
    result["Talk_E"] = result["Talk_E"].fillna(0.1).clip(0, 1)
    result["Talk_S"] = result["Talk_S"].fillna(0.1).clip(0, 1)
    result["Talk_G"] = result["Talk_G"].fillna(0.1).clip(0, 1)
    result["missing_talk"] = result["missing_talk"].fillna(True)
    result["Walk_E"] = result["Walk_E"].fillna(0)
    result["Walk_S"] = result["Walk_S"].fillna(0)
    result["Walk_G"] = result["Walk_G"].fillna(0)
    result["article_count"] = result["article_count"].fillna(0)
    result["missing_walk"] = result["missing_walk"].fillna(False)
    
    for cat in ["E", "S", "G"]:
        result[f"Gap_{cat}"] = result[f"Talk_{cat}"] - result[f"Walk_{cat}"]
    
    greenwash_count = ((result["Gap_E"] > 0.5) | (result["Gap_S"] > 0.5) | (result["Gap_G"] > 0.5)).sum()
    logger.info(f"  ✓ Gap computed: {greenwash_count} companies with greenwashing risk")
    
    return result


# ============================================================================
# 5. APPLY PENALTIES & COMPUTE FINAL COMPONENT
# ============================================================================

def apply_penalties(gap_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ESG_Comp_i and apply Gap penalties."""
    logger.info("Applying penalties...")
    result = gap_df.copy()
    
    for cat in ["E", "S", "G"]:
        tc = f"Talk_{cat}"
        wc = f"Walk_{cat}"
        gap_c = f"Gap_{cat}"
        comp_c = f"Comp_{cat}"
        penalty_c = f"Penalty_{cat}"
        final_c = f"Final_{cat}"
        
        # ESG_Comp_i = 0.4 * Talk_i + 0.6 * Walk_i
        result[comp_c] = ALPHA_TALK * result[tc] + ALPHA_WALK * result[wc]
        
        # Penalty_i = beta * max(Gap_i, 0)
        result[penalty_c] = np.where(result[gap_c] > 0, BETA_PENALTY * result[gap_c], 0)
        
        # Final_i = Comp_i - Penalty_i
        result[final_c] = (result[comp_c] - result[penalty_c]).clip(-1, 1)
    
    logger.info(f"  ✓ Penalties applied")
    return result


# ============================================================================
# 6. COMPUTE VOLATILITY PENALTY
# ============================================================================

def compute_volatility_penalty(result_df: pd.DataFrame) -> pd.DataFrame:
    """Detect extreme gaps and Walk swings."""
    logger.info("Computing volatility...")
    result = result_df.copy()
    result["volatility_flag"] = False
    result["volatility_penalty"] = 0.0
    
    # Flag if any category has extreme gap (>0.7) or extreme Walk swing (>0.8)
    for cat in ["E", "S", "G"]:
        extreme_gap = (result[f"Gap_{cat}"].abs() > 0.7)
        extreme_walk = (result[f"Walk_{cat}"].abs() > 0.8)
        result.loc[extreme_gap | extreme_walk, "volatility_flag"] = True
    
    result["volatility_penalty"] = np.where(result["volatility_flag"], VOLATILITY_PENALTY, 0)
    logger.info(f"  ✓ Volatility: {result['volatility_flag'].sum()} companies flagged")
    
    return result


# ============================================================================
# 7. COMPUTE FINAL ESG SCORE (0-100 scale)
# ============================================================================

def compute_final_esg(penalty_df: pd.DataFrame) -> pd.DataFrame:
    """Weighted sum + volatility penalty, scaled to 0-100."""
    logger.info("Computing Final ESG Score...")
    result = penalty_df.copy()
    
    # Weighted combination
    result["Final_ESG"] = (
        ESG_WEIGHTS["E"] * result["Final_E"] +
        ESG_WEIGHTS["S"] * result["Final_S"] +
        ESG_WEIGHTS["G"] * result["Final_G"]
    ).clip(0, 1)
    
    # Scale to [0, 100]
    result["Final_ESG_Score"] = (result["Final_ESG"] * 100) - (result["volatility_penalty"] * 10)
    result["Final_ESG_Score"] = result["Final_ESG_Score"].clip(0, 100)
    
    logger.info(f"  ✓ Scores: mean={result['Final_ESG_Score'].mean():.1f}, std={result['Final_ESG_Score'].std():.1f}")
    
    return result


# ============================================================================
# 8. COMPUTE CONFIDENCE SCORE
# ============================================================================

def compute_confidence(result_df: pd.DataFrame) -> pd.DataFrame:
    """Confidence based on article coverage."""
    logger.info("Computing confidence...")
    result = result_df.copy()
    
    # Coverage metric
    coverage = np.log1p(result["article_count"]) / np.log(6)
    coverage = np.clip(coverage, 0, 1)
    
    # Assign confidence level
    result["Confidence_Level"] = np.where(
        result["missing_walk"],
        "Low",
        np.where(coverage < 0.33, "Low", np.where(coverage < 0.67, "Medium", "High"))
    )
    result["Confidence_Score"] = coverage
    
    conf_counts = result["Confidence_Level"].value_counts()
    logger.info(f"  ✓ Confidence: {dict(conf_counts)}")
    
    return result


# ============================================================================
# 9. GENERATE RISK FLAGS
# ============================================================================

def generate_flags(result_df: pd.DataFrame) -> pd.DataFrame:
    """Combine all risk indicators."""
    logger.info("Generating risk flags...")
    result = result_df.copy()
    
    flags = []
    for idx, row in result.iterrows():
        flag_list = []
        
        if (row["Gap_E"] > 0.5 or row["Gap_S"] > 0.5 or row["Gap_G"] > 0.5):
            flag_list.append("Greenwashing Risk")
        if row["volatility_flag"]:
            flag_list.append("High Volatility")
        if row["missing_talk"]:
            flag_list.append("Low Disclosure")
        if row["missing_walk"]:
            flag_list.append("Low Coverage")
        
        flags.append(" | ".join(flag_list) if flag_list else "No Risk")
    
    result["Risk_Flag"] = flags
    logger.info(f"  ✓ Flags: {(pd.Series(flags) != 'No Risk').sum()} companies with risks")
    
    return result


# ============================================================================
# 10. VALIDATE OUTPUT
# ============================================================================

def validate_output(result_df: pd.DataFrame) -> bool:
    """Validate numeric bounds and required columns."""
    logger.info("Validating...")
    
    checks = [
        ((result_df["Final_ESG_Score"] >= 0).all() and (result_df["Final_ESG_Score"] <= 100).all(), "Final_ESG_Score in [0,100]"),
        ((result_df[["Talk_E", "Talk_S", "Talk_G"]] >= 0).all().all() and (result_df[["Talk_E", "Talk_S", "Talk_G"]] <= 1).all().all(), "Talk in [0,1]"),
        ((result_df[["Walk_E", "Walk_S", "Walk_G"]] >= -1).all().all() and (result_df[["Walk_E", "Walk_S", "Walk_G"]] <= 1).all().all(), "Walk in [-1,1]"),
    ]
    
    all_valid = True
    for passed, check_name in checks:
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check_name}")
        all_valid = all_valid and passed
    
    return all_valid


# ============================================================================
# 11. EXPORT RESULTS
# ============================================================================

def export_results(result_df: pd.DataFrame) -> Dict:
    """Export to CSV and JSON."""
    logger.info("Exporting...")
    
    output_cols = [
        "Company_Name",
        "Talk_E", "Talk_S", "Talk_G",
        "Walk_E", "Walk_S", "Walk_G",
        "Gap_E", "Gap_S", "Gap_G",
        "Final_E", "Final_S", "Final_G",
        "Final_ESG_Score",
        "Confidence_Level",
        "Risk_Flag",
    ]
    
    export_df = result_df[output_cols].sort_values("Final_ESG_Score", ascending=False).reset_index(drop=True)
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"  ✓ CSV: {OUTPUT_CSV}")
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(export_df.to_dict(orient="records"), f, indent=2, default=str)
    logger.info(f"  ✓ JSON: {OUTPUT_JSON}")
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_companies": len(export_df),
        "confidence_l": (export_df["Confidence_Level"] == "Low").sum(),
        "confidence_m": (export_df["Confidence_Level"] == "Medium").sum(),
        "confidence_h": (export_df["Confidence_Level"] == "High").sum(),
        "greenwashing_risk": (export_df["Risk_Flag"].str.contains("Greenwashing")).sum(),
        "mean_score": float(export_df["Final_ESG_Score"].mean()),
        "median_score": float(export_df["Final_ESG_Score"].median()),
    }
    
    logger.info(f"\n📊 Summary: {len(export_df)} companies, mean score {metadata['mean_score']:.1f}")
    
    return metadata


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> pd.DataFrame:
    """Execute ESG Talk-Walk integration pipeline."""
    logger.info("=" * 80)
    logger.info("ESG TALK-WALK INTEGRATION PIPELINE")
    logger.info("=" * 80)
    
    try:
        talk_df, walk_df = load_data()
        talk_df, walk_df = preprocess_data(talk_df, walk_df)
        talk_scores = compute_talk_scores(talk_df)
        walk_scores = extract_walk_scores(walk_df)
        gap_df = compute_gap(talk_scores, walk_scores)
        penalty_df = apply_penalties(gap_df)
        volatility_df = compute_volatility_penalty(penalty_df)
        final_df = compute_final_esg(volatility_df)
        confidence_df = compute_confidence(final_df)
        flags_df = generate_flags(confidence_df)
        
        is_valid = validate_output(flags_df)
        metadata = export_results(flags_df)

        try:
            plot_paths = generate_esg_visualizations(flags_df, VISUALS_DIR)
            logger.info(f"Generated {len(plot_paths)} ESG visualizations in {VISUALS_DIR}")
        except Exception as viz_err:
            logger.warning(f"Visualization generation failed: {viz_err}")
        
        logger.info("=" * 80)
        logger.info("✓ ESG INTEGRATION COMPLETE")
        logger.info("=" * 80)
        
        return flags_df
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    result_df = main()
