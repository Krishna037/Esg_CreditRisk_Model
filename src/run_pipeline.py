#!/usr/bin/env python
"""
=============================================================================
UNIFIED CREDIT RISK PIPELINE - QUICK START
=============================================================================
Simple example showing how to run the complete pipeline.

Run:
    python run_pipeline.py
"""

import sys
from pipeline import CreditRiskPipeline, main

def run_full_pipeline(include_esg_phase: bool = False):
    """Run full pipeline end-to-end (default: no ESG phase)."""
    print("\n" + "=" * 80)
    print(" " * 15 + "UNIFIED CREDIT RISK PIPELINE v2.0")
    phase_label = "Full Pipeline: 3 Phases (with ESG)" if include_esg_phase else "Core Pipeline: 2 Phases (no ESG)"
    print(" " * 10 + phase_label)
    print("=" * 80 + "\n")

    pipeline = CreditRiskPipeline(verbose=True)
    results = pipeline.run_all_phases(include_esg_phase=include_esg_phase)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    for phase_key, result in results.items():
        if result:
            phase_name = result["phase_name"]
            metrics_df = result["metrics_df"]
            best_auc = metrics_df.iloc[0]["AUC"]
            best_model = metrics_df.iloc[0]["Model"]
            print(f"\n[OK] {phase_name}")
            print(f"  Best Model: {best_model}")
            print(f"  AUC: {best_auc:.4f}")

    print("\n" + "=" * 80)
    print("Output saved to: outputs/unified_pipeline/")
    print("=" * 80 + "\n")

    return results


def run_single_phase(phase_key: str):
    """Run a single phase for testing."""
    print("\n" + "=" * 80)
    print(f"Running single phase: {phase_key}")
    print("=" * 80 + "\n")

    pipeline = CreditRiskPipeline(verbose=True)
    result = pipeline.run_phase(phase_key)

    if result:
        print(f"\n[OK] Phase complete!")
        print(f"  Models trained: {len(result['models'])}")
        print(f"  Best model: {result['metrics_df'].iloc[0]['Model']}")
        print(f"  Best AUC: {result['metrics_df'].iloc[0]['AUC']:.4f}")

    return result


if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1:
        phase = sys.argv[1]
        if phase in ["PHASE_1_HCRL", "PHASE_2_AZS", "PHASE_3_ESG"]:
            run_single_phase(phase)
        elif phase == "--with-esg":
            run_full_pipeline(include_esg_phase=True)
        else:
            print(f"Unknown phase: {phase}")
            print("Valid options: PHASE_1_HCRL, PHASE_2_AZS, PHASE_3_ESG, --with-esg")
    else:
        # Default: run non-ESG phases only
        run_full_pipeline(include_esg_phase=False)
