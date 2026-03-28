"""
=============================================================================
UNIFIED CREDIT RISK MODEL PIPELINE - MAIN ORCHESTRATOR
=============================================================================
End-to-end pipeline that coordinates all phases:
  - Phase 1: HCRL Construction + Baseline Models
  - Phase 2: AZS Baseline Comparison
  - Phase 3: ESG Augmentation
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import config
from config import (
    PHASES, OUT_DIR, MODELS_DIR, PREDS_DIR, PLOTS_DIR, SUMMARIES_DIR,
    VERBOSE, SAVE_MODELS, SAVE_PREDICTIONS, ENABLE_ESG_PHASE,
    RUN_SHAP_ANALYSIS, RUN_CALIBRATION, RUN_DELONG_TESTS,
    RUN_MCNEMAR_TESTS, RUN_PERMUTATION_TEST, RUN_LEAKAGE_SCAN,
    RUN_CV_HOLDOUT, REPRODUCIBILITY_TABLE, RANDOM_STATE
)
from data_helpers import prepare_dataset_for_pipeline, build_azs_hcrl_audit_table
from preprocessing import preprocess_data, apply_train_transforms_to_full_data
from model_training import UnifiedModelTrainer
from evaluation import ModelEvaluator
from comparison_visualization import generate_comparison_visualizations
from credibility_suite import run_complete_credibility_suite, run_esg_gap_deep_analysis

import joblib


class CreditRiskPipeline:
    """Unified pipeline orchestrator for all 3 phases."""

    def __init__(self, verbose: bool = VERBOSE):
        """Initialize pipeline."""
        self.verbose = verbose
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def run_phase(self, phase_key: str) -> dict:
        """
        Run a single phase: data prep > preprocessing > training > evaluation.
        
        Args:
            phase_key: "PHASE_1_HCRL", "PHASE_2_AZS", or "PHASE_3_ESG"
        
        Returns:
            Dictionary with phase results
        """
        phase_config = PHASES[phase_key]
        phase_name = phase_config["name"]

        print("\n" + "=" * 80)
        print(f"{'='*80}")
        print(f" {phase_name}")
        print(f" {phase_config['description']}")
        print(f"{'='*80}")
        print("=" * 80)

        try:
            # ====================================================================
            # STAGE 1: DATA PREPARATION
            # ====================================================================
            if self.verbose:
                print(f"\n[STAGE 1] Data Preparation...")
            
            include_esg = phase_config.get("include_esg", False)
            X, y, data_metadata = prepare_dataset_for_pipeline(
                phase=phase_key,
                include_esg=include_esg,
                drop_leakage_features=True,
            )

            if self.verbose:
                print(f"  [OK] Loaded {data_metadata['n_samples']} samples")
                print(f"  [OK] Features: {data_metadata['n_features']}")
                print(f"  [OK] Target: {data_metadata['target']} ({data_metadata['target_rate']:.2f}% positive)")

            # ====================================================================
            # STAGE 2: PREPROCESSING (Train-Test Split + Transforms)
            # ====================================================================
            if self.verbose:
                print(f"\n[STAGE 2] Preprocessing...")

            prep_data = preprocess_data(
                X=X,
                y=y,
                target_name=data_metadata["target"],
                resample=True,
            )

            # ====================================================================
            # STAGE 3: MODEL TRAINING
            # ====================================================================
            if self.verbose:
                print(f"\n[STAGE 3] Model Training...")

            trainer = UnifiedModelTrainer(prep_data, verbose=self.verbose)
            models = trainer.train_all(skip_ensemble=False)

            if self.verbose:
                print(f"  [OK] Trained {len(models)} models")

            # ====================================================================
            # STAGE 4: EVALUATION
            # ====================================================================
            if self.verbose:
                print(f"\n[STAGE 4] Model Evaluation...")

            phase_name_key = PHASES[phase_key]["name"].replace(" ", "_").replace(":", "").lower()
            phase_dir = os.path.join(OUT_DIR, phase_name_key)
            evaluator = ModelEvaluator(
                prep_data,
                trainer,
                verbose=self.verbose,
                output_dir=phase_dir,
                plots_dir=os.path.join(phase_dir, "plots"),
            )
            metrics_df = evaluator.evaluate_all_models()
            evaluator.generate_all_reports()

            # ====================================================================
            # STAGE 4.5: CREDIBILITY SUITE (XAI, Calibration, Hypothesis Tests)
            # ====================================================================
            if self.verbose:
                print(f"\n[STAGE 4.5] Research Credibility Suite...")

            try:
                # Run credibility suite for all phases
                calibrated_models = run_complete_credibility_suite(
                    trained_models      = models,
                    X_train_original    = prep_data.X_train_original,
                    X_train             = prep_data.X_train_scaled,
                    y_train             = prep_data.y_train,
                    X_test              = prep_data.X_test_scaled,
                    y_test              = prep_data.y_test,
                    df_full             = data_metadata.get("df_full"),
                    feature_names       = data_metadata.get("feature_names", []),
                    config              = config,
                    output_dir          = phase_dir,
                    phase_name          = phase_name,
                )

                # ESG gap analysis for Phase 3 only
                if phase_key == "PHASE_3_ESG" and include_esg:
                    try:
                        run_esg_gap_deep_analysis(
                            df_full    = data_metadata.get("df_full"),
                            config     = config,
                            output_dir = phase_dir,
                        )
                    except Exception as e:
                        if self.verbose:
                            print(f"  ⚠ ESG gap analysis not completed: {e}")

                if self.verbose:
                    print(f"  [OK] Credibility suite complete")

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Credibility suite skipped: {e}")
                calibrated_models = models

            # ====================================================================
            # STAGE 5: SAVE ARTIFACTS
            # ====================================================================
            if self.verbose:
                print(f"\n[STAGE 5] Saving Artifacts...")

            phase_result = {
                "phase": phase_key,
                "phase_name": phase_name,
                "timestamp": self.timestamp,
                "data_metadata": data_metadata,
                "prep_data": prep_data,
                "models": models,
                "metrics_df": metrics_df,
                "evaluator": evaluator,
            }

            self._save_phase_artifacts(phase_key, phase_result)

            if self.verbose:
                print(f"\n{'='*80}")
                print(f"[OK] Phase complete: {phase_name}")
                print(f"{'='*80}\n")

            return phase_result

        except Exception as e:
            print(f"\n[ERROR] Phase {phase_key} failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_phase_artifacts(self, phase_key: str, phase_result: dict) -> None:
        """Save all models, predictions, and metadata for a phase."""
        phase_name = PHASES[phase_key]["name"].replace(" ", "_").replace(":", "").lower()
        phase_dir = f"{OUT_DIR}/{phase_name}"
        os.makedirs(phase_dir, exist_ok=True)

        # Save models
        if SAVE_MODELS:
            for model_name, model in phase_result["models"].items():
                model_path = f"{phase_dir}/{model_name.replace(' ', '_').lower()}.joblib"
                joblib.dump(model, model_path)
                if self.verbose:
                    print(f"  [OK] Saved model: {model_path}")

        # Save metrics
        metrics_path = f"{phase_dir}/metrics.csv"
        phase_result["metrics_df"].to_csv(metrics_path, index=False)
        if self.verbose:
            print(f"  [OK] Saved metrics: {metrics_path}")

        # Save metadata
        import json
        metadata_path = f"{phase_dir}/metadata.json"
        metadata = {
            "phase": phase_key,
            "phase_name": PHASES[phase_key]["name"],
            "timestamp": self.timestamp,
            "data": phase_result["data_metadata"],
            "preprocessing": phase_result["prep_data"].metadata,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        if self.verbose:
            print(f"  [OK] Saved metadata: {metadata_path}")

        # Save best model predictions
        evaluator = phase_result["evaluator"]
        best_model_name = phase_result["metrics_df"].iloc[0]["Model"]
        if best_model_name in evaluator.predictions:
            best_proba = evaluator.predictions[best_model_name]
            pred_df = pd.DataFrame({
                "Model": best_model_name,
                "Predicted_Probability": best_proba,
                "Predicted_Default": (best_proba >= 0.5).astype(int),
            })
            pred_path = f"{phase_dir}/predictions_best_model.csv"
            pred_df.to_csv(pred_path, index=False)
            if self.verbose:
                print(f"  [OK] Saved predictions: {pred_path}")

        # Save rationale for selected model
        top = phase_result["metrics_df"].iloc[0]
        rationale_lines = [
            f"Phase: {PHASES[phase_key]['name']}",
            f"Selected model: {top['Model']}",
            f"Primary selection metric: AUC={top['AUC']:.6f}",
            f"Supporting metrics: F1={top['F1']:.6f}, Precision={top['Precision']:.6f}, Recall={top['Recall']:.6f}, KS={top['KS']:.6f}",
            f"Optimal F1 threshold (test split): {top.get('Optimal_Threshold', 0.5):.6f}",
            f"F1 at optimal threshold: {top.get('F1_Optimal', top['F1']):.6f}",
            "Rule used: choose highest AUC; break ties by F1 then KS.",
        ]
        rationale_path = f"{phase_dir}/best_model_why.txt"
        with open(rationale_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rationale_lines))
        if self.verbose:
            print(f"  [OK] Saved selection rationale: {rationale_path}")

        # Save AZS/HCRL audit outputs in phase 1 for transparency
        if phase_key == "PHASE_1_HCRL":
            audit_df = build_azs_hcrl_audit_table(include_esg=False)
            audit_path = f"{phase_dir}/azs_hcrl_audit_all_companies.csv"
            audit_df.to_csv(audit_path, index=False)
            if self.verbose:
                print(f"  [OK] Saved AZS/HCRL audit table: {audit_path}")

    def run_all_phases(self, include_esg_phase: bool = ENABLE_ESG_PHASE) -> dict:
        """Run all configured phases end-to-end."""
        print("\n" + "=" * 80)
        print(" " * 20 + "UNIFIED CREDIT RISK PIPELINE")
        if include_esg_phase:
            print(" " * 15 + "Phases: HCRL + AZS + ESG")
        else:
            print(" " * 16 + "Phases: HCRL + AZS (No ESG)")
        print("=" * 80 + "\n")

        all_results = {}

        # Phase 1: HCRL
        phase1_result = self.run_phase("PHASE_1_HCRL")
        if phase1_result:
            all_results["PHASE_1_HCRL"] = phase1_result

        # Phase 2: AZS Baseline
        phase2_result = self.run_phase("PHASE_2_AZS")
        if phase2_result:
            all_results["PHASE_2_AZS"] = phase2_result

        # Optional Phase 3: ESG Augmentation
        if include_esg_phase:
            phase3_result = self.run_phase("PHASE_3_ESG")
            if phase3_result:
                all_results["PHASE_3_ESG"] = phase3_result

        # Generate summary report
        self._generate_summary_report(all_results)

        print("\n" + "=" * 80)
        print(" " * 20 + "[OK] ALL PHASES COMPLETE")
        print("=" * 80 + "\n")

        return all_results

    def _generate_summary_report(self, all_results: dict) -> None:
        """Generate high-level summary comparing all phases."""
        print("\n" + "=" * 80)
        print("PHASE COMPARISON SUMMARY")
        print("=" * 80)

        summary_rows = []
        for phase_key, result in all_results.items():
            if result is None:
                continue

            phase_name = PHASES[phase_key]["name"]
            metrics_df = result["metrics_df"]
            best_row = metrics_df.iloc[0]

            summary_rows.append({
                "Phase": phase_name,
                "Best Model": best_row["Model"],
                "AUC": best_row["AUC"],
                "F1": best_row["F1"],
                "Precision": best_row["Precision"],
                "Recall": best_row["Recall"],
            })

        summary_df = pd.DataFrame(summary_rows)
        print("\n" + summary_df.to_string(index=False))

        # Save summary
        summary_path = f"{OUT_DIR}/summary_comparison.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[Saved] Summary > {summary_path}")

        # Save summary plot for quick visual comparison.
        if not summary_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(summary_df["Phase"], summary_df["AUC"], color=["#4e79a7", "#f28e2b", "#59a14f"][:len(summary_df)])
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("Best Model AUC")
            ax.set_title("Phase-wise Best Model AUC Comparison")
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=10)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            summary_plot_path = os.path.join(OUT_DIR, "summary_comparison_auc.png")
            plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[Saved] Summary plot > {summary_plot_path}")

        # Save additional ESG-vs-non-ESG comparison visualizations.
        try:
            comparison_outputs = generate_comparison_visualizations(OUT_DIR)
            if comparison_outputs:
                print(f"[Saved] Comparison visualizations > {OUT_DIR}/comparison_visualizations")
        except Exception as err:
            print(f"[WARN] Could not generate comparison visualizations: {err}")


def main():
    """Main entry point."""
    pipeline = CreditRiskPipeline(verbose=VERBOSE)
    results = pipeline.run_all_phases()
    return results


if __name__ == "__main__":
    main()
