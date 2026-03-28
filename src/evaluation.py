"""
=============================================================================
UNIFIED CREDIT RISK PIPELINE - EVALUATION
=============================================================================
Unified metrics calculation, comparison, and reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import os

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    precision_score, recall_score, f1_score, brier_score_loss,
    average_precision_score, classification_report
)
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from scipy.stats import ks_2samp
import math

from config import PALETTE, PLOT_STYLE, OUT_DIR, PLOTS_DIR
from preprocessing import PreprocessedData, get_cv_splitter
from model_training import UnifiedModelTrainer


class ModelEvaluator:
    """Unified model evaluation and comparison."""

    def __init__(
        self,
        prep_data: PreprocessedData,
        trainer: UnifiedModelTrainer,
        verbose: bool = True,
        output_dir: str = OUT_DIR,
        plots_dir: str = PLOTS_DIR,
    ):
        """
        Initialize evaluator.
        
        Args:
            prep_data: PreprocessedData object
            trainer: UnifiedModelTrainer with trained models
            verbose: Whether to print progress
        """
        self.prep_data = prep_data
        self.trainer = trainer
        self.verbose = verbose
        self.metrics_df = None
        self.predictions = {}
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def _optimal_f1_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Find threshold that maximizes F1 on the evaluated split."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        if len(thresholds) == 0:
            return {
                "Optimal_Threshold": 0.50,
                "Precision_Optimal": 0.0,
                "Recall_Optimal": 0.0,
                "F1_Optimal": 0.0,
            }

        f1_vals = (2 * precisions[:-1] * recalls[:-1]) / np.clip(precisions[:-1] + recalls[:-1], 1e-9, None)
        best_idx = int(np.argmax(f1_vals))
        return {
            "Optimal_Threshold": float(thresholds[best_idx]),
            "Precision_Optimal": float(precisions[best_idx]),
            "Recall_Optimal": float(recalls[best_idx]),
            "F1_Optimal": float(f1_vals[best_idx]),
        }

    def compute_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "AUC": roc_auc_score(y_true, y_pred_proba),
            "KS": ks_2samp(y_pred_proba[y_true == 1], y_pred_proba[y_true == 0]).statistic,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Brier": brier_score_loss(y_true, y_pred_proba),
            "AP": average_precision_score(y_true, y_pred_proba),
            "ECE": self._expected_calibration_error(y_true, y_pred_proba),
        }

        return metrics

    def _expected_calibration_error(self, y_true: np.ndarray, y_pred_proba: np.ndarray, bins: int = 10) -> float:
        """Compute expected calibration error (ECE). Lower is better."""
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        n = len(y_true)
        if n == 0:
            return 0.0

        for i in range(bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_pred_proba >= lo) & (y_pred_proba < hi) if i < bins - 1 else (y_pred_proba >= lo) & (y_pred_proba <= hi)
            if not np.any(mask):
                continue
            conf = float(np.mean(y_pred_proba[mask]))
            acc = float(np.mean(y_true[mask]))
            ece += (np.sum(mask) / n) * abs(acc - conf)

        return float(ece)

    def _cross_validated_metrics(self, model: Any) -> Dict[str, float]:
        """Compute CV summary metrics on original training split for robustness reporting."""
        X_train = self.prep_data.X_train
        y_train = self.prep_data.y_train
        cv = get_cv_splitter(y_train)
        scoring = {"auc": "roc_auc", "f1": "f1"}
        try:
            cv_out = cross_validate(
                clone(model),
                X_train,
                y_train,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                error_score=np.nan,
            )
            auc_vals = cv_out.get("test_auc", np.array([np.nan]))
            f1_vals = cv_out.get("test_f1", np.array([np.nan]))
            return {
                "CV_AUC_Mean": float(np.nanmean(auc_vals)),
                "CV_AUC_Std": float(np.nanstd(auc_vals)),
                "CV_F1_Mean": float(np.nanmean(f1_vals)),
                "CV_F1_Std": float(np.nanstd(f1_vals)),
            }
        except Exception:
            return {
                "CV_AUC_Mean": np.nan,
                "CV_AUC_Std": np.nan,
                "CV_F1_Mean": np.nan,
                "CV_F1_Std": np.nan,
            }

    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate all models and return comparison table."""
        if self.verbose:
            print("\n" + "=" * 70)
            print("STAGE: MODEL EVALUATION")
            print("=" * 70)

        X_test = self.prep_data.X_test
        y_test = self.prep_data.y_test

        results = []

        for model_name, model in self.trainer.get_all_models().items():
            if self.verbose:
                print(f"\n[Evaluating] {model_name}...")

            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics = self.compute_metrics(y_test.values, y_pred_proba)
                metrics.update(self._cross_validated_metrics(model))
                metrics.update(self._optimal_f1_metrics(y_test.values, y_pred_proba))
                metrics["Model"] = model_name

                # Store predictions
                self.predictions[model_name] = y_pred_proba

                results.append(metrics)

                if self.verbose:
                    print(
                        f"  AUC={metrics['AUC']:.4f} | F1={metrics['F1']:.4f} | "
                        f"KS={metrics['KS']:.4f} | ECE={metrics['ECE']:.4f} | "
                        f"CV_AUC={metrics['CV_AUC_Mean']:.4f}+/-{metrics['CV_AUC_Std']:.4f}"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] Evaluation failed: {e}")

        # Create comparison dataframe
        self.metrics_df = (
            pd.DataFrame(results)
            .sort_values(["AUC", "CV_AUC_Mean", "F1"], ascending=False)
            .reset_index(drop=True)
        )

        if self.verbose:
            print(f"\n{'='*70}")
            print("MODEL COMPARISON TABLE")
            print(f"{'='*70}")
            print(
                self.metrics_df[
                    ["Model", "AUC", "CV_AUC_Mean", "CV_AUC_Std", "F1", "Precision", "Recall", "KS", "Brier", "ECE"]
                ].to_string(index=False)
            )

        return self.metrics_df

    def generate_roc_curves(self) -> None:
        """Generate ROC curve comparison plot."""
        if not self.predictions:
            if self.verbose:
                print("[WARN] No predictions to plot")
            return

        y_test = self.prep_data.y_test

        fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])

        for i, (model_name, y_pred) in enumerate(self.predictions.items()):
            auc = roc_auc_score(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", linewidth=2, color=PALETTE[i % len(PALETTE)])

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.plots_dir, "roc_curves.png")
        plt.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
        plt.close()

        if self.verbose:
            print(f"[Saved] ROC curves → {output_path}")

    def generate_pr_curves(self) -> None:
        """Generate Precision-Recall curve comparison plot."""
        if not self.predictions:
            return

        y_test = self.prep_data.y_test

        fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])

        for i, (model_name, y_pred) in enumerate(self.predictions.items()):
            ap = average_precision_score(y_test, y_pred)
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            ax.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})", linewidth=2, color=PALETTE[i % len(PALETTE)])

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        output_path = os.path.join(self.plots_dir, "pr_curves.png")
        plt.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
        plt.close()

        if self.verbose:
            print(f"[Saved] PR curves → {output_path}")

    def generate_confusion_matrices(self) -> None:
        """Generate and save confusion matrices for each model."""
        y_test = self.prep_data.y_test

        for model_name, y_pred_proba in self.predictions.items():
            y_pred = (y_pred_proba >= 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
            ax.figure.colorbar(im, ax=ax)

            classes = ['Normal', 'Default']
            ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes)

            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

            ax.set_ylabel("True Label", fontsize=11)
            ax.set_xlabel("Predicted Label", fontsize=11)
            ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")

            plt.tight_layout()
            output_path = os.path.join(self.plots_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
            plt.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
            plt.close()

    def save_metrics_table(self) -> None:
        """Save metrics comparison table to CSV."""
        if self.metrics_df is not None:
            output_path = os.path.join(self.output_dir, "model_comparison.csv")
            self.metrics_df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"[Saved] Metrics table → {output_path}")

    def generate_all_reports(self) -> None:
        """Generate all evaluation plots and reports."""
        self.generate_roc_curves()
        self.generate_pr_curves()
        self.generate_confusion_matrices()
        self.save_metrics_table()

        if self.verbose:
            print(f"\n[OK] Evaluation complete. Results saved to {self.output_dir}")
