"""
Cross-phase comparison visualizations for ESG impact analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _phase_dir_name(phase_name: str) -> str:
    return phase_name.replace(" ", "_").replace(":", "").lower()


def _annotate_heatmap(ax, matrix: np.ndarray, fmt: str = ".4f") -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                text = "NA"
            else:
                text = format(float(value), fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)


def generate_comparison_visualizations(out_dir: str) -> Dict[str, str]:
    """Generate ESG-vs-non-ESG comparison artifacts from saved pipeline outputs."""
    base = Path(out_dir)
    viz_dir = base / "comparison_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    summary_path = base / "summary_comparison.csv"
    summary_df = _safe_read_csv(summary_path)

    phase1_dir = base / "phase_1_hcrl_construction_&_baseline_models"
    phase3_dir = base / "phase_3_esg_augmentation"
    phase1 = _safe_read_csv(phase1_dir / "model_comparison.csv")
    phase3 = _safe_read_csv(phase3_dir / "model_comparison.csv")

    output_map: Dict[str, str] = {}

    if not summary_df.empty:
        target_phases = [
            "Phase 1: HCRL Construction & Baseline Models",
            "Phase 3: ESG Augmentation",
        ]
        compare = summary_df[summary_df["Phase"].isin(target_phases)].copy()

        if len(compare) == 2:
            metrics = ["AUC", "F1", "Precision", "Recall"]
            x = np.arange(len(metrics))
            width = 0.36

            p1 = compare[compare["Phase"] == target_phases[0]].iloc[0]
            p3 = compare[compare["Phase"] == target_phases[1]].iloc[0]

            p1_vals = [p1[m] for m in metrics]
            p3_vals = [p3[m] for m in metrics]

            fig, ax = plt.subplots(figsize=(11, 6))
            ax.bar(x - width / 2, p1_vals, width, label="Without ESG (Phase 1)", color="#4e79a7")
            ax.bar(x + width / 2, p3_vals, width, label="With ESG (Phase 3)", color="#59a14f")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title("Best Model Metrics: With ESG vs Without ESG", fontweight="bold")
            ax.grid(alpha=0.25, axis="y")
            ax.legend()
            fig.tight_layout()

            out_path = viz_dir / "best_model_with_vs_without_esg.png"
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            output_map["best_model_with_vs_without_esg"] = str(out_path)

    if not phase1.empty and not phase3.empty:
        merged = phase1.merge(
            phase3,
            on="Model",
            how="outer",
            suffixes=("_No_ESG", "_With_ESG"),
        )

        merged = merged.sort_values("Model").reset_index(drop=True)
        metrics_for_delta = ["AUC", "F1", "KS", "Precision", "Recall", "Brier", "AP"]
        for metric in metrics_for_delta:
            left = f"{metric}_No_ESG"
            right = f"{metric}_With_ESG"
            if left in merged.columns and right in merged.columns:
                merged[f"Delta_{metric}"] = merged[right] - merged[left]

        merged.to_csv(viz_dir / "with_vs_without_esg_model_metrics.csv", index=False)
        output_map["with_vs_without_esg_model_metrics"] = str(viz_dir / "with_vs_without_esg_model_metrics.csv")

        # Heatmap for AUC and F1 across phase 1 and phase 3
        heat_cols = ["AUC_No_ESG", "AUC_With_ESG", "F1_No_ESG", "F1_With_ESG", "KS_No_ESG", "KS_With_ESG"]
        available = [c for c in heat_cols if c in merged.columns]
        heat_df = merged[["Model"] + available].copy()

        matrix = heat_df[available].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(available)))
        ax.set_xticklabels(available, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(heat_df)))
        ax.set_yticklabels(heat_df["Model"].tolist())
        ax.set_title("Model Metrics Heatmap: No ESG vs With ESG", fontweight="bold")
        _annotate_heatmap(ax, matrix)
        fig.tight_layout()

        out_path = viz_dir / "model_metrics_heatmap_with_vs_without_esg.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        output_map["model_metrics_heatmap_with_vs_without_esg"] = str(out_path)

        # Delta chart (With ESG - No ESG)
        delta_cols = [c for c in ["Delta_AUC", "Delta_F1", "Delta_KS", "Delta_Precision", "Delta_Recall"] if c in merged.columns]
        if delta_cols:
            delta_long = merged[["Model"] + delta_cols].set_index("Model")

            fig, ax = plt.subplots(figsize=(13, 7))
            x = np.arange(len(delta_long.index))
            width = 0.14
            for i, col in enumerate(delta_cols):
                ax.bar(x + (i - (len(delta_cols) - 1) / 2) * width, delta_long[col].values, width, label=col)

            ax.axhline(0, color="black", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(delta_long.index, rotation=25, ha="right")
            ax.set_title("ESG Impact by Model (With ESG - No ESG)", fontweight="bold")
            ax.set_ylabel("Delta")
            ax.grid(alpha=0.25, axis="y")
            ax.legend(ncol=3)
            fig.tight_layout()

            out_path = viz_dir / "esg_impact_delta_by_model.png"
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            output_map["esg_impact_delta_by_model"] = str(out_path)

    return output_map
