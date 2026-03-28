"""
ESG Visualization Suite
=======================
Generates a broad set of ESG integration visualizations from Talk-Walk output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COLORS = {
    "E": "#2a9d8f",
    "S": "#e9c46a",
    "G": "#264653",
}


def _ensure_output_dir(output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _normalize_flags(series: pd.Series) -> pd.Series:
    expanded: List[str] = []
    for raw in series.fillna("No Risk").astype(str):
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        if not parts:
            parts = ["No Risk"]
        expanded.extend(parts)
    return pd.Series(expanded)


def plot_score_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "esg_score_distribution.png"
    scores = pd.to_numeric(df["Final_ESG_Score"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=25, color="#4e79a7", alpha=0.85, edgecolor="white")
    ax.axvline(scores.mean(), color="#e15759", linestyle="--", linewidth=2, label=f"Mean: {scores.mean():.2f}")
    ax.axvline(scores.median(), color="#59a14f", linestyle="--", linewidth=2, label=f"Median: {scores.median():.2f}")
    ax.set_title("Final ESG Score Distribution", fontweight="bold")
    ax.set_xlabel("Final ESG Score")
    ax.set_ylabel("Company Count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_confidence_vs_score(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "confidence_vs_esg_score.png"
    order = ["Low", "Medium", "High"]
    bucketed = [
        pd.to_numeric(df.loc[df["Confidence_Level"] == level, "Final_ESG_Score"], errors="coerce").dropna().values
        for level in order
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(bucketed, labels=order, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#f28e2b", "#edc948", "#59a14f"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title("Final ESG Score by Confidence Level", fontweight="bold")
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Final ESG Score")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_risk_flag_counts(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "risk_flag_counts.png"
    flags = _normalize_flags(df["Risk_Flag"]).value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    flags.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#e15759")
    ax.set_title("Risk Flag Frequency", fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("Risk Flag")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_talk_walk_final_components(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "talk_walk_final_component_means.png"

    means = {
        "Talk": [df["Talk_E"].mean(), df["Talk_S"].mean(), df["Talk_G"].mean()],
        "Walk": [df["Walk_E"].mean(), df["Walk_S"].mean(), df["Walk_G"].mean()],
        "Final": [df["Final_E"].mean(), df["Final_S"].mean(), df["Final_G"].mean()],
    }

    labels = ["E", "S", "G"]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, means["Talk"], width, label="Talk", color="#4e79a7")
    ax.bar(x, means["Walk"], width, label="Walk", color="#f28e2b")
    ax.bar(x + width, means["Final"], width, label="Final", color="#59a14f")

    ax.set_title("Average Talk, Walk and Final Components", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("ESG Pillar")
    ax.set_ylabel("Average Score")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_talk_vs_walk_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "talk_vs_walk_scatter.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)
    for i, pillar in enumerate(["E", "S", "G"]):
        talk_col = f"Talk_{pillar}"
        walk_col = f"Walk_{pillar}"
        axes[i].scatter(df[talk_col], df[walk_col], s=24, alpha=0.55, color=DEFAULT_COLORS[pillar])
        axes[i].plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="black")
        axes[i].set_title(f"{pillar}: Talk vs Walk")
        axes[i].set_xlabel("Talk")
        axes[i].set_ylabel("Walk")
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].set_ylim(-1.05, 1.05)
        axes[i].grid(alpha=0.25)

    fig.suptitle("Talk-Walk Alignment by ESG Pillar", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_gap_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "gap_distribution_by_pillar.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, pillar in enumerate(["E", "S", "G"]):
        col = f"Gap_{pillar}"
        axes[i].hist(df[col], bins=25, color=DEFAULT_COLORS[pillar], alpha=0.85, edgecolor="white")
        axes[i].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[i].axvline(0.5, color="#e15759", linestyle=":", linewidth=2)
        axes[i].set_title(f"Gap Distribution: {pillar}")
        axes[i].set_xlabel("Talk - Walk")
        axes[i].set_ylabel("Count")
        axes[i].grid(alpha=0.25)

    fig.suptitle("Greenwashing Gap Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_top_bottom_companies(df: pd.DataFrame, output_dir: Path, n: int = 10) -> Path:
    path = output_dir / "top_bottom_esg_companies.png"

    ranked = df[["Company_Name", "Final_ESG_Score"]].dropna().sort_values("Final_ESG_Score", ascending=False)
    top = ranked.head(n).copy()
    bottom = ranked.tail(n).copy().sort_values("Final_ESG_Score", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].barh(top["Company_Name"], top["Final_ESG_Score"], color="#59a14f")
    axes[0].set_title(f"Top {n} Final ESG Scores")
    axes[0].set_xlabel("Final ESG Score")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.25, axis="x")

    axes[1].barh(bottom["Company_Name"], bottom["Final_ESG_Score"], color="#e15759")
    axes[1].set_title(f"Bottom {n} Final ESG Scores")
    axes[1].set_xlabel("Final ESG Score")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "esg_correlation_heatmap.png"

    cols = [
        "Talk_E", "Talk_S", "Talk_G",
        "Walk_E", "Walk_S", "Walk_G",
        "Gap_E", "Gap_S", "Gap_G",
        "Final_E", "Final_S", "Final_G",
        "Final_ESG_Score",
    ]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdYlBu", vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)
    ax.set_title("ESG Feature Correlation Heatmap", fontweight="bold")

    # Keep text sparse to reduce clutter.
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.5:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_score_quantiles(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "esg_score_quantile_bands.png"

    scores = pd.to_numeric(df["Final_ESG_Score"], errors="coerce").dropna().sort_values().reset_index(drop=True)
    x = np.arange(1, len(scores) + 1)

    q20, q40, q60, q80 = np.percentile(scores, [20, 40, 60, 80])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, scores.values, color="#4e79a7", linewidth=2)
    ax.axhline(q20, color="#e15759", linestyle="--", linewidth=1.5, label=f"20th: {q20:.2f}")
    ax.axhline(q40, color="#f28e2b", linestyle="--", linewidth=1.5, label=f"40th: {q40:.2f}")
    ax.axhline(q60, color="#76b7b2", linestyle="--", linewidth=1.5, label=f"60th: {q60:.2f}")
    ax.axhline(q80, color="#59a14f", linestyle="--", linewidth=1.5, label=f"80th: {q80:.2f}")

    ax.set_title("Final ESG Score Curve with Quantile Bands", fontweight="bold")
    ax.set_xlabel("Companies (sorted by score)")
    ax.set_ylabel("Final ESG Score")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def generate_esg_visualizations(df: pd.DataFrame, output_dir: Path | str) -> Dict[str, str]:
    """Generate a comprehensive ESG visualization bundle and return artifact paths."""
    out = _ensure_output_dir(output_dir)

    chart_paths = {
        "score_distribution": plot_score_distribution(df, out),
        "confidence_vs_score": plot_confidence_vs_score(df, out),
        "risk_flag_counts": plot_risk_flag_counts(df, out),
        "talk_walk_final_components": plot_talk_walk_final_components(df, out),
        "talk_vs_walk_scatter": plot_talk_vs_walk_scatter(df, out),
        "gap_distribution": plot_gap_distribution(df, out),
        "top_bottom_companies": plot_top_bottom_companies(df, out),
        "correlation_heatmap": plot_correlation_heatmap(df, out),
        "score_quantiles": plot_score_quantiles(df, out),
    }

    summary = {
        "total_companies": int(len(df)),
        "mean_final_esg_score": float(pd.to_numeric(df["Final_ESG_Score"], errors="coerce").mean()),
        "median_final_esg_score": float(pd.to_numeric(df["Final_ESG_Score"], errors="coerce").median()),
        "high_confidence_share": float((df["Confidence_Level"] == "High").mean()),
    }
    pd.DataFrame([summary]).to_csv(out / "visualization_summary.csv", index=False)

    return {k: str(v) for k, v in chart_paths.items()}
