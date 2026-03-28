"""
=============================================================================
ESG TALK-WALK GAP ANALYSIS
=============================================================================
Derives Talk-Walk gap features and analyzes their predictive power.
Greenwashing detection, gap-based ESG risk scoring, and ablation studies.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
from itertools import combinations


# ── 1. CORE GAP FEATURE ENGINEERING ──────────────────────────────────────────

def engineer_gap_features(
    df: pd.DataFrame,
    talk_col: str,
    walk_col: str,
    company_col: str = "Name",
    year_col: str = "fiscal_year",
) -> pd.DataFrame:
    """
    Derives all Talk-Walk gap features.
    
    Gap = Talk score - Walk score
    - Positive gap → greenwashing (claim > delivery)
    - Negative gap → under-reporting (delivery > claim)
    """
    df = df.copy()

    # Ensure columns exist and fillna with 0
    if talk_col not in df.columns or walk_col not in df.columns:
        print(f"Warning: {talk_col} or {walk_col} not found in dataframe.")
        return df

    df[talk_col] = pd.to_numeric(df[talk_col], errors="coerce").fillna(0)
    df[walk_col] = pd.to_numeric(df[walk_col], errors="coerce").fillna(0)

    # ── Basic gap signals ────────────────────────────────────────────────────
    df["gap_raw"]        = df[talk_col] - df[walk_col]      # signed gap
    df["gap_abs"]        = np.abs(df["gap_raw"])             # inconsistency magnitude
    df["gap_direction"]  = np.sign(df["gap_raw"])           # +1 (greenwash), -1 (underreport), 0 (aligned)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        df["gap_ratio"] = df[talk_col] / (df[walk_col] + 1e-6)
        df["gap_ratio"] = np.where(np.isinf(df["gap_ratio"]), 0, df["gap_ratio"])

    # Quartiles
    try:
        df["gap_quartile"] = pd.qcut(
            df["gap_raw"], q=4, labels=[0, 1, 2, 3], duplicates="drop"
        ).astype(float)
    except Exception:
        df["gap_quartile"] = 0

    # ── Greenwashing flag: top quartile of gap_raw ───────────────────────────
    top_q = df["gap_raw"].quantile(0.75)
    df["greenwashing_flag"] = (df["gap_raw"] >= top_q).astype(int)

    # ── Under-reporter flag: bottom quartile ─────────────────────────────────
    bot_q = df["gap_raw"].quantile(0.25)
    df["under_reporter_flag"] = (df["gap_raw"] <= bot_q).astype(int)

    # ── Consistency score: inverse of abs gap, normalized 0–1 ───────────────
    max_abs = df["gap_abs"].max()
    if max_abs > 0:
        df["esg_consistency"] = 1.0 - (df["gap_abs"] / max_abs)
    else:
        df["esg_consistency"] = 1.0

    print(f"\n✓ Gap features engineered")
    print(f"  Greenwashing flags: {df['greenwashing_flag'].sum()} companies "
          f"({df['greenwashing_flag'].mean():.1%})")
    print(f"  Under-reporter flags: {df['under_reporter_flag'].sum()} companies")

    return df


# ── 2. STANDALONE GAP PREDICTOR TEST ────────────────────────────────────────

def test_gap_as_standalone_predictor(
    df: pd.DataFrame,
    gap_features: list,
    target_col: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Tests each gap feature as a standalone predictor of HCRL.
    Answers: Does the gap independently predict default beyond ESG score?
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    y = df[target_col].fillna(0).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'='*70}")
    print(f"Standalone Gap Predictor Test")
    print(f"{'='*70}\n")

    for feat in gap_features:
        if feat not in df.columns:
            continue

        X_feat = df[[feat]].fillna(0).values
        scaler = StandardScaler()
        aucs = []

        for train_idx, val_idx in cv.split(X_feat, y):
            try:
                X_tr = scaler.fit_transform(X_feat[train_idx])
                X_val = scaler.transform(X_feat[val_idx])
                lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                                       class_weight="balanced")
                lr.fit(X_tr, y[train_idx])
                prob = lr.predict_proba(X_val)[:, 1]
                aucs.append(roc_auc_score(y[val_idx], prob))
            except Exception:
                continue

        if aucs:
            corr, pval = stats.pointbiserialr(y, df[feat].fillna(0).values)
            results.append({
                "feature":      feat,
                "CV_AUC":       round(np.mean(aucs), 4),
                "CV_AUC_std":   round(np.std(aucs), 4),
                "corr_HCRL":    round(corr, 4),
                "p_value":      round(pval, 4),
                "significant":  pval < 0.05,
            })

    results_df = pd.DataFrame(results).sort_values("CV_AUC", ascending=False)
    results_df.to_csv(
        os.path.join(output_dir, "gap_standalone_predictor_results.csv"), 
        index=False
    )

    print("Gap Standalone Predictor Results:")
    print(results_df[["feature", "CV_AUC", "p_value", "significant"]].to_string(index=False))
    print()

    return results_df


# ── 3. DEFAULT RATES BY GAP QUARTILE ────────────────────────────────────────

def compute_gap_quartile_default_rates(
    df: pd.DataFrame,
    gap_col: str,
    target_col: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Segments companies by gap quartile and computes observed default rates.
    Core research table: does high-gap (greenwashing) predict defaults?
    """
    os.makedirs(output_dir, exist_ok=True)

    temp = df[[gap_col, target_col]].dropna()
    if len(temp) == 0:
        print(f"  No data for {gap_col} quartile analysis.")
        return None

    try:
        temp["quartile"] = pd.qcut(
            temp[gap_col], q=4,
            labels=["Q1 (Low gap)", "Q2", "Q3", "Q4 (High gap)"],
            duplicates="drop"
        )
    except Exception:
        print(f"  Cannot create quartiles for {gap_col}.")
        return None

    table = (
        temp.groupby("quartile", observed=True)[target_col]
            .agg(n="count", n_distressed="sum", default_rate="mean")
            .reset_index()
    )
    table["default_rate_pct"] = (table["default_rate"] * 100).round(2)

    # Chi-square test
    contingency = pd.crosstab(temp["quartile"], temp[target_col])
    try:
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    except Exception:
        chi2, p_val = 0, 1

    table["chi2"] = round(chi2, 3)
    table["p_value"] = round(p_val, 4)

    table.to_csv(
        os.path.join(output_dir, f"default_rate_by_{gap_col}_quartile.csv"),
        index=False
    )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1D9E75", "#EF9F27", "#D85A30", "#E24B4A"]
    bars = ax.bar(
        range(len(table)),
        table["default_rate_pct"],
        color=colors[:len(table)],
        width=0.5,
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate with sample size
    for i, (bar, n) in enumerate(zip(bars, table["n"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"n={n}",
            ha="center", va="bottom", fontsize=9, color="#444"
        )

    sig_label = f"χ²={chi2:.2f}, p={p_val:.3f}" if p_val < 0.1 else "Not significant"
    ax.set_title(
        f"Default rate by {gap_col} quartile  ({sig_label})", fontsize=11
    )
    ax.set_xticklabels(table["quartile"].astype(str))
    ax.set_xlabel(f"{gap_col} quartile")
    ax.set_ylabel("Default rate (%)")
    ax.set_ylim(0, max(table["default_rate_pct"]) * 1.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"default_rate_{gap_col}_quartile.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Default rate analysis: {gap_col}")
    print(table[["quartile", "n", "default_rate_pct"]].to_string(index=False))

    return table


# ── 4. TALK-WALK SCATTER PLOT ────────────────────────────────────────────────

def plot_talk_walk_scatter(
    df: pd.DataFrame,
    talk_col: str,
    walk_col: str,
    target_col: str,
    output_dir: str,
    company_col: str = "Name",
    label_top_n: int = 5,
):
    """
    Scatter of Talk vs Walk, colored by distress status.
    Labels top-N distressed companies by gap.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_df = df[[talk_col, walk_col, target_col, "gap_raw",
                  company_col]].dropna()

    if len(plot_df) == 0:
        print(f"  No data for Talk-Walk scatter.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    lim_min = min(plot_df[talk_col].min(), plot_df[walk_col].min()) - 2
    lim_max = max(plot_df[talk_col].max(), plot_df[walk_col].max()) + 2

    # Perfect alignment diagonal
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "k--", linewidth=1, alpha=0.4, label="Perfect alignment")

    # Shade regions
    ax.fill_between([lim_min, lim_max], [lim_min, lim_max], lim_max,
                    alpha=0.04, color="#E24B4A", label="Greenwashing zone")

    # Scatter
    for distressed, color, marker, label in [
        (0, "#1D9E75", "o", "Non-distressed"),
        (1, "#E24B4A", "^", "Distressed (HCRL=1)"),
    ]:
        mask = plot_df[target_col] == distressed
        ax.scatter(
            plot_df.loc[mask, talk_col],
            plot_df.loc[mask, walk_col],
            c=color, marker=marker, s=55, alpha=0.75,
            edgecolors="white", linewidths=0.4,
            label=label,
        )

    # Label high-gap distressed
    distressed_df = plot_df[plot_df[target_col] == 1].nlargest(label_top_n, "gap_raw")
    for _, row in distressed_df.iterrows():
        ax.annotate(
            str(row[company_col])[:10],
            xy=(row[talk_col], row[walk_col]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8, color="#A32D2D",
        )

    ax.set_xlabel(f"Talk score ({talk_col})", fontsize=11)
    ax.set_ylabel(f"Walk score ({walk_col})", fontsize=11)
    ax.set_title("Talk vs Walk ESG scores by default status", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "talk_walk_scatter.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Talk-Walk scatter saved → {output_dir}")


# ── 5. GAP CORRELATION HEATMAP ───────────────────────────────────────────────

def plot_gap_correlation_heatmap(
    df: pd.DataFrame,
    gap_features: list,
    financial_features: list,
    target_col: str,
    output_dir: str,
):
    """
    Correlation matrix: gap features × top financial features × HCRL target.
    Shows whether gap captures something financials miss.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_cols = (
        [c for c in gap_features if c in df.columns] +
        [c for c in financial_features[:8] if c in df.columns] +
        [target_col]
    )

    if len(all_cols) < 3:
        print(f"  Insufficient columns for correlation heatmap.")
        return

    corr_df = df[all_cols].fillna(0).corr()

    fig, ax = plt.subplots(figsize=(max(10, len(all_cols) * 0.7),
                                    max(8, len(all_cols) * 0.6)))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)

    sns.heatmap(
        corr_df,
        ax=ax,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.3,
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.7},
    )

    ax.set_title("Gap features × financial features × HCRL: correlation", 
                 fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "gap_feature_correlation_heatmap.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Correlation heatmap saved → {output_dir}")
