"""
=============================================================================
HYPOTHESIS TESTING & LEAKAGE AUDIT
=============================================================================
DeLong AUC tests, McNemar error comparison, permutation baseline tests,
single-feature leakage scan, and reproducibility documentation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os


# ── 1. DELONG AUC TEST ───────────────────────────────────────────────────────
# Standard test in credit risk literature (DeLong et al., 1988).

def delong_auc_test(y_true: np.ndarray,
                    prob_a: np.ndarray,
                    prob_b: np.ndarray) -> dict:
    """
    Two-sided DeLong test comparing AUC between two models.
    
    Returns significance of AUC difference using robust covariance estimation.
    p < 0.05 → difference is statistically significant.
    """
    def _compute_midrank(x):
        sorted_x = np.sort(x)
        sorted_idx = np.argsort(x)
        n = len(x)
        midranks = np.zeros(n)
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_x[j] == sorted_x[j + 1]:
                j += 1
            midranks[sorted_idx[i:j+1]] = (i + j + 2) / 2
            i = j + 1
        return midranks

    def _structural_components(y_true, prob):
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)

        v_pos = np.zeros(n_pos)
        v_neg = np.zeros(n_neg)

        for i, pi in enumerate(pos_idx):
            v_pos[i] = np.mean(prob[pi] > prob[neg_idx]) + \
                       0.5 * np.mean(prob[pi] == prob[neg_idx])

        for j, ni in enumerate(neg_idx):
            v_neg[j] = np.mean(prob[ni] < prob[pos_idx]) + \
                       0.5 * np.mean(prob[ni] == prob[pos_idx])

        return v_pos, v_neg

    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n1, n0 = len(pos_idx), len(neg_idx)

    va_pos, va_neg = _structural_components(y_true, prob_a)
    vb_pos, vb_neg = _structural_components(y_true, prob_b)

    # Covariance computations
    s01_aa = np.cov(va_pos, va_pos)[0, 1] if n1 > 1 else 0
    s10_aa = np.cov(va_neg, va_neg)[0, 1] if n0 > 1 else 0
    s01_bb = np.cov(vb_pos, vb_pos)[0, 1] if n1 > 1 else 0
    s10_bb = np.cov(vb_neg, vb_neg)[0, 1] if n0 > 1 else 0
    s01_ab = np.cov(va_pos, vb_pos)[0, 1] if n1 > 1 else 0
    s10_ab = np.cov(va_neg, vb_neg)[0, 1] if n0 > 1 else 0

    var_a = s01_aa / n1 + s10_aa / n0
    var_b = s01_bb / n1 + s10_bb / n0
    cov_ab = s01_ab / n1 + s10_ab / n0

    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        return {
            "AUC_A": round(auc_a, 4), "AUC_B": round(auc_b, 4),
            "AUC_diff": round(auc_a - auc_b, 4),
            "z_stat": 0.0, "p_value": 1.0, "significant": False,
        }

    z_stat = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "AUC_A": round(auc_a, 4),
        "AUC_B": round(auc_b, 4),
        "AUC_diff": round(auc_a - auc_b, 4),
        "z_stat": round(z_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def run_all_delong_comparisons(
    y_true: np.ndarray,
    model_probas: dict,
    output_dir: str,
    phase_name: str = "Phase 1",
) -> pd.DataFrame:
    """Runs DeLong test for every pair of models."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    model_names = list(model_probas.keys())
    n_comparisons = len(list(combinations(model_names, 2)))

    print(f"\n  DeLong AUC Comparisons ({n_comparisons} pairs):")

    for name_a, name_b in combinations(model_names, 2):
        result = delong_auc_test(y_true, model_probas[name_a], model_probas[name_b])
        result["Model_A"] = name_a
        result["Model_B"] = name_b
        result["p_bonferroni"] = round(
            min(result["p_value"] * n_comparisons, 1.0), 4
        )
        result["significant_bonf"] = result["p_bonferroni"] < 0.05
        records.append(result)

    df = pd.DataFrame(records)[
        ["Model_A", "Model_B", "AUC_A", "AUC_B", "AUC_diff",
         "z_stat", "p_value", "significant", "p_bonferroni", "significant_bonf"]
    ].sort_values("AUC_diff", ascending=False)

    df.to_csv(os.path.join(output_dir, "delong_test_results.csv"), index=False)
    _plot_delong_heatmap(df, model_names, output_dir, phase_name)

    n_sig = df["significant"].sum()
    print(f"    {n_sig}/{len(df)} pairs significant at p<0.05 "
          f"({df['significant_bonf'].sum()} after Bonferroni correction)")

    return df


def _plot_delong_heatmap(df, model_names, output_dir, phase_name):
    """Heatmap of DeLong p-values and AUC differences."""
    try:
        n = len(model_names)
        pmat = pd.DataFrame(np.ones((n, n)), index=model_names, columns=model_names)
        dmat = pd.DataFrame(np.zeros((n, n)), index=model_names, columns=model_names)

        for _, row in df.iterrows():
            pmat.loc[row["Model_A"], row["Model_B"]] = row["p_value"]
            pmat.loc[row["Model_B"], row["Model_A"]] = row["p_value"]
            dmat.loc[row["Model_A"], row["Model_B"]] = row["AUC_diff"]
            dmat.loc[row["Model_B"], row["Model_A"]] = -row["AUC_diff"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(dmat, ax=axes[0], cmap="RdBu_r", center=0,
                    annot=True, fmt=".4f", linewidths=0.3,
                    cbar_kws={"label": "AUC_A − AUC_B"})
        axes[0].set_title(f"{phase_name}: AUC difference (DeLong)")

        sns.heatmap(pmat, ax=axes[1], cmap="YlOrRd_r", vmin=0, vmax=0.2,
                    annot=True, fmt=".3f", linewidths=0.3,
                    cbar_kws={"label": "p-value"})
        axes[1].set_title(f"{phase_name}: DeLong p-values (* = p<0.05)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "delong_heatmap.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close()
    except Exception as e:
        pass


# ── 2. McNEMAR TEST ──────────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray,
                 pred_a: np.ndarray,
                 pred_b: np.ndarray) -> dict:
    """
    Tests whether two models make systematically different errors.
    If b01 ≠ b10 → models disagree on error patterns.
    """
    err_a = (pred_a != y_true)
    err_b = (pred_b != y_true)

    b01 = np.sum(~err_a & err_b)   # A right, B wrong
    b10 = np.sum(err_a & ~err_b)   # A wrong, B right
    n_discordant = b01 + b10

    if n_discordant == 0:
        return {
            "b01": 0, "b10": 0, "chi2": 0.0, "p_value": 1.0,
            "significant": False, "note": "No discordant errors"
        }

    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = stats.chi2.sf(chi2, df=1)

    return {
        "b01": int(b01),
        "b10": int(b10),
        "n_discordant": int(n_discordant),
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def run_all_mcnemar_comparisons(
    y_true: np.ndarray,
    model_preds: dict,
    output_dir: str,
    phase_name: str = "Phase 1",
) -> pd.DataFrame:
    """Runs McNemar test for all model pairs."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    print(f"\n  McNemar Error Comparisons:")

    for name_a, name_b in combinations(model_preds.keys(), 2):
        result = mcnemar_test(y_true, model_preds[name_a], model_preds[name_b])
        result["Model_A"] = name_a
        result["Model_B"] = name_b
        records.append(result)

    df = pd.DataFrame(records).sort_values("p_value")
    df.to_csv(os.path.join(output_dir, "mcnemar_results.csv"), index=False)

    n_sig = df["significant"].sum()
    print(f"    {n_sig}/{len(df)} pairs significant at p<0.05")

    return df


# ── 3. SINGLE-FEATURE LEAKAGE SCAN ───────────────────────────────────────────

def scan_feature_auc(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: str,
    leakage_threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Trains LR on each feature individually.
    AUC > leakage_threshold suggests proxy leakage.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []
    scaler = StandardScaler()

    print(f"\n  Single-Feature Leakage Scan (threshold={leakage_threshold}):")

    for col in X_train.columns:
        x_tr = X_train[[col]].fillna(0).values
        x_te = X_test[[col]].fillna(0).values

        try:
            x_tr_s = scaler.fit_transform(x_tr)
            x_te_s = scaler.transform(x_te)

            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                                   class_weight="balanced")
            lr.fit(x_tr_s, y_train)
            prob = lr.predict_proba(x_te_s)[:, 1]
            auc = roc_auc_score(y_test, prob)
        except Exception:
            auc = np.nan

        records.append({
            "feature": col,
            "single_feat_AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
            "leak_suspect": auc > leakage_threshold if not np.isnan(auc) else False,
        })

    df = pd.DataFrame(records).sort_values("single_feat_AUC", ascending=False, 
                                           na_position="last")
    df.to_csv(os.path.join(output_dir, "single_feature_auc_scan.csv"), index=False)

    suspects = df[df["leak_suspect"]]
    if len(suspects) > 0:
        print(f"    ⚠ {len(suspects)} features EXCEED threshold:")
        for _, row in suspects.iterrows():
            print(f"      {row['feature']:30s}  AUC={row['single_feat_AUC']:.4f}")
    else:
        print(f"    ✓ No features exceed {leakage_threshold} threshold")

    _plot_feature_auc(df, output_dir, leakage_threshold)
    return df


def _plot_feature_auc(df, output_dir, threshold):
    """Bar plot of top features by single-feature AUC."""
    try:
        top = df.dropna().head(30)
        colors = ["#E24B4A" if s else "#185FA5" for s in top["leak_suspect"]]

        fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.28)))
        ax.barh(top["feature"][::-1], top["single_feat_AUC"][::-1],
                color=colors[::-1], height=0.65)
        ax.axvline(threshold, color="#E24B4A", linewidth=1.2,
                   linestyle="--", label=f"Leakage suspect threshold ({threshold})")
        ax.axvline(0.5, color="#888", linewidth=0.8, linestyle=":",
                   label="Random baseline (0.5)")
        ax.set_xlabel("Single-feature AUC (logistic regression)")
        ax.set_title("Leakage scan: features that independently predict HCRL")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "single_feature_auc_scan.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close()
    except Exception as e:
        pass


# ── 4. PERMUTATION BASELINE TEST ────────────────────────────────────────────

def permutation_baseline_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    model_name: str,
    output_dir: str,
    n_permutations: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Trains on randomly shuffled labels.
    Real AUC should be far above the permutation distribution.
    If real AUC is inside permutation dist → leakage or label construction issue.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    # Real AUC
    m = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
    m.fit(X_train, y_train)
    real_prob = m.predict_proba(X_test)[:, 1]
    real_auc = roc_auc_score(y_test, real_prob)

    # Permuted AUCs
    perm_aucs = []
    for i in range(n_permutations):
        y_shuffled = rng.permutation(y_train)
        try:
            m_perm = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
            m_perm.fit(X_train, y_shuffled)
            prob_perm = m_perm.predict_proba(X_test)[:, 1]
            perm_aucs.append(roc_auc_score(y_test, prob_perm))
        except Exception:
            perm_aucs.append(0.5)

    perm_aucs = np.array(perm_aucs)
    p_value = np.mean(perm_aucs >= real_auc)
    z_score = (real_auc - perm_aucs.mean()) / (perm_aucs.std() + 1e-9)

    result = {
        "model": model_name,
        "real_AUC": round(real_auc, 4),
        "perm_AUC_mean": round(perm_aucs.mean(), 4),
        "perm_AUC_std": round(perm_aucs.std(), 4),
        "perm_AUC_max": round(perm_aucs.max(), 4),
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n_permutations": n_permutations,
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(perm_aucs, bins=30, color="#185FA5", alpha=0.7,
            edgecolor="white", label=f"Permuted AUC (n={n_permutations})")
    ax.axvline(real_auc, color="#E24B4A", linewidth=2,
               label=f"Real AUC = {real_auc:.4f}")
    ax.axvline(perm_aucs.mean(), color="#888", linewidth=1, linestyle="--",
               label=f"Perm mean = {perm_aucs.mean():.4f}")
    ax.set_xlabel("AUC under shuffled labels")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name}: permutation baseline test  (z={z_score:.1f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"permutation_test_{model_name.replace(' ','_')}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()

    status = "✓ PASS" if p_value < 0.05 else "⚠ FAIL"
    print(f"\n  Permutation test {model_name}: {status}")
    print(f"    Real AUC:        {real_auc:.4f}")
    print(f"    Perm mean ± std: {perm_aucs.mean():.4f} ± {perm_aucs.std():.4f}")
    print(f"    z-score:         {z_score:.2f}")
    print(f"    p-value:         {p_value:.4f}")

    return result


# ── 5. REPRODUCIBILITY TABLE ────────────────────────────────────────────────

def generate_reproducibility_table(config, output_dir: str) -> pd.DataFrame:
    """
    Generates reproducibility table for report appendix.
    Lists all random seeds, library versions, split parameters.
    """
    os.makedirs(output_dir, exist_ok=True)

    records = [
        {"Category": "Random seed", "Parameter": "Global numpy seed",
         "Value": str(getattr(config, "RANDOM_STATE", 42))},
        {"Category": "Random seed", "Parameter": "Train-test split seed",
         "Value": str(getattr(config, "RANDOM_STATE", 42))},
        {"Category": "Random seed", "Parameter": "CV splitter seed",
         "Value": str(getattr(config, "RANDOM_STATE", 42))},
        {"Category": "Data split", "Parameter": "Test set fraction",
         "Value": str(getattr(config, "TEST_SIZE", 0.20))},
        {"Category": "Data split", "Parameter": "CV folds",
         "Value": str(getattr(config, "CV_FOLDS", 5))},
        {"Category": "Preprocessing", "Parameter": "Imputation strategy",
         "Value": "Median (fit on train only)"},
        {"Category": "Preprocessing", "Parameter": "Winsorization",
         "Value": "1st–99th percentile (fit on train only)"},
        {"Category": "Preprocessing", "Parameter": "Scaling",
         "Value": "StandardScaler (fit on train only)"},
        {"Category": "Preprocessing", "Parameter": "Resampling",
         "Value": "SMOTE+ENN (train only)"},
        {"Category": "Leakage control", "Parameter": "Pillar flags excluded",
         "Value": "AZS_Flag, PD_Ohlson, Market flags"},
        {"Category": "Leakage control", "Parameter": "Proxy scan threshold",
         "Value": "Single-feature AUC > 0.85"},
        {"Category": "Leakage control", "Parameter": "Permutation test",
         "Value": "200 permutations per best model"},
    ]

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "reproducibility_table.csv"), index=False)

    print(f"\n✓ Reproducibility table saved\n")
    return df
