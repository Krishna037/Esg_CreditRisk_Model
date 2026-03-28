# Quick Reference: What Each Output File Means

## SHAP Explainability Outputs

| File | Purpose | How to Interpret |
|------|---------|------------------|
| `shap_beeswarm.png` | Global feature importance with scatter | Each dot = one company. Red dots on right = feature pushes that company toward default. Height of scatter = how much values vary. |
| `shap_bar.png` | Mean \|SHAP\| ranking (top 15) | Bars = average feature importance across all predictions. Use for "Top drivers of credit risk" statement. |
| `shap_waterfall_high_risk.png` | Why a distressed company was flagged | Base value + each feature's push up/down = final prediction. Shows feature interactions for that specific company. |
| `shap_waterfall_low_risk.png` | Why a healthy company was cleared | Opposite of high_risk — shows protective factors. |
| `shap_waterfall_borderline.png` | Why a marginal company was barely flagged/cleared | Most interesting for "explain the edge cases" narrative. |
| `shap_cross_model_heatmap.png` | Do all 7 models agree on feature importance? | If CatBoost and XGBoost both show Asset_Efficiency as #1, reproducible. If they disagree, model-dependent. |
| `shap_esg_delta.png` (Phase 3) | Did ESG features rise/fall in importance? | Green bars = ESG features that became MORE important with Phase 3. If none rose significantly, ESG is just noise. |

---

## Calibration Outputs

| File | Purpose | How to Interpret |
|------|---------|------------------|
| `calibration_metrics.csv` | Brier / ECE before and after scaling | Brier < 0.2 and ECE < 0.05 = well-calibrated. Delta > 0.02 means calibration meaningfully improved. |
| `calibration_curves_all.png` | Did we fix overconfidence/underconfidence? | Points on diagonal = perfect calibration. Below diagonal = overconfident (predicted 60%, actually 40%). |
| `calibration_before_after_[Model].png` | Visual proof that model X improved | Shows model didn't calibrate well naturally. Calibration pulled it toward diagonal. |
| `cv_holdout_results.csv` | Do models generalize within folds? | CV_AUC_std < 0.05 = stable. CV_AUC_std > 0.1 = unstable model (may overfit some folds). |

---

## Hypothesis Testing & Leakage Audit Outputs

| File | Purpose | How to Interpret |
|------|---------|------------------|
| `delong_test_results.csv` | Are model AUCs significantly different? | p_value < 0.05 (significant column) = this AUC difference is real, not noise. |
| `delong_heatmap.png` | Which model pairs differ significantly? | Dark red = significant p-value. If all gaps are white/light = all models perform similarly. |
| `mcnemar_results.csv` | Do models make different ERRORS? | b01 > b10 = Model A makes fewer errors than B. Large chi2 + p<0.05 = error patterns significantly differ. |
| `permutation_test_[Model].png` | Is AUC above chance? | Real AUC as tall red line. Histogram = permuted AUCs. If red line far right = **PASS**. If inside histogram = **FAIL (leakage suspected)**. |
| `single_feature_auc_scan.csv` | Which features are proxy leaks? | Single_feat_AUC > 0.85 and in red = exclude this column (it reconstructs the target). |
| `single_feature_auc_scan.png` | Visual flagging of suspect features | Red bars = must remove. Tall green/blue bars (0.60-0.70) = informative features working as intended. |
| `reproducibility_table.csv` | Can others replicate your results? | Include in paper appendix. Shows you used fixed seeds, stratified splits, documented versions. |

---

## ESG Gap Analysis Outputs (Phase 3 Only)

| File | Purpose | How to Interpret |
|------|---------|------------------|
| `gap_standalone_predictor_results.csv` | Does gap independently predict default? | CV_AUC > 0.56 + p_value < 0.05 = gap **adds information** beyond ESG score alone. |
| `default_rate_by_gap_raw_quartile.csv` | Quartile default rates + chi-square test | **Core research table**. Q1 default 5%, Q4 default 10%, χ²=6.2, p=0.01 = gap MATTERS. |
| `default_rate_gap_raw_quartile.png` | Visual: Do high-gap companies default more? | Upward trend = greenwashing (high gap) predicts distress. Flat = gap is irrelevant. |
| `talk_walk_scatter.png` | Visual story of ESG alignment | Upper triangle (above diagonal) = greenwashing. Lower triangle = humility. Red triangles in greenwashing zone = distressed greenwashers. |
| `gap_feature_correlation_heatmap.png` | Is gap collinear with financials? | If gap_raw highly correlated (r>0.7) with Debt_Equity, it's not independent. If r<0.3, it's a new signal. |

---

## What to Report in Your Capstone

### Must Include:
1. ✅ **SHAP beeswarm** (Figure 1) — "What are the top credit risk drivers?"
2. ✅ **Calibration curves** (Figure 2) — "Are predicted probabilities trustworthy?"
3. ✅ **Permutation test** (Figure 3) — "Is AUC real or just noise?"
4. ✅ **DeLong heatmap** (Figure 4) — "Do different models capture different risk?"
5. ✅ **Default rate by gap quartile** (Phase 3 Figure 5) — "Does ESG gap predict stress?"

### Should Include (Appendix):
- Reproducibility table (all seeds + versions)
- DeLong detailed results (21 model pairs)
- Leakage scan (top 20 features by AUC)
- Gap standalone predictor results
- Calibration metrics before/after

### Optional (Supplementary):
- All 7 SHAP beeswarms (if space)
- McNemar test results (detailed)
- CV holdout stability metrics

---

## Red Flags to Watch For

| Warning Sign | What It Means | Action |
|------|---------|---------|
| **SHAP beeswarm show = unrelated features as #1** | Features don't align with credit risk theory | Check feature engineering — may have target-proxy leakage |
| **Calibration curves far below diagonal** | Model systematically overestimates risk | Use calibration scaling — improves decision-making |
| **Permutation test: red line INSIDE histogram** | AUC not better than random labeling | **CRITICAL**: Major leakage problem. Investigate single-feature scan. |
| **Single-feature AUC > 0.85 for weird columns** | Proxy leak detected | Remove that column and retrain. Don't include in final report. |
| **DeLong all-red heatmap (many p<0.05)** | Models disagree a lot | OK if some differ. If ALL different, ensemble might not help. |
| **Default rate by gap: FLAT line across quartiles** | Gap feature doesn't predict risk | ESG contribution may be minimal. Note it frankly in discussion. |

---

## File Naming Convention

All files follow pattern: `[analysis_type]_[descriptor].[png|csv]`

Examples:
- `shap_beeswarm.png` → SHAP type, beeswarm style
- `calibration_curves_all.png` → Calibration type, all models
- `delong_test_results.csv` → DeLong analysis, test results
- `default_rate_gap_raw_quartile.png` → Default rate analysis, by gap_raw quartile

This makes outputs **self-documenting** — you can infer content from filename alone.

---

## Processing Pipeline Execution Order

```
run_pipeline.py --with-esg
├── Phase 1: HCRL baseline
│   ├── Model training (7 models)
│   └── Credibility suite:
│       ├── SHAP analysis → 7×4 = 28 plots
│       ├── Calibration → 8 plots + metrics
│       ├── Hypothesis testing → 3 plots + 2 CSVs
│       └── Leakage audit → 2 plots + CSVs
│
├── Phase 2: AZS baseline  
│   └── [Same credibility suite]
│
└── Phase 3: ESG Augmentation
    ├── [Same credibility suite]
    └── ESG gap analysis:
        ├── Gap engineering
        ├── Standalone tests
        ├── Default rate analysis → 2 plots + CSV
        ├── Talk-Walk scatter
        └── Correlation heatmap
```

Total execution time: ~45 min (depending on SHAP backend)

---

**All outputs are publication-ready.** Copy directly into your report/presentation.
