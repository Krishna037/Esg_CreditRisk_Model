# ESG Talk-Walk Integration Module

## Overview
Integrates **BRSR disclosure data (Talk)** and **behavioral news data (Walk)** into a unified **Final ESG Score** for use in credit risk and probability of default (PD) models.

**Status**: ✅ Complete and tested (March 26, 2026)

## Files

### Main Module
- **`Capstone/src/esg_integration.py`** — Complete pipeline (11 stages)
- **`Capstone/src/esg_visualization.py`** — ESG visualization suite (auto-generated charts)

### Input Data
- **`Capstone/data/brsr_esg_scores_optimized.csv`** — Talk dataset (476 companies)
  - Columns: `company`, `env_score`, `soc_score`, `gov_score`
  - These are pre-computed keyword-based ESG disclosure scores

- **`Capstone/data/walk_data_esg_score.xlsx`** — Walk dataset (499 companies, pre-aggregated)
  - Columns: `company`, `environmental_weighted_score`, `social_weighted_score`, `governance_weighted_score`, `total_articles`
  - Walk scores are time-decay weighted news sentiment aggregates

### Output Data
- **`Capstone/data/esg_talk_walk_integrated.csv`** — Final integrated dataset (771 companies)
- **`Capstone/data/esg_talk_walk_integrated.json`** — Same data in JSON format
- **`Capstone/outputs/unified_pipeline/esg_integration_visualizations/`** — ESG analytics plots and summary

## Pipeline Stages

### Phase 1: Load & Preprocess
- Load BRSR (Talk) and pre-aggregated news (Walk) datasets
- Standardize column names
- Normalize Talk to [0, 1] and Walk to [-1, 1]

### Phase 2: Talk Scores
- Extract pre-computed BRSR disclosure scores
- Handle missing data with transparency penalty (0.1)
- Output: `Talk_E`, `Talk_S`, `Talk_G` ∈ [0, 1]

### Phase 3: Walk Scores
- Extract pre-aggregated time-decay weighted news sentiment
- Output: `Walk_E`, `Walk_S`, `Walk_G` ∈ [-1, 1]

### Phase 4: Gap Calculation (Greenwashing Detection)
$$\text{Gap}_i = \text{Talk}_i - \text{Walk}_i$$

**Risk Flags:**
- If Gap > 0.5: High Greenwashing Risk
- If Gap < 0: Understated Performance

### Phase 5: ESG Component Score
$$\text{ESG\_Comp}_i = 0.4 \cdot \text{Talk}_i + 0.6 \cdot \text{Walk}_i$$

### Phase 6: Gap Penalty
$$\text{Penalty}_i = \begin{cases} 0.25 \cdot \text{Gap}_i & \text{if Gap}_i > 0 \\ 0 & \text{otherwise} \end{cases}$$

### Phase 7: Final Component Score
$$\text{Final}_i = \text{ESG\_Comp}_i - \text{Penalty}_i$$

### Phase 8: Volatility Detection
- Flag companies with:
  - Extreme gaps (|Gap| > 0.7)
  - Extreme Walk swings (|Walk| > 0.8)
- Apply 10% penalty to Final ESG Score

### Phase 9: Final ESG Score (0-100 Scale)
$$\text{Final\_ESG\_Score} = 100 \times (0.44 \cdot \text{Final}_E + 0.31 \cdot \text{Final}_S + 0.25 \cdot \text{Final}_G)$$

Minus volatility penalty if flagged.

### Phase 10: Confidence Level
Based on article coverage:
- **Low**: No news articles (missing Walk)
- **Medium**: Sparse coverage
- **High**: Robust coverage

### Phase 11: Risk Flags
Combines:
- Greenwashing Risk (Gap > 0.5)
- High ESG Volatility
- Low Disclosure (missing Talk)
- Low Coverage (no news articles)

## Usage

### Run Standalone
```bash
cd Capstone/src
python esg_integration.py
```

### Output
Generates:
- **CSV**: Ready for import into PD models
- **JSON**: For API/web service integration
- **PNG charts**: Distribution, risk, gap, alignment, ranking, and correlation visuals

## ESG Visualization Pack

Running `esg_integration.py` now auto-creates a multi-chart ESG visualization bundle:

- `esg_score_distribution.png` — Final ESG score histogram with mean/median lines
- `confidence_vs_esg_score.png` — Score spread by confidence level (Low/Medium/High)
- `risk_flag_counts.png` — Frequency of all risk flags
- `talk_walk_final_component_means.png` — Talk vs Walk vs Final mean by E/S/G pillar
- `talk_vs_walk_scatter.png` — Three scatter plots for Talk-Walk alignment (E/S/G)
- `gap_distribution_by_pillar.png` — Greenwashing gap distributions with threshold markers
- `top_bottom_esg_companies.png` — Top and bottom ranked companies by Final ESG score
- `esg_correlation_heatmap.png` — Correlation matrix across Talk/Walk/Gap/Final fields
- `esg_score_quantile_bands.png` — Sorted score curve with quantile bands
- `visualization_summary.csv` — Quick numeric summary for the visualization run

## Output Schema

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `Company_Name` | str | - | Company identifier |
| `Talk_E/S/G` | float | [0, 1] | ESG disclosure scores (BRSR) |
| `Walk_E/S/G` | float | [-1, 1] | ESG behavior scores (news sentiment) |
| `Gap_E/S/G` | float | [-1, 1] | Talk - Walk (greenwashing proxy) |
| `Final_E/S/G` | float | [-1, 1] | Post-penalty component scores |
| `Final_ESG_Score` | float | [0, 100] | **Primary output for PD model** |
| `Confidence_Level` | str | {Low, Medium, High} | Data quality indicator |
| `Risk_Flag` | str | - | Combined risk summary |

## Key Metrics (771 Companies)

- **Mean ESG Score**: 16.9
- **Std Dev**: 16.6
- **Greenwashing Risk**: 415 companies (53.8%)
- **High Volatility**: 468 companies (60.7%)
- **High Confidence**: 303 companies (39.3%)

## Integration Points

### With Capstone Credit Risk Pipeline
The integrated ESG scores can be imported into existing PD models:

1. Load output CSV:
```python
esg_df = pd.read_csv("Capstone/data/esg_talk_walk_integrated.csv")
```

2. Merge with financial data:
```python
credit_df = credit_df.merge(esg_df[["Company_Name", "Final_ESG_Score"]], on="Company_Name")
```

3. Use `Final_ESG_Score` directly in:
   - Phase 3 ESG Augmentation (existing pipeline)
   - Alternative risk features for PD model

### Recommended PD Model Integration
- Add `Final_ESG_Score` as feature
- Weight ESG contribution: 5-10% of total PD signal
- Monitor for:
  - Non-linear relationships (high ESG ≠ low PD necessarily)
  - Correlation with financial distress indicators
  - Temporal stability (runs quarterly/annually)

## Edge Cases Handled

| Case | Action | Flag |
|------|--------|------|
| No BRSR data | Talk = 0.1 (penalty) | "Low Disclosure" |
| No news articles | Walk = 0 (neutral) | "Low Coverage" |
| Extreme gap | Applies penalty | "Greenwashing Risk" |
| High volatility | -10% ESG penalty | "High Volatility" |

## Quality Notes

- **Outer join on Company_Name**: Preserves all companies from both sources (771 total)
- **Normalization**: Talk normalized 0-100 → [0, 1]; Walk pre-normalized to [-1, 1]
- **Missing data strategy**: Conservative (penalty scores for missing Talk; zero for missing Walk)
- **Reproducibility**: Deterministic; re-running produces identical results

## Future Enhancements

1. **Dynamic time decay**: Update λ parameter based on industry/company lifecycle
2. **Sentiment fine-tuning**: Use industry-specific sentiment lexicons
3. **Multi-source Walk**: Incorporate supply chain, regulatory, social media data
4. **Threshold optimization**: Calibrate Gap, volatility thresholds based on default outcomes
5. **Temporal tracking**: Build time series to detect ESG deterioration signals

## Questions?

- Module location: `Capstone/src/esg_integration.py`
- Output location: `Capstone/data/esg_talk_walk_integrated.{csv,json}`
- Execution time: ~1 second for 771 companies
- Dependencies: pandas, numpy, json, logging
