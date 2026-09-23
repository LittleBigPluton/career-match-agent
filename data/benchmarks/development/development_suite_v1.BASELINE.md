# Development Suite v1 — Baseline

**Suite:** `career_match_development`  
**Version:** `1.0.0`  
**Split:** Development  

## Scenarios

Four synthetic candidate profiles are evaluated using 20 labelled job postings each:

| Dataset | Scenario | Job cases |
|---|---|---:|
| `ml_junior` | Junior ML/AI candidate; specialized versus adjacent technical roles | 20 |
| `backend_mid` | Mid-level backend engineer; language, framework and seniority alignment | 20 |
| `data_junior` | Junior data scientist; modelling versus analytics and keyword-only overlap | 20 |
| `career_switcher` | Manufacturing professional transitioning into data/ML; transferable skills versus direct experience | 20 |

Each scenario is run using the same three ablation configurations: `hybrid_default`, `semantic_only` and `deterministic_only`. Filtering is performed before ranking. LLM-generated job reports are **not** evaluated here.

## Aggregate results

All configurations achieved filtering F1 = **1.0000** and rejection-reason F1 = **1.0000** on all four synthetic scenarios.

Metrics below are **macro averages**: each scenario contributes equal weight. Means and ranges are from the saved development-suite validation output (rounded to four decimal places).

| Configuration | Mean nDCG@5 | Range nDCG@5 | Mean nDCG@10 | Range nDCG@10 |
|---|---:|---:|---:|---:|
| Hybrid | 0.8767 | 0.6941–1.0000 | 0.8976 | 0.8010–0.9858 |
| Semantic only | 0.8277 | 0.6408–0.9908 | 0.8737 | 0.7783–0.9756 |
| Deterministic only | 0.9129 | 0.8127–1.0000 | 0.9309 | 0.8215–1.0000 |

### Per-scenario nDCG@10

| Scenario | Hybrid | Semantic only | Deterministic only |
|---|---:|---:|---:|
| `ml_junior` | 0.832 | 0.778 | 0.928 |
| `backend_mid` | 0.986 | 0.976 | 1.000 |
| `data_junior` | 0.801 | 0.780 | 0.821 |
| `career_switcher` | 0.972 | 0.960 | 0.974 |

Values in the per-scenario table are rounded to three decimals for readability; the underlying JSON contains greater precision.

## Paired Ranking Analysis

Paired comparisons were conducted across the four development scenarios using nDCG@5 and nDCG@10.

| Comparison (Hybrid − Baseline) | Mean Δ nDCG@5 | Exploratory 95% CI | Mean Δ nDCG@10 | Exploratory 95% CI |
| ------------------------------ | ------------: | -----------------: | -------------: | -----------------: |
| Deterministic                  |       -0.0362 |  [-0.1085, 0.0000] |        -0.0333 | [-0.0760, -0.0066] |
| Semantic                       |       +0.0490 |   [0.0202, 0.0780] |        +0.0239 |   [0.0109, 0.0428] |

Hybrid ranking exceeded semantic-only on both metrics in all four scenarios. Deterministic-only achieved higher nDCG@10 in all four scenarios, while nDCG@5 was identical for three scenarios.

The bootstrap intervals are exploratory because the evaluation contains only four synthetic candidate scenarios. They do not establish statistical significance or predict performance on real job advertisements.

These results are preserved without further development-set optimization.


## Interpretation

The deterministic-only configuration has the highest average nDCG at both cutoffs on these four labelled development scenarios. The hybrid configuration improves on semantic-only ranking on these same scenarios. Variation is substantial: `ml_junior` and `data_junior` present more ordering difficulty than `backend_mid` and `career_switcher`. Preserve errors such as the irrelevant dashboard-migration job promoted by explicit keyword overlap; they are useful regression cases, not labels to adjust to raise scores.

These observations describe **agreement with the manually assigned synthetic labels**. They do not establish real-world superiority or justify selecting final ranking weights without independent evaluation.

## Limitations

- Only four synthetic candidate scenarios and 80 synthetic job cases are included. The scenarios are not a representative sample of job markets or users.
- Development labels and classifier behavior were reviewed together during calibration, so perfect filtering and reason-code F1 values are not independent estimates of accuracy.
- The label rubric may reward explicit skill overlap; all models are evaluated against the same labels, but this does not eliminate possible annotation bias.
- Several metrics, notably MRR and Recall@10 in some scenarios, are saturated and offer limited discrimination.
- Single-run latency measurements include initial embedding/model warm-up and do not support a fair speed comparison.
- Scenario-resampled uncertainty with four scenarios is exploratory; independent holdout evaluation and real-job validation remain necessary.
