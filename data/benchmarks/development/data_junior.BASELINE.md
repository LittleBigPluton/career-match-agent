# `data_junior` - Development Benchmark Baseline
## Scenario

Synthetic junior data scientist with 14 months of applied experience in Python, SQL, statistical analysis, churn classification and experiment evaluation. The dataset has 20 labelled job cases: 12 expected accepts and 8 expected rejects. Eight accepted cases have relevance grades of at least 2.

Filtering acceptance measures **hard constraints**; ranking relevance measures **suitability among accepted jobs**. Consequently, an accepted job can have relevance grade 0.

## Observed baseline

Results below were measured against draft dataset version `0.1.0-draft`. After changing only the version field to `1.0.0`, rerun the benchmark and confirm or update this table from the new output.

| Metric | `hybrid_default` | `semantic_only` | `deterministic_only` |
|---|---:|---:|---:|
| Filtering F1 | 1.000 | 1.000 | 1.000 |
| Reason-code F1 | 1.000 | 1.000 | 1.000 |
| Precision@5 | 1.000 | 1.000 | 1.000 |
| Recall@5 | 0.625 | 0.625 | 0.625 |
| nDCG@5 | 0.813 | 0.724 | 0.813 |
| Precision@10 | 0.700 | 0.700 | 0.800 |
| Recall@10 | 0.875 | 0.875 | 1.000 |
| nDCG@10 | 0.801 | 0.780 | 0.821 |
| MRR | 1.000 | 1.000 | 1.000 |

All filtering decisions and rejection-reason labels match their expected values in this scenario.

## Methodological limitations

- Synthetic and deliberately constructed: not representative evidence of production performance.
- MRR and Precision@5 are saturated in this scenario.
- Single-run ranking latency contains first-use/model warm-up effects and should **not** be used for comparisons between configurations.
- Ablations change scoring weights; verify whether semantic computations remain enabled before presenting computational cost comparisons.
