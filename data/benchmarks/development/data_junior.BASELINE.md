# `data_junior` — Development Benchmark Baseline

**Target frozen dataset version:** `1.0.0`  
**Status:** Labels reviewed; update the local dataset version to `1.0.0`, rerun the benchmark, and commit the matching result JSON before treating this snapshot as frozen.

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

## Preserved ranking errors

- **`synthetic:data-junior-04` — Junior Data Scientist - Forecasting, grade 2:** Relevant statistical and modeling responsibilities, but the candidate's specialized time-series background is unestablished. Hybrid and semantic configurations omit this accepted case from their top ten; deterministic places it tenth.
- **`synthetic:data-junior-11` — Junior Data Analyst - Dashboard Migration, grade 0:** Explicit lexical decoy. The description and tags mention Python, pandas, scikit-learn and XGBoost, but job responsibilities exclude statistical analysis, experimentation and model development. All three configurations include it in the top ten; deterministic places it sixth.
- The first two jobs ranked by all configurations have grade 2 while grade-3 positions appear lower, providing another useful ordering challenge.

Do not edit labels to increase evaluation metrics. Future changes to the dataset, relevance rubric or expected filtering outcomes require a new version and a recorded explanation.

## Methodological limitations

- Synthetic and deliberately constructed: not representative evidence of production performance.
- MRR and Precision@5 are saturated in this scenario.
- Single-run ranking latency contains first-use/model warm-up effects and should **not** be used for comparisons between configurations.
- Ablations change scoring weights; verify whether semantic computations remain enabled before presenting computational cost comparisons.

## Freeze checklist

1. Change `data/benchmarks/development/data_junior.json` from `0.1.0-draft` to `1.0.0` without modifying labels or jobs.
2. Rerun corpus tests, classifier tests, and `scripts/run_benchmark.py`.
3. Check that saved `benchmark_results/data_junior.json` states `dataset_version: "1.0.0"` for all configurations, and that its metrics match or update this snapshot.
4. Record the embedding model name/revision, package commit and relevant runtime settings alongside the benchmark run if available.
5. Commit the dataset, this baseline document and the resulting JSON together.
