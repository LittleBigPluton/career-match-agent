# ml_junior - Development benchmark baseline

**Dataset version:** `1.0.0`  
**Status:** frozen development scenario  
**Baseline date:** 2026-09-23  
**Corpus:** `data/benchmarks/development/ml_junior.json`  
**Machine-readable results:** `benchmark_results/ml_junior.json`

## Scope

This synthetic development scenario has 20 labeled jobs: 10 expected accepts and 10 expected rejects. It was iteratively calibrated on the development split to resolve observed mismatches. It is *not* an independent holdout and must not be presented as an unbiased generalization estimate.

## Final reported metrics

| Metric | hybrid_default | semantic_only | deterministic_only |
|---|---:|---:|---:|
| Filtering F1 | 1.000 | 1.000 | 1.000 |
| Rejection-reason F1 | 1.000 | 1.000 | 1.000 |
| P@5 | 0.800 | 0.800 | 1.000 |
| R@5 | 0.571 | 0.571 | 0.714 |
| nDCG@5 | 0.694 | 0.641 | 0.839 |
| P@10 | 0.700 | 0.700 | 0.700 |
| R@10 | 1.000 | 1.000 | 1.000 |
| nDCG@10 | 0.832 | 0.778 | 0.928 |
| MRR | 1.000 | 1.000 | 1.000 |

All filtering decisions and rejection-reason labels match their expected values in this scenario.

## Interpretation

The benchmark achieved perfect filtering and rejection-reason F1 scores (1.000) across all three configurations. For ranking, deterministic-only achieved the highest nDCG@10 (0.928), followed by hybrid (0.832) and semantic-only (0.778). These results indicate that explicit skill and role matching aligns more closely with the assigned relevance labels in this scenario. However, the results are insufficient to establish which ranking configuration generalizes best.

## Limitations

This benchmark contains only 20 synthetic job postings for one junior ML candidate. Its relevance labels are manually assigned and may favor explicit skill overlap. Perfect filtering scores therefore do not establish real-world accuracy. Additionally, MRR and Recall@10 are saturated across all configurations, limiting their usefulness for comparison. Ranking latency measurements include potential model warm-up effects and should not be interpreted as reliable performance comparisons.
