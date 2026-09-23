# backend_mid - Development Benchmark Baseline

**Target frozen dataset version:** `1.0.0`  

## Scenario and scope

Synthetic mid-level backend engineer **Jonas Weber**, with four years of Python REST API, asynchronous service, and relational database experience. Relevant skills include FastAPI, Django, PostgreSQL, Docker, testing, and some AWS deployment. Target roles are Backend Engineer, Python Engineer, and Software Engineer; accepted locations are Berlin and Cologne, with remote/hybrid arrangements, full-time employment, and mid-level seniority.

The draft corpus contains **20 labeled jobs**: **12 expected accepts** and **8 expected rejects**. Among expected accepts, there are three grade-3, four grade-2, three grade-1, and two grade-0 cases. Acceptance means a job passes the intended hard filters; relevance grade expresses suitability *within* the accepted pool. An accepted grade-0 job is therefore not a contradictory label.

This is an iteratively reviewed **synthetic development** scenario. It is not an independent holdout and must not be presented as evidence of real-world accuracy or unbiased generalization.

## Observed benchmark metrics

These are the results reported from the behavioral benchmark run. The results-file dataset version was not independently verified from that run's output; after updating the local dataset to `1.0.0`, rerun and preserve the matching result JSON before declaring a final versioned freeze.

| Metric | `hybrid_default` | `semantic_only` | `deterministic_only` |
|---|---:|---:|---:|
| Filtering F1 | 1.000 | 1.000 | 1.000 |
| Rejection-reason F1 | 1.000 | 1.000 | 1.000 |
| Precision@5 | 1.000 | 0.800 | 1.000 |
| Recall@5 | 0.714 | 0.571 | 0.714 |
| nDCG@5 | 1.000 | 0.955 | 1.000 |
| Precision@10 | 0.700 | 0.700 | 0.700 |
| Recall@10 | 1.000 | 1.000 | 1.000 |
| nDCG@10 | 0.986 | 0.976 | 1.000 |
| MRR | 1.000 | 1.000 | 1.000 |

All filtering decisions and rejection-reason labels match their expected values in this scenario.

## Interpretation and preserved limitations

- All 20 filtering acceptance decisions and expected rejection reasons agree with the labels in this scenario. This does **not** establish perfect filtering on unseen job ads.
- Both `hybrid_default` and `deterministic_only` achieve nDCG@5 of `1.000`; `semantic_only` scores `0.955`. The configurations are close on this deliberately constructed dataset.
- All configurations retrieve the seven grade-2-or-higher jobs by rank 10; consequently Recall@10 and MRR are saturated. This scenario provides limited discrimination between ranking configurations.
- `deterministic_only` produces nDCG@10 of `1.000` against the *assigned* relevance labels. Avoid changing the corpus to preserve that score; use other development scenarios to test whether the ordering generalizes.
- Observed ranking times were **8,025.4 ms** (hybrid), **802.8 ms** (semantic), and **948.1 ms** (deterministic). These were sequential single runs. First-use embedding/model initialization can dominate the initial hybrid time; the present runs do **not** support like-for-like performance conclusions. Scoring-weight ablations may still execute shared semantic calculations.
