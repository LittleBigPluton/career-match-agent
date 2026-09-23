# Holdout Suite v1 — Baseline

**Suite:** `career_match_holdout`  
**Version:** `1.0.0`  
**Split:** Holdout  
**Status:** Original holdout evaluation recorded; frozen labels and configurations must not be changed in response to these results.  
**Scope:** 2 synthetic candidate scenarios, 40 labelled job postings, 3 ranking configurations, 6 evaluations.

## Scenarios

| Dataset | Scenario | Jobs | Expected accepts | Expected rejects |
|---|---|---:|---:|---:|
| `nlp_junior` | Junior multilingual NLP candidate; related engineering titles and specialized NLP requirements | 20 | 11 | 9 |
| `ml_platform_junior` | Junior ML serving/platform candidate; ML infrastructure roles versus adjacent frontend and other roles | 20 | 10 | 10 |

Both datasets were evaluated using the unchanged `hybrid_default`, `semantic_only` and `deterministic_only` configurations after development-suite evaluation. Filtering precedes ranking; LLM-generated job reports are not part of this benchmark.

## Filtering results

Filtering does not depend on the ranking configuration, so each scenario's values are reported once.

| Metric | `nlp_junior` | `ml_platform_junior` | Macro mean* |
|---|---:|---:|---:|
| Filtering F1 | 0.957 | 0.952 | 0.955 |
| Rejection-reason F1 | 0.941 | 0.952 | 0.947 |

*Macro means are calculated from the rounded terminal output and are therefore approximate. Full-precision values are preserved in the machine-readable results.*

### Observed filtering disagreements

Each scenario contained one false acceptance relative to its frozen labels:

| Dataset | Job ID | Expected | Actual | Expected reason | Actual reasons |
|---|---|---|---|---|---|
| `nlp_junior` | `synthetic:nlp-junior-12` | Reject | Accept | `role_mismatch` | None |
| `ml_platform_junior` | `synthetic:ml-platform-junior-11` | Reject | Accept | `role_mismatch` | None |

No other disagreements appeared in the saved per-job diagnostic inspection. The common failure mode is failure to reject these two off-target job postings by role. The precise classifier root cause has **not** been confirmed on the evaluated code revision; it should be investigated and regression-tested in a separate engineering change, without rewriting this baseline.

## Ranking results

### Per-scenario nDCG

| Scenario | Configuration | nDCG@5 | nDCG@10 |
|---|---|---:|---:|
| `nlp_junior` | Hybrid | 0.818 | 0.910 |
| `nlp_junior` | Semantic only | 0.810 | 0.898 |
| `nlp_junior` | Deterministic only | 0.974 | 0.922 |
| `ml_platform_junior` | Hybrid | 0.984 | 0.985 |
| `ml_platform_junior` | Semantic only | 0.984 | 0.982 |
| `ml_platform_junior` | Deterministic only | 0.944 | 0.949 |

### Macro-averaged nDCG

| Configuration | Mean nDCG@5* | Mean nDCG@10* |
|---|---:|---:|
| Hybrid | 0.901 | 0.948 |
| Semantic only | 0.897 | 0.940 |
| Deterministic only | 0.959 | 0.936 |

*Approximate means calculated from the displayed three-decimal scenario metrics; consult `holdout_suite_v1_summary.json` for full-precision aggregation.*

## Interpretation

Filtering generalized imperfectly from the calibrated development scenarios: the two holdout datasets each exposed one unexpected role-based acceptance. Ranking behavior also differed by scenario. On `nlp_junior`, deterministic-only produced a higher nDCG@5 than hybrid; on `ml_platform_junior`, hybrid and semantic-only produced higher nDCG@5 than deterministic-only. Across the two holdout scenarios, deterministic-only had the higher mean nDCG@5, while hybrid had the higher mean nDCG@10. These are **descriptions of agreement with the frozen synthetic labels**, not evidence that one configuration is generally superior.

## Limitations

- Only two synthetic candidate scenarios and 40 synthetic job postings were evaluated. The data are not a representative sample of real hiring markets or users.
- Relevance and rejection labels were assigned for the synthetic corpus rather than independently collected from real recruiter or applicant judgments. The rubric may favor explicit technical keywords.
- Filtering and ranking are a pipeline: incorrect acceptance decisions can affect the set of ranked jobs and therefore ranking metrics.
- Two scenarios do not support reliable statistical-significance or population-level confidence claims. The displayed macro averages are descriptive.
- Single-run latency includes possible embedding/model warm-up effects; no speed comparison is claimed. LLM report quality is outside this benchmark.
- The exact evaluation code revision, embedding model identifier/revision and relevant runtime configuration should accompany the published artifacts for full reproducibility (see the publication checklist below).

## Frozen dataset integrity

These SHA-256 checksums were recorded from the finalized **local** holdout JSON files before evaluation; earlier downloadable draft files are not interchangeable with these frozen copies.

| File | SHA-256 |
|---|---|
| `data/benchmarks/holdout/ml_platform_junior.json` | `2591e130f467cea0245f7fb28523ce6f8bda3dc2f4d97da31bb4d6748962b29d` |
| `data/benchmarks/holdout/nlp_junior.json` | `2378d70ba57e69eac0ccdf8510e199ac9f5fe219d28ddbb1e0ef77b342c1b4bc` |

Verify integrity before publishing:

```bash
sha256sum data/benchmarks/holdout/*.json
```

If a checksum differs, investigate against the originally frozen commit. Do not silently replace the baseline checksum or relabel the datasets after seeing the results.

## Reproduction and artifacts

The recorded evaluation was run from the repository root with the frozen datasets and unchanged ranking configurations:

```bash
python scripts/run_benchmark_suite.py \
    --split holdout \
    --output benchmark_results/holdout_suite_v1.json
```

The suite runner also wrote `benchmark_results/holdout_suite_v1_summary.json`. These original files and the dataset checksums are the reference evaluation artifacts; preserve them rather than overwriting them after subsequent fixes.
