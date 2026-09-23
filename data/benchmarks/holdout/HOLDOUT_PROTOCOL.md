# Holdout evaluation protocol — v1.0.0

## Purpose

Evaluate CareerMatch Agent's **frozen filtering and ranking pipeline** on two previously unscored synthetic holdout scenarios. The holdout is separate from the four development scenarios and is intended as a final controlled evaluation, **not** evidence of real-world accuracy or an independently human-labelled dataset.

## Holdout scenarios

| Dataset | Focus | Job cases | Expected accepts | Expected rejects |
|---|---|---:|---:|---:|
| `nlp_junior` | Junior multilingual NLP candidate | 20 | 11 | 9 |
| `ml_platform_junior` | Junior ML serving/platform candidate | 20 | 10 | 10 |

The datasets deliberately include ambiguous titles, accepted jobs with different relevance grades, hard exclusions, seniority, location and work-mode constraints. The counts above describe the planned corpus; verify them against the finalized local files.

## Pre-evaluation checklist

1. **Review labels independently of model output.** Inspect candidate profiles and preferences, all acceptance/rejection labels, rejection reasons and ordinal relevance grades (0–3). Resolve any annotation errors before scoring.
2. Run **JSON syntax, Pydantic schema and structural tests only**. Do not run the filter, ranker or benchmark on draft holdout files during review.
3. Set both dataset versions to `1.0.0`, update version assertions in the corpus tests and finalize the files in `data/benchmarks/holdout/`.
4. Verify that the SHA-256 values below match your local finalized files; commit the datasets, protocol and checksums **before the first benchmark run**. A file edit changes its checksum and requires an updated record before evaluation.

## Frozen dataset checksums

SHA-256 values supplied from the local holdout files, prior to evaluation:

| File | SHA-256 |
|---|---|
| `ml_platform_junior.json` | `2591e130f467cea0245f7fb28523ce6f8bda3dc2f4d97da31bb4d6748962b29d` |
| `nlp_junior.json` | `2378d70ba57e69eac0ccdf8510e199ac9f5fe219d28ddbb1e0ef77b342c1b4bc` |

Verify and, optionally, write a checksum file:

```bash
sha256sum data/benchmarks/holdout/*.json
sha256sum data/benchmarks/holdout/*.json > data/benchmarks/holdout/SHA256SUMS
```

**Important:** These checksums refer to the user's edited local files, not to the original downloadable drafts. Recompute them if you change anything, including dataset versions or formatting, before declaring the holdout frozen.

## Frozen evaluation configuration

- Run the unchanged `hybrid_default`, `semantic_only` and `deterministic_only` ranking configurations.
- Preserve the embedding model, embedding settings, filtering/classification logic and metric definitions used for the frozen development suite. Record the evaluated commit hash and embedding model identifier in the final report.
- Run each holdout dataset only after its labels and checksum are frozen. Do not tune weights, change labels or patch the pipeline in response to the holdout results. If a correctness defect emerges, document the original result and address the fix separately rather than silently rerunning the same holdout as if it were untouched.

## Reporting

- Save complete holdout suite output and per-job diagnostics separately from the development results.
- Generate a holdout-only aggregated summary and report both scenario-level and macro-averaged filtering F1, reason-code F1, precision/recall, nDCG@5, nDCG@10 and MRR.
- Compare holdout and frozen development results descriptively without combining their metrics into a single mean.
- With only **two** synthetic holdout scenarios, do not present bootstrap confidence intervals or statistical-significance claims as reliable population inference. Treat latency numbers as exploratory unless model warm-up and repeated timing are standardized.

## Completion criterion

Publish the frozen checksums, evaluated code/configuration version, complete holdout outputs and a concise final development-versus-holdout report. Then close the benchmark milestone and move on to the end-to-end application demo, deployment and portfolio documentation.
