# CareerMatch Agent — Calibration v1

## Status

Frozen calibration baseline.

No further ranking configuration, benchmark labels, chunking strategy or embedding-model changes should be made specifically to improve performance on `calibration_v1.json`.

The calibration dataset may still be rerun in the future as a regression benchmark.

## Dataset

Dataset:

`data/benchmarks/calibration_v1.json`

Jobs: 30

Accepted by deterministic filtering: 17

Rejected by deterministic filtering: 13

Relevant ranking jobs: 12

## Frozen embedding model

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

A larger multilingual MPNet model was also evaluated but rejected because it reduced ranking quality and substantially increased runtime.

## Frozen hybrid ranking configuration

- semantic: 0.60
- skill overlap: 0.20
- required keywords: 0.10
- role alignment: 0.05
- warning quality: 0.05

Other parameters:

- semantic evidence count: 3
- chunk max characters: 700
- maximum candidate chunks: 24
- maximum job chunks: 12
- warning penalty: 0.10

## Calibration results

### Deterministic filtering

- Accuracy: 1.000
- Precision: 1.000
- Recall: 1.000
- F1: 1.000
- False acceptance rate: 0.000
- False rejection rate: 0.000
- Reason-code F1: 1.000

### Hybrid ranking

- Precision@5: 1.000
- Recall@5: 0.417
- nDCG@5: 0.794

- Precision@10: 0.900
- Recall@10: 0.750
- nDCG@10: 0.814

- MRR: 1.000

## Ablation result

The hybrid configuration outperformed semantic-only ranking at the top-10 level and outperformed deterministic-only ranking at the top-10 level.

The larger multilingual MPNet embedding model was rejected:

MiniLM hybrid:
- nDCG@5: 0.794
- nDCG@10: 0.814
- Precision@10: 0.900

MPNet hybrid:
- nDCG@5: 0.703
- nDCG@10: 0.759
- Precision@10: 0.800

## Benchmark policy

`calibration_v1.json` is now frozen.

It may be used for:

- regression testing
- future model comparisons
- architecture comparisons
- reproducibility checks

It should not be repeatedly modified or used for further manual tuning.

Generalization performance will be measured separately using `holdout_v1.json`.
