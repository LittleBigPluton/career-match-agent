# Holdout v1 Results

The holdout dataset was labeled and committed before running the frozen calibration configuration.

## Dataset

- Jobs: 24
- Expected accepted: 14
- Expected rejected: 10
- Relevant accepted jobs: 10

## Filtering

- Accuracy: 1.000
- Precision: 1.000
- Recall: 1.000
- F1: 1.000
- Reason-code F1: 1.000

## Hybrid ranking

- Precision@5: 1.000
- Recall@5: 0.500
- nDCG@5: 0.803
- Precision@10: 0.900
- Recall@10: 0.900
- nDCG@10: 0.861
- MRR: 1.000

## Interpretation

The frozen configuration maintained filtering performance and generalized to unseen ranking cases without further tuning.

No configuration changes were made based on holdout results.
