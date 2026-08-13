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

## Grounded LLM evaluation

The top five recommendations from the frozen hybrid ranking configuration were evaluated with the local Ollama report generator.

- Reports attempted: 5
- Reports completed: 5
- Reports failed: 0
- Success rate: 1.000
- Average cited evidence items: 10.4
- Candidate-and-job evidence scope rate: 1.000

All five hybrid recommendations successfully produced grounded structured reports containing evidence from both candidate and job contexts.

### Latency

- Filtering: 0.041 s
- Ranking: 9.029 s
- LLM evaluation: 2,209.316 s (~36.8 min)
- Total: 2,218.387 s (~37.0 min)

Local LLM inference is therefore the dominant runtime bottleneck.

## Interpretation

The frozen configuration generalized well to unseen jobs, maintaining perfect filtering performance and strong ranking quality. All five evaluated recommendations also produced grounded reports with evidence from both candidate and job contexts.

The main remaining limitation is runtime, as local Ollama inference accounts for almost all evaluation latency.
