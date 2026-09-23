# Career Switcher - Development Baseline

**Dataset version:** 1.0.0
**Cases:** 20 synthetic job postings
**Status:** Frozen development baseline

## Scenario

This scenario evaluates CareerMatch Agent's ability to match a candidate transitioning from manufacturing quality engineering into data science and machine learning.

The synthetic candidate, Alex Morgan, has five years of manufacturing experience and has recently developed Python and machine learning skills through independent portfolio projects. However, the candidate has no professional experience deploying machine learning models.

The dataset contains 20 synthetic job postings: 12 expected acceptances and 8 expected rejections. It includes entry-level technical opportunities, positions where manufacturing experience provides transferable skills, and roles requiring more specialized machine learning experience.

The scenario tests whether the system can recognize transferable experience without treating unrelated professional experience as equivalent to direct machine learning experience. It also examines whether the ranking system can distinguish genuine career-transition opportunities from vacancies that merely share relevant technical keywords.

## Results

| Metric         | Hybrid | Semantic | Deterministic |
| -------------- | -----: | -------: | ------------: |
| Filtering F1   |  1.000 |    1.000 |         1.000 |
| Reason-code F1 |  1.000 |    1.000 |         1.000 |
| nDCG@5         |  1.000 |    0.991 |         1.000 |
| nDCG@10        |  0.972 |    0.960 |         0.974 |
| Recall@10      |  1.000 |    1.000 |         1.000 |

## Interpretation

All configurations achieved perfect filtering results and retrieved every relevant job within the top ten. Ranking quality was similar across configurations, with near-perfect nDCG values. These results provide a baseline for the career-transition scenario but offer limited differentiation between ranking approaches.

## Limitations

This dataset contains 20 synthetic postings for one career-switching candidate. Relevance grades depend on manually assigned judgments about transferable skills and suitability. The near-ceiling ranking metrics limit the scenario's ability to distinguish configurations. Results should not be interpreted as evidence of real-world performance.
