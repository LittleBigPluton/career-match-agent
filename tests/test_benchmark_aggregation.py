import pytest

from career_match_agent.models.benchmark import (
    BinaryClassificationMetrics,
    JobMatchingBenchmarkResult,
    JobMatchingBenchmarkSuiteResult,
    RankingAtKMetrics,
    RankingBenchmarkMetrics,
    ReasonCodeMetrics
)
from career_match_agent.services.benchmark_aggregation import summarize_benchmark_suite


def make_result(*, dataset_name: str, configuration_name: str, filtering_f1: float, reason_f1: float, ndcg_5: float, ndcg_10: float, mrr: float) -> JobMatchingBenchmarkResult:
    return JobMatchingBenchmarkResult.model_construct(dataset_name=dataset_name,
                                                      configuration_name=configuration_name,
                                                      filtering=BinaryClassificationMetrics.model_construct(f1=filtering_f1),
                                                      reason_codes=ReasonCodeMetrics.model_construct(f1=reason_f1),
                                                      ranking=RankingBenchmarkMetrics(at_k=[RankingAtKMetrics(k=5,
                                                                                                              precision=0.6,
                                                                                                              recall=0.5,
                                                                                                              ndcg=ndcg_5),
                                                                                            RankingAtKMetrics(k=10,
                                                                                                              precision=0.5,
                                                                                                              recall=0.8,
                                                                                                              ndcg=ndcg_10)],
                                                      mean_reciprocal_rank=mrr, relevant_job_count=4))

def test_summarize_benchmark_suite_macro_averages_scenarios() -> None:
    suite_result = JobMatchingBenchmarkSuiteResult(suite_name="career-match-development", suite_version="1.0.0", split="development",
                                                   results=[make_result(dataset_name="ml-junior",
                                                                        configuration_name="hybrid",
                                                                        filtering_f1=0.8,
                                                                        reason_f1=0.7,
                                                                        ndcg_5=0.9,
                                                                        ndcg_10=0.8,
                                                                        mrr=1.0),
                                                            make_result(dataset_name="backend-mid",
                                                                        configuration_name="hybrid",
                                                                        filtering_f1=0.6,
                                                                        reason_f1=0.5,
                                                                        ndcg_5=0.7,
                                                                        ndcg_10=0.6,
                                                                        mrr=0.5)])

    summary = summarize_benchmark_suite(suite_result)
    hybrid = summary.configurations[0]
    assert hybrid.scenario_count == 2
    assert hybrid.filtering_f1.mean == pytest.approx(0.7)
    assert hybrid.reason_code_f1.mean == pytest.approx(0.6)
    assert hybrid.mean_reciprocal_rank.mean == pytest.approx(0.75)

    ndcg_10 = next(metric for metric in hybrid.ranking_at_k if metric.k == 10)
    assert ndcg_10.ndcg.mean == pytest.approx(0.7)

def test_summarize_benchmark_suite_rejects_mismatched_scenarios() -> None:
    suite_result = JobMatchingBenchmarkSuiteResult(
        suite_name="career-match-development",
        suite_version="1.0.0",
        split="development",
        results=[make_result(dataset_name="ml-junior", configuration_name="hybrid", filtering_f1=0.8, reason_f1=0.7, ndcg_5=0.9, ndcg_10=0.8, mrr=1.0),
                 make_result(dataset_name="backend-mid", configuration_name="hybrid", filtering_f1=0.7, reason_f1=0.6, ndcg_5=0.8, ndcg_10=0.7, mrr=0.8),
                 make_result(dataset_name="ml-junior", configuration_name="semantic", filtering_f1=0.8, reason_f1=0.7, ndcg_5=0.85, ndcg_10=0.75, mrr=0.9)])

    with pytest.raises(ValueError, match="All benchmark configurations must cover the same scenarios"):
        summarize_benchmark_suite(suite_result)


def make_result_with_cutoffs(*, dataset_name: str, configuration_name: str, cutoffs: list[int]) -> JobMatchingBenchmarkResult:
    return JobMatchingBenchmarkResult.model_construct(dataset_name=dataset_name,
                                                      configuration_name=configuration_name,
                                                      filtering=BinaryClassificationMetrics.model_construct(f1=0.8),
                                                      reason_codes=ReasonCodeMetrics.model_construct(f1=0.7),
                                                      ranking=RankingBenchmarkMetrics(at_k=[RankingAtKMetrics(k=k,
                                                                                                              precision=0.6,
                                                                                                              recall=0.5,
                                                                                                              ndcg=0.8) for k in cutoffs],
                                                      mean_reciprocal_rank=0.9,
                                                      relevant_job_count=4))

def test_summarize_benchmark_suite_rejects_mismatched_ranking_cutoffs() -> None:
    suite_result = JobMatchingBenchmarkSuiteResult(suite_name="career-match-development",
                                                   suite_version="1.0.0",
                                                   split="development",
                                                   results=[make_result_with_cutoffs(dataset_name="ml-junior",
                                                                                     configuration_name="hybrid",
                                                                                     cutoffs=[5, 10]),
                                                            make_result_with_cutoffs(dataset_name="backend-mid",
                                                                                     configuration_name="hybrid",
                                                                                     cutoffs=[5, 8])])

    with pytest.raises(ValueError, match="Ranking cutoffs must match across benchmark scenarios"):
        summarize_benchmark_suite(suite_result)
