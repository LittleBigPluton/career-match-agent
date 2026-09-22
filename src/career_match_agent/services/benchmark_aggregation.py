from collections import defaultdict
from statistics import mean

from career_match_agent.models.benchmark import (
    AggregateMetricSummary,
    BenchmarkConfigurationSummary,
    JobMatchingBenchmarkResult,
    JobMatchingBenchmarkSuiteResult,
    JobMatchingBenchmarkSuiteSummary,
    RankingAtKSummary,
    RankingAtKMetrics
)


def _summarize(values: list[float]) -> AggregateMetricSummary:
    if not values:
        raise ValueError("Cannot summarize an empty metric collection.")

    return AggregateMetricSummary(mean=mean(values), minimum=min(values), maximum=max(values))


def _ranking_metric_at_k(result: JobMatchingBenchmarkResult, k: int) -> RankingAtKMetrics:
    for metric in result.ranking.at_k:
        if metric.k == k:
            return metric

    raise ValueError(f"Ranking cutoff k={k} is missing from dataset '{result.dataset_name}'.")


def summarize_benchmark_suite(suite_result: JobMatchingBenchmarkSuiteResult) -> JobMatchingBenchmarkSuiteSummary:
    """Macro-average benchmark metrics across scenarios."""
    grouped_results: dict[str, list[JobMatchingBenchmarkResult]] = defaultdict(list)
    for result in suite_result.results:
        grouped_results[result.configuration_name].append(result)

    if not grouped_results:
        raise ValueError("Benchmark suite contains no results.")

    configuration_groups = list(grouped_results.items())
    expected_dataset_names = {result.dataset_name for result in configuration_groups[0][1]}
    for configuration_name, results in configuration_groups[1:]:
        dataset_names = {result.dataset_name for result in results}
        if dataset_names != expected_dataset_names:
            raise ValueError(f"All benchmark configurations must cover the same scenarios. Configuration '{configuration_name}' does not.")

    summaries: list[BenchmarkConfigurationSummary] = []
    for configuration_name, results in configuration_groups:
        expected_k_values = {metric.k for metric in results[0].ranking.at_k}
        for result in results[1:]:
            current_k_values = {metric.k for metric in result.ranking.at_k}
            if current_k_values != expected_k_values:
                raise ValueError("Ranking cutoffs must match across benchmark scenarios.")

        ranking_summaries: list[RankingAtKSummary] = []
        for k in sorted(expected_k_values):
            metrics_at_k = [_ranking_metric_at_k(result, k) for result in results]
            ranking_summaries.append(RankingAtKSummary(k=k,
                                                       precision=_summarize([metric.precision for metric in metrics_at_k]),
                                                       recall=_summarize([metric.recall for metric in metrics_at_k]),
                                                       ndcg=_summarize([metric.ndcg for metric in metrics_at_k])))

        summaries.append(BenchmarkConfigurationSummary(configuration_name=configuration_name,
                                                       scenario_count=len(results),
                                                       filtering_f1=_summarize([result.filtering.f1 for result in results]),
                                                       reason_code_f1=_summarize([result.reason_codes.f1 for result in results]),
                                                       ranking_at_k=ranking_summaries,
                                                       mean_reciprocal_rank=_summarize([result.ranking.mean_reciprocal_rank for result in results])))

    return JobMatchingBenchmarkSuiteSummary(suite_name=suite_result.suite_name, suite_version=suite_result.suite_version, split=suite_result.split, configurations=summaries)
