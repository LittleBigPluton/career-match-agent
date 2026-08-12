import math

from career_match_agent.models.benchmark import (
    BinaryClassificationMetrics,
    RankingAtKMetrics,
    RankingBenchmarkMetrics,
    ReasonCodeMetrics
)
from career_match_agent.models.matching import JobFilterDecision


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide safely when a metric denominator is zero."""
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_binary_metrics(*, expected: dict[str, bool], predicted: dict[str, bool]) -> BinaryClassificationMetrics:
    """Calculate accept/reject classification metrics."""
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for source_id, expected_accept in expected.items():
        predicted_accept = predicted.get(source_id, False)
        if expected_accept and predicted_accept:
            true_positive += 1

        elif not expected_accept and not predicted_accept:
            true_negative += 1

        elif not expected_accept and predicted_accept:
            false_positive += 1

        else:
            false_negative += 1

    total = (true_positive + true_negative + false_positive + false_negative)
    precision = safe_divide(true_positive, true_positive + false_positive,)
    recall = safe_divide(true_positive, true_positive + false_negative,)
    f1 = safe_divide(2 * precision * recall, precision + recall,)
    return BinaryClassificationMetrics(true_positive=true_positive, true_negative=true_negative, false_positive=false_positive, false_negative=false_negative,
                                       accuracy=safe_divide(true_positive + true_negative, total), precision=precision, recall=recall, f1=f1,
                                       false_acceptance_rate=safe_divide(false_positive, false_positive + true_negative),
                                       false_rejection_rate=safe_divide(false_negative, false_negative + true_positive))


def calculate_reason_code_metrics(*, expected: dict[str, set[str]], decisions: list[JobFilterDecision]) -> ReasonCodeMetrics:
    """Evaluate whether rejection reasons were correctly identified."""
    expected_pairs = {(source_id, reason_code) for source_id, reason_codes in expected.items() for reason_code in reason_codes}
    predicted_pairs = {(decision.job.source_id, reason.code.value) for decision in decisions for reason in decision.rejection_reasons}
    correct_pairs = (expected_pairs.intersection(predicted_pairs))
    precision = safe_divide(len(correct_pairs), len(predicted_pairs))
    recall = safe_divide(len(correct_pairs),len(expected_pairs))
    f1 = safe_divide(2 * precision * recall, precision + recall,)
    return ReasonCodeMetrics(precision=precision, recall=recall, f1=f1, expected_count=len(expected_pairs), predicted_count=len(predicted_pairs),
                             correct_count=len(correct_pairs),)


def discounted_cumulative_gain(relevance_grades: list[int]) -> float:
    """Calculate DCG using graded relevance."""
    return sum((2.0**relevance_grade - 1.0) / math.log2(rank + 2) for rank, relevance_grade in enumerate(relevance_grades))


def calculate_ndcg(*, ranked_grades: list[int], all_grades: list[int], k: int) -> float:
    """Calculate normalized discounted cumulative gain."""
    actual_grades = ranked_grades[:k]
    if len(actual_grades) < k:
        actual_grades = [*actual_grades, *([0] * (k - len(actual_grades)))]

    ideal_grades = sorted(all_grades, reverse=True)[:k]
    if len(ideal_grades) < k:
        ideal_grades = [*ideal_grades, *([0] * (k - len(ideal_grades)))]

    actual_dcg = discounted_cumulative_gain(actual_grades)
    ideal_dcg = discounted_cumulative_gain(ideal_grades)
    return safe_divide(actual_dcg, ideal_dcg)


def calculate_ranking_metrics(*, ranked_source_ids: list[str], relevance_by_source_id: dict[str, int], k_values: tuple[int, ...] = (5, 10),
                              relevant_threshold: int = 2) -> RankingBenchmarkMetrics:
    """Calculate Precision@K, Recall@K, nDCG and MRR."""
    all_grades = list(relevance_by_source_id.values())
    total_relevant = sum(grade >= relevant_threshold for grade in all_grades)
    ranked_grades = [relevance_by_source_id.get(source_id, 0) for source_id in ranked_source_ids]
    metrics_at_k: list[RankingAtKMetrics] = []
    dataset_size = len(relevance_by_source_id)
    for requested_k in k_values:
        k = min(requested_k, dataset_size)
        if k == 0:
            continue

        top_grades = ranked_grades[:k]
        relevant_in_top_k = sum(grade >= relevant_threshold for grade in top_grades)
        metrics_at_k.append(RankingAtKMetrics( k=k, precision=safe_divide(relevant_in_top_k, k), recall=safe_divide(relevant_in_top_k, total_relevant),
                                               ndcg=calculate_ndcg(ranked_grades=ranked_grades, all_grades=all_grades, k=k)))

    reciprocal_rank = 0.0
    for index, grade in enumerate(ranked_grades, start=1):
        if grade >= relevant_threshold:
            reciprocal_rank = 1 / index
            break

    return RankingBenchmarkMetrics(at_k=metrics_at_k, mean_reciprocal_rank=(reciprocal_rank), relevant_job_count=total_relevant)
