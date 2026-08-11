import pytest

from career_match_agent.services.benchmark_metrics import (
    calculate_binary_metrics,
    calculate_ndcg,
    calculate_ranking_metrics
)


def test_binary_metrics_perfect_classifier() -> None:
    metrics = calculate_binary_metrics(expected={"job-1": True, "job-2": False, "job-3": True}, predicted={"job-1": True, "job-2": False, "job-3": True})
    assert metrics.accuracy == 1
    assert metrics.precision == 1
    assert metrics.recall == 1
    assert metrics.f1 == 1


def test_binary_metrics_detect_false_acceptance() -> None:
    metrics = calculate_binary_metrics(expected={"good": True, "bad": False}, predicted={"good": True, "bad": True})
    assert metrics.false_positive == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.false_acceptance_rate == 1

def test_ndcg_is_one_for_ideal_ranking() -> None:
    ndcg = calculate_ndcg(ranked_grades=[3, 2, 1, 0], all_grades=[3, 2, 1, 0], k=4)
    assert ndcg == pytest.approx(1.0)

def test_ndcg_penalizes_bad_order() -> None:
    ndcg = calculate_ndcg(ranked_grades=[0, 1, 2, 3], all_grades=[3, 2, 1, 0], k=4)
    assert ndcg < 1

def test_ranking_metrics_reward_relevant_top_results() -> None:
    metrics = calculate_ranking_metrics(
        ranked_source_ids=["strong", "good", "bad", "weak"],
        relevance_by_source_id={"strong": 3, "good": 2, "weak": 1, "bad": 0}, k_values=(2, 4))
    first_cutoff = metrics.at_k[0]
    assert first_cutoff.precision == 1
    assert first_cutoff.recall == 1
    assert metrics.mean_reciprocal_rank == 1
