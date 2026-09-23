from pathlib import Path

from career_match_agent.models.benchmark import JobMatchingBenchmarkDataset
from career_match_agent.models.matching import JobFilteringRequest
from career_match_agent.services.job_filter import filter_jobs_for_candidate

def test_ml_junior_benchmark_dataset_is_valid() -> None:
    dataset_path = Path("data/benchmarks/development/ml_junior.json")
    dataset = JobMatchingBenchmarkDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    assert dataset.name == "ml_junior"
    assert dataset.version == "1.0.0"
    assert len(dataset.jobs) == 20

    accepted = [case for case in dataset.jobs if case.expected_accept ]
    rejected = [case for case in dataset.jobs if not case.expected_accept]
    assert len(accepted) == 10
    assert len(rejected) == 10

def test_ml_junior_accepted_jobs_have_no_rejection_reasons() -> None:
    dataset = JobMatchingBenchmarkDataset.model_validate_json(
        Path("data/benchmarks/development/ml_junior.json").read_text(encoding="utf-8"))

    for case in dataset.jobs:
        if case.expected_accept:
            assert case.expected_rejection_reasons == []

def test_ml_junior_job_ids_are_unique() -> None:
    dataset = JobMatchingBenchmarkDataset.model_validate_json(
        Path("data/benchmarks/development/ml_junior.json").read_text(encoding="utf-8"))
    source_ids = [case.job.source_id for case in dataset.jobs]
    assert len(source_ids) == len(set(source_ids))

def test_ml_junior_remaining_filtering_cases() -> None:
    dataset = JobMatchingBenchmarkDataset.model_validate_json(Path("data/benchmarks/development/ml_junior.json").read_text(encoding="utf-8"))
    response = filter_jobs_for_candidate(JobFilteringRequest(profile=dataset.profile, preferences=dataset.preferences, jobs=[case.job for case in dataset.jobs]))
    decisions = {decision.job.source_id: decision for decision in [*response.accepted_jobs, *response.rejected_jobs]}

    # Equivalent target-role titles should be accepted.
    assert decisions["synthetic:ml-junior-07"].accepted
    assert decisions["synthetic:ml-junior-08"].accepted

    # Explicitly onsite-only work should violate remote/hybrid preferences.
    onsite = decisions["synthetic:ml-junior-20"]
    assert not onsite.accepted
    assert "work_mode_mismatch" in {reason.code.value for reason in onsite.rejection_reasons}

def test_backend_mid_benchmark_dataset_is_valid() -> None:
    dataset_path = Path("data/benchmarks/development/backend_mid.json")
    dataset = JobMatchingBenchmarkDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    assert dataset.name == "backend_mid"
    assert len(dataset.jobs) == 20

    accepted = [case for case in dataset.jobs if case.expected_accept]
    rejected = [case for case in dataset.jobs if not case.expected_accept]
    assert len(accepted) == 12
    assert len(rejected) == 8

    source_ids = [case.job.source_id for case in dataset.jobs]
    assert len(source_ids) == len(set(source_ids))

def test_data_junior_benchmark_dataset_is_valid() -> None:
    dataset_path = Path("data/benchmarks/development/data_junior.json")
    dataset = JobMatchingBenchmarkDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    assert dataset.name == "data_junior"
    assert dataset.version == "0.1.0"
    assert len(dataset.jobs) == 20

    accepted = [case for case in dataset.jobs if case.expected_accept]
    rejected = [case for case in dataset.jobs if not case.expected_accept]
    assert len(accepted) == 12
    assert len(rejected) == 8

    source_ids = [case.job.source_id for case in dataset.jobs]
    assert len(source_ids) == len(set(source_ids))
    assert sum(case.relevance_grade >= 2 for case in accepted) == 8
    assert all(not case.expected_rejection_reasons for case in accepted)
