import pytest
from pydantic import ValidationError

from career_match_agent.models.benchmark import (
    JobMatchingBenchmarkDataset,
    JobMatchingBenchmarkSuite)


def make_benchmark_dataset(name: str) -> JobMatchingBenchmarkDataset:
    """Create a minimal benchmark dataset for suite-model tests."""
    return JobMatchingBenchmarkDataset.model_construct(name=name)


def test_benchmark_suite_accepts_multiple_datasets() -> None:
    first_dataset = make_benchmark_dataset("ml-junior")
    second_dataset = make_benchmark_dataset("backend-mid")
    suite = JobMatchingBenchmarkSuite(name="career-match-development", version="1.0.0", split="development", datasets=[first_dataset, second_dataset])
    assert suite.split == "development"
    assert len(suite.datasets) == 2


def test_benchmark_suite_rejects_duplicate_dataset_names() -> None:
    first_dataset = make_benchmark_dataset("ml-junior")
    duplicate = make_benchmark_dataset("ml-junior")
    with pytest.raises(ValidationError, match="Benchmark suite dataset names must be unique"):
        JobMatchingBenchmarkSuite(name="career-match-development", version="1.0.0", split="development", datasets=[first_dataset, duplicate])


def test_benchmark_suite_rejects_invalid_split() -> None:
    dataset = make_benchmark_dataset("ml-junior")
    with pytest.raises(ValidationError):
        JobMatchingBenchmarkSuite.model_validate({"name": "career-match", "version": "1.0.0", "split": "training", "datasets": [dataset]})
