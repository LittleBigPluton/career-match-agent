import pytest
from typing import Any

from career_match_agent.models.ranking import HybridRankingConfiguration
from career_match_agent.models.benchmark import (
    JobMatchingBenchmarkDataset,
    JobMatchingBenchmarkResult,
    JobMatchingBenchmarkSuite
)
from career_match_agent.services.benchmark_runner import JobMatchingBenchmarkSuiteRunner


class FakeBenchmarkRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def run(self, *, dataset: JobMatchingBenchmarkDataset, configuration_name: str, ranking_configuration: HybridRankingConfiguration) -> Any:
        self.calls.append((dataset.name, configuration_name))

        return JobMatchingBenchmarkResult.model_construct(dataset_name=dataset.name,
                                                          dataset_version=dataset.version,
                                                          configuration_name=configuration_name,
                                                          ranking_configuration=ranking_configuration)
def make_dataset(name: str) -> JobMatchingBenchmarkDataset:
    return JobMatchingBenchmarkDataset.model_construct(name=name, version="1.0.0")

@pytest.mark.anyio
async def test_suite_runner_executes_every_dataset_and_configuration() -> None:
    suite = JobMatchingBenchmarkSuite(name="career-match-development", version="1.0.0", split="development", datasets=[make_dataset("ml-junior"), make_dataset("backend-mid")])
    configurations = {"hybrid": HybridRankingConfiguration(), "semantic": HybridRankingConfiguration()}
    benchmark_runner = FakeBenchmarkRunner()
    suite_runner = JobMatchingBenchmarkSuiteRunner(benchmark_runner=benchmark_runner)
    result = await suite_runner.run(suite=suite, ranking_configurations=configurations)
    assert len(result.results) == 4
    assert benchmark_runner.calls == [("ml-junior", "hybrid"), ("ml-junior", "semantic"), ("backend-mid", "hybrid"), ("backend-mid", "semantic")]
    assert result.suite_name == "career-match-development"
    assert result.suite_version == "1.0.0"
    assert result.split == "development"

@pytest.mark.anyio
async def test_suite_runner_rejects_empty_configurations() -> None:
    suite = JobMatchingBenchmarkSuite(name="career-match-development", version="1.0.0", split="development", datasets=[make_dataset("ml-junior")])
    suite_runner = JobMatchingBenchmarkSuiteRunner(benchmark_runner=FakeBenchmarkRunner())
    with pytest.raises(ValueError, match="At least one ranking configuration is required"):
        await suite_runner.run(suite=suite, ranking_configurations={})
