import asyncio
import math

from career_match_agent.models.benchmark import (
    BenchmarkJobCase,
    JobMatchingBenchmarkDataset
)
from career_match_agent.models.candidate import (
    CandidateProfile,
    EmploymentType,
    JobPreferences,
    SeniorityLevel,
    WorkMode
)
from career_match_agent.models.job import JobPosting
from career_match_agent.services.benchmark_runner import JobMatchingBenchmarkRunner
from career_match_agent.services.job_normalizer import create_job_fingerprint
from career_match_agent.models.ranking import HybridRankingConfiguration

class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-benchmark-embedding"
    dimension: int | None = 3

    def vector(self, text: str) -> list[float]:
        lowered = text.casefold()
        vector = [1.0 if ("machine learning" in lowered or "python" in lowered or "pytorch" in lowered)
                      else 0.1, 1.0 if "frontend" in lowered else 0.1, 1.0 if "sales" in lowered else 0.1]

        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [
            self.vector(text)
            for text in texts
        ]

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self.vector(text)
            for text in texts
        ]


def make_job(
    *,
    external_id: str,
    title: str,
    description: str,
) -> JobPosting:
    company = "Example GmbH"
    location = "Berlin"

    return JobPosting(
        source_id=(
            f"benchmark:{external_id}"
        ),
        provider="benchmark",
        external_id=external_id,
        title=title,
        company=company,
        description=description,
        location=location,
        remote=False,
        employment_types=[
            EmploymentType.FULL_TIME
        ],
        url=(
            "https://example.com/jobs/"
            f"{external_id}"
        ),
        fingerprint=create_job_fingerprint(
            title=title,
            company=company,
            location=location,
        ),
    )


def test_benchmark_runner_scores_pipeline() -> None:
    async def run_test() -> None:
        dataset = JobMatchingBenchmarkDataset(
            name="test-benchmark",
            version="1",
            profile=CandidateProfile(
                skills=[
                    "Python",
                    "PyTorch",
                ],
            ),
            preferences=JobPreferences(
                roles=[
                    "Machine Learning Engineer"
                ],
                locations=["Berlin"],
                work_modes=[
                    WorkMode.ON_SITE
                ],
                employment_types=[
                    EmploymentType.FULL_TIME
                ],
                seniority_levels=[
                    SeniorityLevel.JUNIOR
                ],
            ),
            jobs=[
                BenchmarkJobCase(
                    job=make_job(
                        external_id="1",
                        title=(
                            "Junior Machine "
                            "Learning Engineer"
                        ),
                        description=(
                            "Build Python and "
                            "PyTorch ML models."
                        ),
                    ),
                    expected_accept=True,
                    relevance_grade=3,
                ),
                BenchmarkJobCase(
                    job=make_job(
                        external_id="2",
                        title=(
                            "Senior Machine "
                            "Learning Engineer"
                        ),
                        description=(
                            "Lead Python ML projects."
                        ),
                    ),
                    expected_accept=False,
                    relevance_grade=0,
                    expected_rejection_reasons=[
                        "seniority_mismatch"
                    ],
                ),
            ],
        )

        runner = JobMatchingBenchmarkRunner(
            embedding_provider=(
                FakeEmbeddingProvider()
            )
        )

        result = await runner.run(
            dataset=dataset,
            configuration_name="default",
            ranking_configuration=(
                __import__(
                    "career_match_agent.models.ranking",
                    fromlist=["HybridRankingConfiguration"]).HybridRankingConfiguration()))
        assert result.filtering.accuracy == 1
        assert result.filtering.f1 == 1
        assert result.ranked_source_ids == ["benchmark:1"]

    asyncio.run(run_test())
