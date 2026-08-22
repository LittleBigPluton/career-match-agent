import asyncio

import pytest

from career_match_agent.models.job import (
    JobPosting,
    JobProviderSearchResult,
    JobSearchQuery
)
from career_match_agent.providers.base import JobProviderUnavailableError
from career_match_agent.providers.composite import CompositeJobProvider
from career_match_agent.services.job_normalizer import create_job_fingerprint

def create_test_job(*, provider: str, external_id: str, title: str = "Machine Learning Engineer") -> JobPosting:
    company = "Example GmbH"
    location = "Berlin"

    return JobPosting(source_id=f"{provider}:{external_id}",
                      provider=provider,
                      external_id=external_id,
                      title=title,
                      company=company,
                      description=("Develop Python machine-learning systems."),
                      location=location,
                      remote=False,
                      url=(f"https://example.com/jobs/{external_id}"),
                      fingerprint=create_job_fingerprint(title=title, company=company, location=location))

class StaticJobProvider:
    def __init__(self, *, provider_name: str, jobs: list[JobPosting]) -> None:
        self.provider_name = provider_name
        self.jobs = jobs

    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        del query
        return JobProviderSearchResult(provider=self.provider_name, jobs=self.jobs, pages_fetched=1, received_count=len(self.jobs), skipped_count=0, warnings=[])

    async def aclose(self) -> None:
        pass

class FailingJobProvider:
    provider_name = "broken"
    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        del query
        raise JobProviderUnavailableError("Provider unavailable.")

    async def aclose(self) -> None:
        pass

def test_composite_provider_merges_results() -> None:
    async def run_test() -> None:
        arbeitnow_job = create_test_job(provider="arbeitnow", external_id="1")
        adzuna_job = create_test_job(provider="adzuna", external_id="2", title="Data Scientist")
        composite = CompositeJobProvider(providers=[StaticJobProvider(provider_name="arbeitnow", jobs=[arbeitnow_job]), StaticJobProvider(provider_name="adzuna", jobs=[adzuna_job])])
        result = await composite.search(JobSearchQuery(keywords=["Machine Learning"]))
        assert result.provider == "multi"
        assert len(result.jobs) == 2
        assert result.received_count == 2
        assert result.pages_fetched == 2
        assert {job.provider for job in result.jobs} == {"arbeitnow", "adzuna"}

    asyncio.run(run_test())


def test_composite_provider_survives_partial_failure() -> None:
    async def run_test() -> None:
        successful_job = create_test_job(provider="arbeitnow", external_id="1")
        composite = CompositeJobProvider(providers=[StaticJobProvider(provider_name="arbeitnow", jobs=[successful_job]), FailingJobProvider()])
        result = await composite.search(JobSearchQuery(keywords=["Machine Learning Engineer"]))
        assert len(result.jobs) == 1
        assert result.jobs[0].provider == ("arbeitnow")
        assert any("broken" in warning for warning in result.warnings)

    asyncio.run(run_test())

def test_composite_provider_raises_when_all_providers_fail() -> None:
    async def run_test() -> None:
        composite = CompositeJobProvider(providers=[FailingJobProvider(), FailingJobProvider()])
        with pytest.raises( JobProviderUnavailableError, match="All configured job providers failed"):
            await composite.search(JobSearchQuery(keywords=["Machine Learning Engineer"]))

    asyncio.run(run_test())
