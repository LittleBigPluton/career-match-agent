import asyncio

from career_match_agent.models.candidate import (
    EmploymentType
)
from career_match_agent.models.job import (
    JobPosting,
    JobProviderSearchResult,
    JobSearchQuery
)
from career_match_agent.providers.composite import CompositeJobProvider
from career_match_agent.providers.mock import MockJobProvider
from career_match_agent.services.job_normalizer import create_job_fingerprint
from career_match_agent.services.job_search import JobSearchService

class StaticJobProvider:
    def __init__(self, *, provider_name: str, jobs: list[JobPosting]) -> None:
        self.provider_name = provider_name
        self.jobs = jobs

    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        del query
        return JobProviderSearchResult(provider=self.provider_name, jobs=self.jobs, pages_fetched=1, received_count=len(self.jobs), skipped_count=0, warnings=[])

    async def aclose(self) -> None:
        pass

def make_job(*, external_id: str, title: str, location: str, remote: bool, employment_type: EmploymentType) -> JobPosting:
    company = "Example GmbH"
    return JobPosting(source_id=f"mock:{external_id}",
                      provider="mock",
                      external_id=external_id,
                      title=title,
                      company=company,
                      description=("Develop Python machine-learning systems and production APIs."),
                      location=location,
                      remote=remote,
                      employment_types=[employment_type],
                      url=f"https://example.com/jobs/{external_id}",
                      fingerprint=create_job_fingerprint(title=title, company=company, location=location))


def test_job_search_filters_results() -> None:
    async def run_test() -> None:
        provider = MockJobProvider(jobs=[make_job(external_id="1", title="Machine Learning Engineer", location="Berlin", remote=True, employment_type=(EmploymentType.FULL_TIME)),
                                         make_job(external_id="2", title="Data Analyst", location="Munich", remote=False, employment_type=(EmploymentType.FULL_TIME))])

        service = JobSearchService(provider)
        response = await service.search(JobSearchQuery(keywords=["Machine Learning"], locations=["Berlin"], remote_only=True, employment_types=[EmploymentType.FULL_TIME]))
        assert len(response.jobs) == 1
        assert response.jobs[0].title == ("Machine Learning Engineer")
        assert response.statistics.received_count == 2
        assert response.statistics.matched_count == 1
        assert response.statistics.returned_count == 1

    asyncio.run(run_test())

def test_job_search_deduplicates_across_providers() -> None:
    async def run_test() -> None:
        title = "Junior Machine Learning Engineer"
        company = "Example GmbH"
        location = "Berlin"
        fingerprint = create_job_fingerprint(title=title, company=company, location=location)
        arbeitnow_job = JobPosting(source_id="arbeitnow:abc",
                                   provider="arbeitnow",
                                   external_id="abc",
                                   title=title,
                                   company=company,
                                   description=("Develop Python machine-learning systems."),
                                   location=location,
                                   remote=False,
                                   url=("https://arbeitnow.example/jobs/abc"), fingerprint=fingerprint)

        adzuna_job = JobPosting(source_id="adzuna:123",
                                provider="adzuna",
                                external_id="123",
                                title=title,
                                company=company,
                                description=("Develop Python machine-learning systems."),
                                location=location,
                                remote=False,
                                url=("https://adzuna.example/jobs/123"), fingerprint=fingerprint)

        composite = CompositeJobProvider(providers=[StaticJobProvider(provider_name="arbeitnow", jobs=[arbeitnow_job]), StaticJobProvider(provider_name="adzuna", jobs=[adzuna_job])])
        service = JobSearchService(composite)
        response = await service.search(JobSearchQuery(keywords=["Machine Learning Engineer"], locations=["Berlin"], maximum_results=20))
        assert (response.statistics.received_count == 2)
        assert (response.statistics.normalized_count == 2)
        assert (response.statistics.matched_count == 2)
        assert (response.statistics.duplicate_count == 1)
        assert (response.statistics.returned_count == 1)
        assert len(response.jobs) == 1

    asyncio.run(run_test())
