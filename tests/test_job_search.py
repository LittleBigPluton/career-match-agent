import asyncio

from career_match_agent.models.candidate import (
    EmploymentType
)
from career_match_agent.models.job import (
    JobPosting,
    JobSearchQuery
)
from career_match_agent.providers.mock import MockJobProvider
from career_match_agent.services.job_normalizer import create_job_fingerprint
from career_match_agent.services.job_search import JobSearchService


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
