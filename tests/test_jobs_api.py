from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from career_match_agent.api.dependencies import get_job_provider
from career_match_agent.api.main import app
from career_match_agent.models.job import (
    JobPosting,
    JobProviderSearchResult,
    JobSearchQuery
)
from career_match_agent.providers.base import JobProviderUnavailableError
from career_match_agent.providers.mock import MockJobProvider
from career_match_agent.services.job_normalizer import create_job_fingerprint


client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()

def create_test_job() -> JobPosting:
    title = "Machine Learning Engineer"
    company = "Example AI GmbH"
    location = "Berlin"
    return JobPosting(source_id="mock:1",
                      provider="mock",
                      external_id="1",
                      title=title,
                      company=company,
                      description=("Develop Python and PyTorch models."),
                      location=location,
                      remote=True,
                      employment_types=["full_time"],
                      tags=["Python", "Machine Learning"],
                      url="https://example.com/jobs/1",
                      fingerprint=create_job_fingerprint(title=title, company=company, location=location))


def test_job_search_endpoint_returns_jobs() -> None:
    app.dependency_overrides[get_job_provider] = (lambda: MockJobProvider(jobs=[create_test_job()]))
    response = client.post("/jobs/search",json={"keywords": ["Machine Learning Engineer"], "locations": ["Berlin"], "remote_only": True, "maximum_results": 10})
    assert response.status_code == 200

    response_payload = response.json()
    assert response_payload["provider"] == "mock"
    assert len(response_payload["jobs"]) == 1

    job = response_payload["jobs"][0]
    assert job["title"] == ("Machine Learning Engineer")
    assert job["company"] == "Example AI GmbH"
    assert job["remote"] is True
    assert (response_payload["statistics"]["returned_count"] == 1)


class UnavailableProvider(MockJobProvider):
    async def search(self, query: JobSearchQuery) -> JobProviderSearchResult:
        del query
        raise JobProviderUnavailableError("The job provider could not be reached.")

def test_job_search_endpoint_returns_503() -> None:
    app.dependency_overrides[get_job_provider] = (lambda: UnavailableProvider())
    response = client.post("/jobs/search", json={"keywords": ["Data Scientist"]})
    assert response.status_code == 503
    assert "could not be reached" in (response.json()["detail"])

def test_job_search_endpoint_rejects_empty_keywords() -> None:
    response = client.post("/jobs/search", json={"keywords": ["", "   "]})
    assert response.status_code == 422
