import asyncio
from typing import Any

import httpx

from career_match_agent.models.candidate import EmploymentType
from career_match_agent.models.job import JobSearchQuery
from career_match_agent.providers.arbeitnow import (
    ArbeitnowJobProvider,
    ArbeitnowRawJob
)

def build_response_payload() -> dict[str, Any]:
    return {"data": [{"slug": "machine-learning-engineer-123",
                      "company_name": "Example AI GmbH",
                      "title": "Machine Learning Engineer",
                      "description": ("<p>Develop machine-learning models using Python &amp; PyTorch.</p>"),
                      "remote": True,
                      "url": ("https://www.arbeitnow.com/jobs/machine-learning-engineer-123"),
                      "tags": ["Machine Learning", "Python"],
                      "job_types": ["Full Time"],
                      "location": "Berlin",
                      "created_at": 1785600000}],
                      "links": {"next": None},
                      "meta": {"current_page": 1}}


def test_arbeitnow_provider_normalizes_response() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["page"] == "1"
            return httpx.Response(status_code=200, json=build_response_payload())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://arbeitnow.test") as client:
            provider = ArbeitnowJobProvider(base_url="https://arbeitnow.test", timeout_seconds=10, maximum_pages=3, user_agent="career-match-agent-test", client=client)
            result = await provider.search(JobSearchQuery(keywords=["Machine Learning Engineer"]))

        assert result.provider == "arbeitnow"
        assert result.pages_fetched == 1
        assert result.received_count == 1
        assert result.skipped_count == 0

        job = result.jobs[0]
        assert job.title == ("Machine Learning Engineer")
        assert job.company == "Example AI GmbH"
        assert job.remote is True
        assert job.location == "Berlin"
        assert job.employment_types == [EmploymentType.FULL_TIME]
        assert "<p>" not in job.description
        assert "Python & PyTorch" in job.description

    asyncio.run(run_test())


def test_arbeitnow_provider_sends_visa_filter() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert (request.url.params["visa_sponsorship"] == "true")
            return httpx.Response(status_code=200, json=build_response_payload())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://arbeitnow.test") as client:
            provider = ArbeitnowJobProvider(base_url="https://arbeitnow.test", timeout_seconds=10,
                                            maximum_pages=3, user_agent="career-match-agent-test", client=client)

            result = await provider.search(JobSearchQuery(keywords=["Machine Learning"], visa_sponsorship=True))
        assert result.jobs[0].visa_sponsorship is True

    asyncio.run(run_test())

def test_arbeitnow_accepts_mapping_job_types() -> None:
    raw_job = ArbeitnowRawJob.model_validate({"slug": "example-job",
                                              "company_name": "Example GmbH",
                                              "title": "Software Engineer",
                                              "description": "<p>Build software.</p>",
                                              "remote": False,
                                              "url": "https://example.com/job",
                                              "tags": ["Engineering"],
                                              "job_types": {"1": "professional / experienced"},
                                              "location": "Berlin", "created_at": 1787216434})

    assert raw_job.job_types == ["professional / experienced"]
