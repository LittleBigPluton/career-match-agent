import asyncio
from typing import Any

import httpx

from career_match_agent.models.candidate import EmploymentType
from career_match_agent.models.job import JobSearchQuery
from career_match_agent.providers.adzuna import AdzunaJobProvider


def build_adzuna_payload() -> dict[str, Any]:
    return {"count": 1,"results": [{"id": "12345",
                                    "title": "Machine Learning Engineer",
                                    "description": ("Develop Python and PyTorch machine learning systems."),
                                    "redirect_url": ("https://www.adzuna.de/jobs/12345"),
                                    "created": "2026-08-20T10:30:00Z",
                                    "company": {"display_name": "Example AI GmbH"},
                                    "location": {"display_name": "Berlin", "area": ["Germany", "Berlin"]},
                                    "category": {"label": "IT Jobs", "tag": "it-jobs"},
                                    "contract_time": "full_time", "contract_type": "permanent"}]}


def create_provider(client: httpx.AsyncClient, *, maximum_requests: int = 6) -> AdzunaJobProvider:
    return AdzunaJobProvider(base_url="https://adzuna.test",
                             country="de",
                             app_id="test-app-id",
                             app_key="test-app-key",
                             timeout_seconds=10,
                             results_per_page=20,
                             maximum_requests=maximum_requests,
                             user_agent="career-match-agent-test",
                             client=client)


def test_adzuna_provider_normalizes_job() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert (request.url.params["what"] == "Machine Learning Engineer")
            assert (request.url.params["where"] == "Berlin")
            assert (request.url.params["app_id"] == "test-app-id")
            assert (request.url.params["app_key"] == "test-app-key")
            return httpx.Response(status_code=200, json=build_adzuna_payload())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://adzuna.test") as client:
            provider = create_provider(client)
            result = await provider.search(JobSearchQuery(keywords=["Machine Learning Engineer"], locations=["Berlin"]))
        assert result.provider == "adzuna"
        assert result.received_count == 1
        assert result.skipped_count == 0
        assert len(result.jobs) == 1

        job = result.jobs[0]
        assert job.provider == "adzuna"
        assert job.source_id == "adzuna:12345"
        assert job.external_id == "12345"
        assert job.title == ("Machine Learning Engineer")
        assert job.company == "Example AI GmbH"
        assert job.location == "Berlin"
        assert EmploymentType.FULL_TIME in (job.employment_types)
        assert "IT Jobs" in job.tags
        assert job.posted_at is not None

    asyncio.run(run_test())

def test_adzuna_provider_skips_job_that_cannot_be_normalized() -> None:
    async def run_test() -> None:
        payload = build_adzuna_payload()
        results = payload["results"]
        assert isinstance(results, list)

        results[0]["redirect_url"] = "not-a-valid-url"
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(status_code=200, json=payload)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://adzuna.test") as client:
            provider = create_provider(client)
            result = await provider.search(JobSearchQuery(keywords=["Machine Learning"]))
        assert result.received_count == 1
        assert result.skipped_count == 1
        assert result.jobs == []
        assert any("could not be normalized" in warning for warning in result.warnings)

    asyncio.run(run_test())

def test_adzuna_provider_respects_request_budget() -> None:
    async def run_test() -> None:
        request_count = 0
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            return httpx.Response(status_code=200, json=build_adzuna_payload())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://adzuna.test") as client:
            provider = create_provider( client, maximum_requests=2)
            result = await provider.search(JobSearchQuery(keywords=["Machine Learning Engineer", "Data Scientist"], locations=["Berlin", "Munich"], max_pages=3))

        assert request_count == 2
        assert result.pages_fetched == 2
        assert any("request budget was reached" in warning for warning in result.warnings)

    asyncio.run(run_test())
