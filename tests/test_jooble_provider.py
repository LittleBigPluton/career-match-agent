import asyncio
import json
from typing import Any

import httpx
import pytest

from career_match_agent.models.candidate import EmploymentType
from career_match_agent.models.job import JobSearchQuery
from career_match_agent.providers.base import JobProviderResponseError
from career_match_agent.providers.jooble import JoobleJobProvider


def build_jooble_payload() -> dict[str, Any]:
    return {"totalCount": 1, "jobs": [{"id": 987654,
                                       "title": ("Junior Machine Learning Engineer"),
                                       "location": "Munich",
                                       "snippet": ("Build Python machine-learning services and APIs."),
                                       "salary": "",
                                       "source": "Example",
                                       "type": "Full-time",
                                       "link": ("https://de.jooble.org/jdp/987654"),
                                       "company": "Example GmbH",
                                       "updated": ("2026-08-20T15:30:00")}]}


def create_provider(client: httpx.AsyncClient, *, maximum_requests: int = 4) -> JoobleJobProvider:
    return JoobleJobProvider(base_url="https://jooble.test",
                             api_key="test-api-key",
                             default_location="Germany",
                             timeout_seconds=10,
                             results_per_page=20,
                             maximum_requests=maximum_requests,
                             user_agent="career-match-agent-test",
                             client=client)


def test_jooble_provider_normalizes_job() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert (request.url.path == "/api/test-api-key")
            request_payload = json.loads(request.content)
            assert (request_payload["keywords"] == "Machine Learning Engineer")
            assert (request_payload["location"] == "Munich")
            assert request_payload["page"] == 1
            assert (request_payload["ResultOnPage"] == 20)
            return httpx.Response(status_code=200, json=build_jooble_payload())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://jooble.test") as client:
            provider = create_provider(client)
            result = await provider.search(JobSearchQuery(keywords=["Machine Learning Engineer"], locations=["Munich"]))
        assert result.provider == "jooble"
        assert result.received_count == 1
        assert len(result.jobs) == 1

        job = result.jobs[0]
        assert job.provider == "jooble"
        assert (job.source_id == "jooble:987654")
        assert (job.title == "Junior Machine Learning Engineer")
        assert job.company == "Example GmbH"
        assert job.location == "Munich"
        assert EmploymentType.FULL_TIME in (job.employment_types)
        assert job.posted_at is not None

    asyncio.run(run_test())

def test_jooble_provider_uses_default_location() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            request_payload = json.loads(request.content)
            assert (request_payload["location"] == "Germany")
            return httpx.Response(status_code=200, json=build_jooble_payload())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://jooble.test") as client:
            provider = create_provider(client)
            await provider.search(JobSearchQuery(keywords=["Machine Learning"]))

    asyncio.run(run_test())

def test_jooble_provider_raises_on_http_error() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(status_code=503)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://jooble.test") as client:
            provider = create_provider(client)
            with pytest.raises(JobProviderResponseError):
                await provider.search(JobSearchQuery(keywords=["Machine Learning"]))

    asyncio.run(run_test())


def test_jooble_provider_respects_request_budget() -> None:
    async def run_test() -> None:
        request_count = 0
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            return httpx.Response(status_code=200, json=build_jooble_payload())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://jooble.test") as client:
            provider = create_provider(client, maximum_requests=2)
            result = await provider.search(JobSearchQuery(keywords=["ML Engineer", "Data Scientist"], locations=["Berlin", "Munich"], max_pages=3))

        assert request_count == 2
        assert result.pages_fetched == 2
        assert any("request budget was reached" in warning for warning in result.warnings)

    asyncio.run(run_test())
