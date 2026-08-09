from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from career_match_agent.api.dependencies import (
    get_embedding_provider,
    get_job_provider,
    get_job_report_generator,
    get_search_planner
)
from career_match_agent.api.main import app
from career_match_agent.providers.mock import MockJobProvider
from test_agent_graph import (
    FakeEmbeddingProvider,
    FakeJobReportGenerator,
    FakeSearchPlanner,
    create_agent_request,
    create_ai_job
)


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


def test_agent_search_endpoint() -> None:
    app.dependency_overrides[get_search_planner] = lambda: FakeSearchPlanner()
    app.dependency_overrides[get_job_provider] = lambda: MockJobProvider(jobs=[create_ai_job()])
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_job_report_generator] = lambda: FakeJobReportGenerator()

    response = client.post("/agent/search", json=create_agent_request().model_dump(mode="json"))
    assert response.status_code == 200

    response_payload = response.json()
    assert response_payload["search_attempts"] == 2
    assert (response_payload["filtering_statistics"]["accepted_count"] == 1)
    assert (response_payload["ranking"]["statistics"]["ranked_count"] == 1)
    assert (response_payload["evaluation"]["statistics"]["completed_count"] == 1)

    trace_steps = [entry["step"] for entry in response_payload["trace"]]
    assert "broaden_search" in trace_steps
