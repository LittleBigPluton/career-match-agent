from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from career_match_agent.api.dependencies import get_job_report_generator
from career_match_agent.api.main import app
from test_job_evaluator import (
    FakeJobReportGenerator,
    make_request
)


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


def test_job_evaluation_endpoint() -> None:
    app.dependency_overrides[get_job_report_generator] = lambda: FakeJobReportGenerator()
    request = make_request()
    response = client.post("/matching/evaluate", json=request.model_dump(mode="json"))
    assert response.status_code == 200

    response_payload = response.json()
    assert response_payload["generation"] == {"provider": "fake", "model": "fake-report-model", "prompt_version": "job-report-test-v1"}
    assert response_payload["statistics"]["completed_count"] == 1
    assert response_payload["statistics"]["failed_count"]== 0

    evaluated_report = response_payload["reports"][0]
    assert evaluated_report["rank"] == 1
    assert evaluated_report["source_id"] == "mock:1"
    assert evaluated_report["cited_evidence"]
    assert evaluated_report["report"]["recommendation"] == "strong_match"


def test_job_evaluation_rejects_failed_filter_decision() -> None:
    request_payload = make_request().model_dump(mode="json")
    request_payload["ranked_jobs"][0]["decision"]["accepted"] = False
    response = client.post("/matching/evaluate",json=request_payload,)
    assert response.status_code == 422


def test_job_evaluation_rejects_duplicate_jobs() -> None:
    request_payload = make_request().model_dump(mode="json")
    request_payload["ranked_jobs"].append(request_payload["ranked_jobs"][0])
    response = client.post("/matching/evaluate", json=request_payload)
    assert response.status_code == 422
