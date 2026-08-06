from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from career_match_agent.api.dependencies import get_embedding_provider
from career_match_agent.api.main import app
from career_match_agent.models.candidate import (
    CandidateProfile,
    EmploymentType,
    JobPreferences
)
from career_match_agent.models.job import JobPosting
from career_match_agent.models.matching import JobFilterDecision
from career_match_agent.models.ranking import HybridRankingRequest
from career_match_agent.services.job_normalizer import create_job_fingerprint
from test_semantic_ranker import FakeEmbeddingProvider


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


def create_request_payload() -> dict[str, object]:
    title = "Machine Learning Engineer"
    company = "Example AI GmbH"
    location = "Berlin"

    job = JobPosting(source_id="mock:1", provider="mock", external_id="1", title=title, company=company,
                     description=("Develop Python and PyTorch machine-learning models and inference APIs."),
                     location=location, remote=False, employment_types=[EmploymentType.FULL_TIME],
                     tags=["Python", "PyTorch"], url="https://example.com/jobs/1",
                     fingerprint=create_job_fingerprint(title=title, company=company, location=location))

    decision = JobFilterDecision(job=job,accepted=True,matched_roles=["Machine Learning Engineer"],matched_required_keywords=["Python"],)
    request = HybridRankingRequest(profile=CandidateProfile(skills=["Python", "PyTorch"]),
                                   preferences=JobPreferences(roles=["Machine Learning Engineer"], required_keywords=["Python"]), accepted_jobs=[decision])

    return request.model_dump(mode="json")


def test_hybrid_ranking_endpoint() -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    response = client.post("/matching/rank", json=create_request_payload())
    assert response.status_code == 200

    response_payload = response.json()
    assert response_payload["embedding"] == {"provider": "fake", "model": "fake-semantic-model", "dimension": 3}
    assert (response_payload["statistics"]["ranked_count"] == 1)

    ranked_job = response_payload["ranked_jobs"][0]
    assert ranked_job["rank"] == 1
    assert ranked_job["hybrid_score"] > 0
    assert (ranked_job["decision"]["job"]["title"] == "Machine Learning Engineer")
    assert ranked_job["semantic_matches"]


def test_hybrid_ranking_rejects_failed_decision() -> None:
    request_payload = create_request_payload()
    accepted_jobs = request_payload["accepted_jobs"]
    assert isinstance(accepted_jobs, list)

    accepted_jobs[0]["accepted"] = False
    response = client.post("/matching/rank", json=request_payload)
    assert response.status_code == 422


def test_hybrid_ranking_validates_weights() -> None:
    request_payload = create_request_payload()
    request_payload["configuration"] = {"weights": {"semantic": 0.8, "skill_overlap": 0.3, "required_keywords": 0.1, "role_alignment": 0.1, "warning_quality": 0.1}}
    response = client.post("/matching/rank", json=request_payload)
    assert response.status_code == 422
