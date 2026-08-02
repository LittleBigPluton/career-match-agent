from collections.abc import Generator

import pymupdf
import pytest
from fastapi.testclient import TestClient

from career_match_agent.api.dependencies import (
    get_profile_extractor,
)
from career_match_agent.api.main import app
from career_match_agent.models.candidate import (
    CandidateProfile,
    EducationEntry,
    SkillEvidence,
)
from career_match_agent.services.profile_extractor import (
    ProfileModelUnavailableError,
    ProfileResponseValidationError,
)


client = TestClient(app)


def create_pdf_bytes(text: str) -> bytes:
    """Create an in-memory PDF for profile API tests."""
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return document.tobytes()

@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()

class FakeProfileExtractor:
    provider_name = "fake"
    model_name = "fake-profile-model"
    prompt_version = "candidate-profile-test-v1"

    async def extract(self, cv_text: str) -> CandidateProfile:
        assert "Machine Learning Engineer" in cv_text
        return CandidateProfile(full_name="Buggs Bunny",
                                location="Germany",
                                skills=["Python", "PyTorch"],
                                skill_evidence=[SkillEvidence(skill="PyTorch", evidence=["Machine Learning Engineer with PyTorch"])],
                                education=[EducationEntry(degree="M.Sc.", field_of_study="Computational Science", institution="University of Regensburg")])

class UnavailableProfileExtractor(FakeProfileExtractor):
    async def extract(self, cv_text: str) -> CandidateProfile:
        raise ProfileModelUnavailableError("The configured Ollama model could not be reached.")

class InvalidResponseProfileExtractor(FakeProfileExtractor):
    async def extract(self, cv_text: str) -> CandidateProfile:
        raise ProfileResponseValidationError("The model returned an invalid candidate-profile response.")

def test_candidate_profile_endpoint_returns_profile() -> None:
    app.dependency_overrides[get_profile_extractor] = (lambda: FakeProfileExtractor())
    pdf_bytes = create_pdf_bytes("Machine Learning Engineer with PyTorch experience.")
    response = client.post("/profiles/candidate/extract",files={"file": ("candidate_cv.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 200
    response_payload = response.json()
    assert response_payload["document"]["filename"] == ("candidate_cv.pdf")
    assert "text" not in response_payload["document"]
    assert response_payload["profile"]["full_name"] == ("Buggs Bunny")
    assert response_payload["profile"]["skills"] == ["Python", "PyTorch"]
    assert response_payload["extraction"] == {"provider": "fake", "model": "fake-profile-model", "prompt_version": "candidate-profile-test-v1"}


def test_candidate_profile_endpoint_returns_503_when_model_fails() -> None:
    app.dependency_overrides[get_profile_extractor] = (lambda: UnavailableProfileExtractor())
    pdf_bytes = create_pdf_bytes("Machine Learning Engineer")
    response = client.post("/profiles/candidate/extract", files={"file": ("candidate_cv.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 503
    assert "could not be reached" in response.json()["detail"]


def test_candidate_profile_endpoint_returns_502_for_invalid_output() -> None:
    app.dependency_overrides[get_profile_extractor] = (lambda: InvalidResponseProfileExtractor())
    pdf_bytes = create_pdf_bytes("Machine Learning Engineer")
    response = client.post("/profiles/candidate/extract",files={"file": ("candidate_cv.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 502
    assert "invalid candidate-profile" in (response.json()["detail"])
