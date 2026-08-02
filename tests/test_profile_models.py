import pytest
from pydantic import ValidationError

from career_match_agent.models.candidate import (
    CandidateContext,
    CandidateProfile,
    ExperienceEntry,
    JobPreferences,
)


def test_candidate_profile_accepts_structured_information() -> None:
    profile = CandidateProfile(
        full_name="Buggs Bunny",
        skills=["Python", "PyTorch"],
        experience=[ExperienceEntry(job_title="AI/ML Engineer Intern", organization="Example AI", technologies=["Python", "PyTorch"])])

    assert profile.full_name == "Buggs Bunny"
    assert profile.skills == ["Python", "PyTorch"]
    assert len(profile.experience) == 1

def test_candidate_profile_removes_duplicate_skills() -> None:
    profile = CandidateProfile(
        skills=["Python", "python", "PyTorch", ""])

    assert profile.skills == ["Python", "PyTorch"]

def test_candidate_profile_rejects_empty_profile() -> None:
    with pytest.raises(ValidationError, match="contains no information"):
        CandidateProfile()

def test_experience_requires_title_or_organization() -> None:
    with pytest.raises(ValidationError, match="job title or organization"):
        ExperienceEntry(technologies=["Python"])

def test_candidate_context_combines_profile_and_preferences() -> None:
    context = CandidateContext(profile=CandidateProfile(full_name="Buggs Bunny", skills=["Python"]),
                               preferences=JobPreferences(roles=["Machine Learning Engineer"]))

    assert context.profile.full_name == "Buggs Bunny"
    assert context.preferences.roles == ["Machine Learning Engineer"]

def test_candidate_profile_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CandidateProfile.model_validate({"full_name": "Buggs Bunny", "invented_field": "unexpected value"})
