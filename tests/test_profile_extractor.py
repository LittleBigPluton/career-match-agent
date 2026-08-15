import json

import pytest

from career_match_agent.services.profile_extractor import (
    CvTextTooLongError,
    EmptyCvTextError,
    ProfileResponseValidationError,
    build_candidate_profile_prompt,
    parse_candidate_profile_response,
    prepare_cv_text
)

from career_match_agent.models.candidate import CandidateProfile

def test_prepare_cv_text_removes_null_characters() -> None:
    prepared_text = prepare_cv_text("Machine\x00 Learning Engineer", maximum_characters=100)
    assert prepared_text == "Machine Learning Engineer"

def test_prepare_cv_text_rejects_empty_text() -> None:
    with pytest.raises(EmptyCvTextError, match="no usable text"):
        prepare_cv_text("   ", maximum_characters=100)

def test_prepare_cv_text_rejects_excessive_length() -> None:
    with pytest.raises(CvTextTooLongError, match="character limit"):
        prepare_cv_text("a" * 101, maximum_characters=100)

def test_parse_candidate_profile_response() -> None:
    response_content = json.dumps({"full_name": "Buggs Bunny", "location": "Germany",
                                   "professional_summary": ("Machine learning professional with a computational science background."),
                                   "skills": ["Python", "PyTorch"],
                                   "skill_evidence": [{"skill": "PyTorch","evidence": ["Developed neural-network models using PyTorch"]}],
                                   "experience": [],
                                   "projects": [],
                                   "education": [],
                                   "languages": [{"language": "English","proficiency": "C1"}],"certifications": [], "years_of_experience": None})

    profile = parse_candidate_profile_response(response_content)
    assert profile.full_name == "Buggs Bunny"
    assert profile.skills == ["Python", "PyTorch"]
    assert profile.languages[0].language == "English"

def test_parse_candidate_profile_rejects_invalid_json() -> None:
    with pytest.raises(ProfileResponseValidationError):
        parse_candidate_profile_response("This is not valid JSON.")

def test_parse_candidate_profile_rejects_unknown_fields() -> None:
    response_content = json.dumps({"full_name": "Buggs Bunny", "unexpected": "invalid"})
    with pytest.raises(ProfileResponseValidationError):
        parse_candidate_profile_response(response_content)

def test_profile_prompt_contains_schema_and_cv_text() -> None:
    prompt = build_candidate_profile_prompt(cv_text="Python and PyTorch experience",schema={"type": "object","properties": {"skills": {"type": "array"}}})
    assert "<JSON_SCHEMA>" in prompt
    assert "<CV_TEXT_JSON_STRING>" in prompt
    assert "Python and PyTorch experience" in prompt

def test_candidate_profile_skills_are_atomic() -> None:
    profile = CandidateProfile(skills=["Python", "PyTorch", "FastAPI", "SQL", "AWS EC2"])
    assert "Python" in profile.skills
    assert "PyTorch" in profile.skills
    assert not any(":" in skill for skill in profile.skills)
