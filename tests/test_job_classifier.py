from career_match_agent.models.candidate import (
    CandidateProfile,
    LanguageEntry,
    SeniorityLevel,
    WorkMode
)
from career_match_agent.models.job import JobPosting
from career_match_agent.services.job_classifier import (
    build_candidate_language_map,
    detect_language_requirements,
    detect_matching_roles,
    detect_seniority,
    detect_work_modes,
    language_level_satisfies,
    contains_normalized_phrase
)
from career_match_agent.services.job_filter import location_matches
from career_match_agent.services.job_normalizer import create_job_fingerprint


def make_job(*, title: str, description: str, remote: bool | None = False, location: str = "Berlin") -> JobPosting:
    company = "Example GmbH"
    return JobPosting(source_id=f"mock:{title}",
                      provider="mock",
                      external_id=title,
                      title=title,
                      company=company,
                      description=description,
                      location=location,
                      remote=remote,
                      url="https://example.com/jobs/1",
                      fingerprint=create_job_fingerprint(
                      title=title,
                      company=company,
                      location=location))


def test_detect_matching_role_alias() -> None:
    job = make_job(title="AI Engineer", description="Build machine-learning systems.")
    matched_roles = detect_matching_roles(job, ["Machine Learning Engineer"])
    assert matched_roles == ["Machine Learning Engineer"]


def test_detect_senior_seniority() -> None:
    job = make_job(title="Senior Machine Learning Engineer", description="Lead ML deployment projects.")
    assert detect_seniority(job) == SeniorityLevel.SENIOR


def test_detect_internship_seniority() -> None:
    job = make_job(title="Machine Learning Intern", description="Join the research team.")
    assert detect_seniority(job) == (SeniorityLevel.INTERNSHIP)


def test_detect_hybrid_work_mode() -> None:
    job = make_job(title="Data Scientist",description=("We offer a hybrid working model with three office days per week."), remote=False)
    assert detect_work_modes(job) == [WorkMode.HYBRID]


def test_remote_provider_flag_detects_remote_mode() -> None:
    job = make_job(title="ML Engineer", description="Work from anywhere in Germany.", remote=True)
    assert WorkMode.REMOTE in detect_work_modes(job)


def test_detect_language_requirement() -> None:
    job = make_job(title="Machine Learning Engineer", description=("Requirements:\n Fluent English is required.\n German B2 is mandatory."))
    requirements = detect_language_requirements(job)
    assert len(requirements) == 2

    requirements_by_language = {requirement.language: requirement for requirement in requirements}
    assert (requirements_by_language["English"].minimum_level== "C1")
    assert (requirements_by_language["German"].minimum_level == "B2")


def test_build_candidate_language_map() -> None:
    profile = CandidateProfile(skills=["Python"],languages=[LanguageEntry(language="English", proficiency="C1"), LanguageEntry(language="German", proficiency="A2")])
    language_map = build_candidate_language_map(profile, preferred_languages=["Turkish"])
    assert language_map == {"English": "C1", "German": "A2", "Turkish": None}


def test_language_level_comparison() -> None:
    assert language_level_satisfies("C1", "B2")
    assert not language_level_satisfies("A2", "B2")

def test_ml_engineer_does_not_match_mlops_engineer() -> None:
    assert not contains_normalized_phrase("MLOps Engineer", "ML Engineer")

def test_german_does_not_match_germany() -> None:
    assert not contains_normalized_phrase("This is a remote role within Germany.", "German")

def test_role_phrase_matches_with_intermediate_title_word() -> None:
    assert contains_normalized_phrase("Machine Learning Platform Engineer", "Machine Learning Engineer")

def test_germany_does_not_create_german_requirement() -> None:
    job = make_job(title="Data Scientist",description=("Create statistical models with Python. Fluent English is required. This is a fully remote role within Germany."))
    requirements = detect_language_requirements(job)
    assert len(requirements) == 1
    assert requirements[0].language == "English"
    assert requirements[0].minimum_level == "C1"

def test_explicit_german_requirement_is_detected() -> None:
    job = make_job(title="Machine Learning Engineer", description=("German B2 is required. English is used internally."))
    requirements = detect_language_requirements(job)
    requirements_by_language = {requirement.language: requirement for requirement in requirements}
    assert (requirements_by_language["German"].minimum_level == "B2")

def test_language_levels_are_scoped_to_each_language() -> None:
    job = make_job(title="Finance Engineer", description=("Fluent in English and conversational in German."))
    requirements = detect_language_requirements(job)
    by_language = {requirement.language: requirement for requirement in requirements}
    assert by_language["English"].minimum_level == "C1"
    assert by_language["German"].minimum_level is None

def test_native_and_fluent_languages_are_not_merged() -> None:
    job = make_job(title="Support Engineer", description=("You are native in German and fluent in English."))
    requirements = detect_language_requirements(job)
    by_language = {requirement.language: requirement for requirement in requirements}
    assert by_language["German"].minimum_level == "native"
    assert by_language["English"].minimum_level == "C1"

def test_native_and_fluent_languages_are_not_merged() -> None:
    job = make_job(title="Support Engineer", description=("You are native in German and fluent in English."))
    requirements = detect_language_requirements(job)
    by_language = {requirement.language: requirement for requirement in requirements}
    assert by_language["German"].minimum_level == "native"
    assert by_language["English"].minimum_level == "C1"

def test_munich_matches_muenchen_metro_location() -> None:
    job = make_job(title="Machine Learning Engineer", location="Garching bei München, Bayern", description="Build ML systems.")
    assert location_matches(job, ["Munich"])
