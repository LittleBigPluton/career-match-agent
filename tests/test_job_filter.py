from career_match_agent.models.candidate import (
    CandidateProfile,
    EmploymentType,
    JobPreferences,
    LanguageEntry,
    SeniorityLevel,
    WorkMode
)
from career_match_agent.models.job import JobPosting
from career_match_agent.models.matching import (
    JobFilteringRequest,
    JobFilterReasonCode
)
from career_match_agent.services.job_filter import filter_jobs_for_candidate
from career_match_agent.services.job_normalizer import create_job_fingerprint


def make_job(*, external_id: str, title: str, description: str, location: str = "Berlin", remote: bool | None = False,
             employment_types: list[EmploymentType] | None = None) -> JobPosting:
    company = "Example AI GmbH"
    return JobPosting(source_id=f"mock:{external_id}",
                      provider="mock",
                      external_id=external_id,
                      title=title,
                      company=company,
                      description=description,
                      location=location,
                      remote=remote,
                      employment_types=employment_types or [],
                      url=f"https://example.com/jobs/{external_id}",
                      fingerprint=create_job_fingerprint(
                      title=title,
                      company=company,
                      location=location))


def create_profile() -> CandidateProfile:
    return CandidateProfile(skills=["Python", "PyTorch"],
                            languages=[LanguageEntry(language="English", proficiency="C1"), LanguageEntry(language="German", proficiency="A2")])


def create_preferences() -> JobPreferences:
    return JobPreferences(roles=["Machine Learning Engineer"],
                          locations=["Berlin"],
                          work_modes=[WorkMode.HYBRID, WorkMode.ON_SITE],
                          employment_types=[EmploymentType.FULL_TIME],
                          seniority_levels=[SeniorityLevel.ENTRY_LEVEL, SeniorityLevel.JUNIOR],
                          required_keywords=["Python"],
                          excluded_keywords=["Principal", "Staff Engineer"], preferred_languages=["English"])


def reason_codes(decision_reasons) -> set[JobFilterReasonCode]:
    return {reason.code for reason in decision_reasons}


def test_accepts_matching_job() -> None:
    job = make_job(external_id="1", title="Junior Machine Learning Engineer",
                   description=("Build Python and PyTorch models.\n Fluent English is required.\n We use a hybrid working model."),
                   employment_types=[EmploymentType.FULL_TIME])

    response = filter_jobs_for_candidate(JobFilteringRequest(profile=create_profile(), preferences=create_preferences(), jobs=[job]))
    assert response.statistics.accepted_count == 1
    assert response.statistics.rejected_count == 0

    decision = response.accepted_jobs[0]
    assert decision.accepted is True
    assert decision.matched_roles == ["Machine Learning Engineer"]
    assert decision.detected_seniority == (SeniorityLevel.JUNIOR)


def test_rejects_senior_role() -> None:
    job = make_job(external_id="2", title="Senior Machine Learning Engineer",
                   description=("Build Python models in our Berlin office."), employment_types=[EmploymentType.FULL_TIME])

    response = filter_jobs_for_candidate(JobFilteringRequest(profile=create_profile(), preferences=create_preferences(), jobs=[job]))
    decision = response.rejected_jobs[0]
    assert JobFilterReasonCode.SENIORITY_MISMATCH in (reason_codes(decision.rejection_reasons))


def test_rejects_language_level_mismatch() -> None:
    job = make_job(external_id="3",title="Junior Machine Learning Engineer",
                   description=("Python experience required.\n German B2 is mandatory.\n On-site position in Berlin."), employment_types=[EmploymentType.FULL_TIME])
    response = filter_jobs_for_candidate(JobFilteringRequest(profile=create_profile(), preferences=create_preferences(), jobs=[job]))
    decision = response.rejected_jobs[0]
    assert JobFilterReasonCode.LANGUAGE_LEVEL_MISMATCH in reason_codes(decision.rejection_reasons)


def test_rejects_excluded_keyword() -> None:
    job = make_job(external_id="4", title="Principal Machine Learning Engineer",
                   description=("Build Python machine-learning platforms."), employment_types=[EmploymentType.FULL_TIME])
    response = filter_jobs_for_candidate(JobFilteringRequest(profile=create_profile(), preferences=create_preferences(), jobs=[job]))
    decision = response.rejected_jobs[0]
    assert JobFilterReasonCode.EXCLUDED_KEYWORD_PRESENT in reason_codes(decision.rejection_reasons)


def test_unknown_seniority_creates_warning() -> None:
    job = make_job(external_id="5",
                   title="Machine Learning Engineer",
                   description=("Build Python models in a hybrid team."),
                   employment_types=[EmploymentType.FULL_TIME])
    response = filter_jobs_for_candidate(JobFilteringRequest(profile=create_profile(), preferences=create_preferences(), jobs=[job]))
    decision = response.accepted_jobs[0]
    assert JobFilterReasonCode.UNKNOWN_SENIORITY in reason_codes(decision.warnings)


def test_remote_job_can_override_location() -> None:
    job = make_job(external_id="6",
                   title="Junior Machine Learning Engineer",
                   description=("Fully remote role using Python."),
                   location="Hamburg",
                   remote=True,
                   employment_types=[EmploymentType.FULL_TIME])
    preferences = create_preferences()
    preferences.work_modes.append(WorkMode.REMOTE)
    response = filter_jobs_for_candidate(JobFilteringRequest(profile=create_profile(), preferences=preferences, jobs=[job]))
    assert response.statistics.accepted_count == 1
