from career_match_agent.models.candidate import (
    EmploymentType,
    JobPreferences,
    WorkMode
)
from career_match_agent.models.job import JobPosting
from career_match_agent.models.matching import (
    JobFilterDecision,
    JobFilteringRequest,
    JobFilteringResponse,
    JobFilteringStatistics,
    JobFilterPolicy,
    JobFilterReason,
    JobFilterReasonCode
)
from career_match_agent.services.job_classifier import (
    build_candidate_language_map,
    contains_normalized_phrase,
    create_job_searchable_text,
    detect_language_requirements,
    detect_matching_roles,
    detect_seniority,
    detect_work_modes,
    language_level_satisfies
)
from career_match_agent.services.job_normalizer import normalize_for_matching

LOCATION_ALIASES: dict[str, set[str]] = { "munich": {"munich", "munchen", "muenchen"}, "berlin": {"berlin"}, "stuttgart": {"stuttgart"}}

def create_reason(code: JobFilterReasonCode, message: str, evidence: list[str] | None = None) -> JobFilterReason:
    """Create a filter reason with optional evidence."""
    return JobFilterReason(code=code, message=message, evidence=evidence or [])


def location_terms(location: str) -> set[str]:
    """Return normalized aliases for a requested location."""
    normalized_location = normalize_for_matching(location)
    for canonical, aliases in LOCATION_ALIASES.items():
        normalized_aliases = {normalize_for_matching(alias) for alias in aliases}
        if (normalized_location == normalize_for_matching(canonical) or normalized_location in normalized_aliases):
            return normalized_aliases

    return {normalized_location}


def location_matches(job: JobPosting, preferred_locations: list[str]) -> bool:
    """Check whether a job matches a preferred location."""
    if not preferred_locations:
        return True

    if not job.location:
        return False

    normalized_job_location = normalize_for_matching(job.location)
    for preferred_location in preferred_locations:
        aliases = location_terms(preferred_location)
        if any(alias in normalized_job_location for alias in aliases):
            return True

    return False


def detect_keyword_matches(job: JobPosting, keywords: list[str]) -> list[str]:
    """Return keywords found in the job posting."""
    searchable_text = create_job_searchable_text(job)
    return [keyword for keyword in keywords if contains_normalized_phrase(searchable_text, keyword)]


def detect_title_keyword_matches(job: JobPosting, keywords: list[str]) -> list[str]:
    """Match role-level exclusions against title and tags."""
    searchable_text = " ".join([job.title, " ".join(job.tags)])
    return [keyword for keyword in keywords if contains_normalized_phrase(searchable_text, keyword)]


def employment_types_match(job_types: list[EmploymentType], preferred_types: list[EmploymentType]) -> bool:
    """Check whether normalized employment types overlap."""
    return bool(set(job_types).intersection(preferred_types))


def evaluate_job(job: JobPosting, *, profile_languages: dict[str, str | None], preferences: JobPreferences, policy: JobFilterPolicy) -> JobFilterDecision:
    """Evaluate one job against deterministic suitability rules."""
    rejection_reasons: list[JobFilterReason] = []
    warnings: list[JobFilterReason] = []
    matched_roles = detect_matching_roles(job, preferences.roles)
    detected_seniority = detect_seniority(job)
    detected_work_modes = detect_work_modes(job)
    language_requirements = detect_language_requirements(job)
    matched_required_keywords = detect_keyword_matches(job, preferences.required_keywords)
    matched_excluded_keywords = detect_title_keyword_matches(job, preferences.excluded_keywords)

    # Role compatibility
    if (preferences.roles and not matched_roles and policy.reject_role_mismatch):
        rejection_reasons.append(create_reason(JobFilterReasonCode.ROLE_MISMATCH,f"The job title '{job.title}' does not match the preferred roles.",[job.title]))

    # Work mode
    preferred_work_modes = set(preferences.work_modes)
    detected_work_mode_set = set(detected_work_modes)
    if not detected_work_modes:
        unknown_work_mode_reason = create_reason(JobFilterReasonCode.UNKNOWN_WORK_MODE,"The job's work mode could not be determined.")

        if policy.allow_unknown_work_mode:
            warnings.append(unknown_work_mode_reason)

        else:
            rejection_reasons.append(unknown_work_mode_reason)

    elif not preferred_work_modes.intersection(detected_work_mode_set):
        if policy.reject_work_mode_mismatch:
            rejection_reasons.append(create_reason(JobFilterReasonCode.WORK_MODE_MISMATCH,
                                                   "The job's detected work mode is not among the candidate's preferred work modes.",
                                                   [mode.value for mode in detected_work_modes]))

    # Location
    remote_location_override = (policy.remote_overrides_location and WorkMode.REMOTE in preferred_work_modes and WorkMode.REMOTE in detected_work_mode_set)
    if (preferences.locations and not remote_location_override and not location_matches(job, preferences.locations) and policy.reject_location_mismatch):
        rejection_reasons.append(create_reason(JobFilterReasonCode.LOCATION_MISMATCH,
                                               f"The job location '{job.location or 'unknown'}' does not match the preferred locations.",
                                               [job.location] if job.location else []))

    # Employment type
    if preferences.employment_types:
        if not job.employment_types:
            unknown_employment_reason = create_reason(JobFilterReasonCode.UNKNOWN_EMPLOYMENT_TYPE, "The job's employment type could not be determined.", job.raw_employment_types)

            if policy.allow_unknown_employment_type:
                warnings.append(unknown_employment_reason)

            else:
                rejection_reasons.append(unknown_employment_reason)

        elif (not employment_types_match(job.employment_types, preferences.employment_types) and policy.reject_employment_type_mismatch):
            rejection_reasons.append(create_reason(JobFilterReasonCode.EMPLOYMENT_TYPE_MISMATCH,
                                                   "The job's employment type does not match the candidate's preference.",
                                                   [employment_type.value for employment_type in job.employment_types]))

    # Seniority
    if detected_seniority is None:
        unknown_seniority_reason = create_reason(JobFilterReasonCode.UNKNOWN_SENIORITY,"No explicit seniority level was detected.",[job.title])
        if policy.allow_unknown_seniority:
            warnings.append(unknown_seniority_reason)
        else:
            rejection_reasons.append(unknown_seniority_reason)

    elif (preferences.seniority_levels and detected_seniority not in preferences.seniority_levels and policy.reject_seniority_mismatch):
        rejection_reasons.append(create_reason(JobFilterReasonCode.SENIORITY_MISMATCH,
                                               f"The detected seniority '{detected_seniority.value}' is not among the preferred levels.",[job.title]))

    # Required keywords
    if preferences.required_keywords:
        if policy.require_all_required_keywords:
            missing_required_keywords = [keyword for keyword in preferences.required_keywords if keyword not in matched_required_keywords]

        elif matched_required_keywords:
            missing_required_keywords = []

        else:
            missing_required_keywords = list(preferences.required_keywords)

        if (missing_required_keywords and policy.reject_missing_required_keywords):
            rejection_reasons.append(create_reason(JobFilterReasonCode.REQUIRED_KEYWORD_MISSING,"The job does not contain all required candidate keywords.",
                                                    missing_required_keywords))

    # Excluded keywords
    if (matched_excluded_keywords and policy.reject_excluded_keywords):
        rejection_reasons.append(create_reason(JobFilterReasonCode.EXCLUDED_KEYWORD_PRESENT, "The job contains excluded keywords.", matched_excluded_keywords))

    # Language requirements
    for requirement in language_requirements:
        candidate_level = profile_languages.get(requirement.language)
        if requirement.language not in profile_languages:
            if policy.reject_language_mismatch:
                rejection_reasons.append(create_reason(JobFilterReasonCode.LANGUAGE_MISMATCH,
                                                       f"The job requires {requirement.language}, which is not listed in the candidate profile or accepted languages.",
                                                       [requirement.evidence]))
            continue

        if requirement.minimum_level is None:
            continue

        if candidate_level is None:
            unknown_level_reason = create_reason(JobFilterReasonCode.UNKNOWN_LANGUAGE_LEVEL,
                                                 f"The candidate's {requirement.language} level is unknown, while the job specifies {requirement.minimum_level}.",
                                                 [requirement.evidence])

            if policy.allow_unknown_language_level:
                warnings.append(unknown_level_reason)

            else:
                rejection_reasons.append(unknown_level_reason)

            continue

        if (not language_level_satisfies(candidate_level, requirement.minimum_level) and policy.reject_language_level_mismatch):
            rejection_reasons.append(create_reason(JobFilterReasonCode.LANGUAGE_LEVEL_MISMATCH,
                                                  f"The candidate's {requirement.language} level '{candidate_level}' is below the required '{requirement.minimum_level}'.",
                                                  [requirement.evidence]))

    return JobFilterDecision(job=job,
                             accepted=not rejection_reasons,
                             rejection_reasons=rejection_reasons,
                             warnings=warnings,
                             matched_roles=matched_roles,
                             detected_seniority=detected_seniority,
                             detected_work_modes=detected_work_modes,
                             detected_language_requirements=language_requirements,
                             matched_required_keywords=matched_required_keywords,
                             matched_excluded_keywords=matched_excluded_keywords)


def filter_jobs_for_candidate(request: JobFilteringRequest) -> JobFilteringResponse:
    """Filter all submitted jobs against candidate constraints."""
    profile_languages = build_candidate_language_map(request.profile, request.preferences.preferred_languages)
    decisions = [evaluate_job(job, profile_languages=profile_languages, preferences=request.preferences,
                 policy=request.policy) for job in request.jobs]
    accepted_jobs = [decision for decision in decisions if decision.accepted]
    rejected_jobs = [decision for decision in decisions if not decision.accepted]
    warning_count = sum(len(decision.warnings) for decision in decisions)
    return JobFilteringResponse(
        accepted_jobs=accepted_jobs,
        rejected_jobs=rejected_jobs,
        statistics=JobFilteringStatistics(received_count=len(decisions), accepted_count=len(accepted_jobs),
                                          rejected_count=len(rejected_jobs), warning_count=warning_count))
