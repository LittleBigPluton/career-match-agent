from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from career_match_agent.models.candidate import (
    CandidateProfile,
    JobPreferences,
    SeniorityLevel,
    WorkMode
)
from career_match_agent.models.job import JobPosting


class MatchingModel(BaseModel):
    """Base configuration for job-matching models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class JobFilterReasonCode(StrEnum):
    """Machine-readable filtering reason."""
    ROLE_MISMATCH = "role_mismatch"
    LOCATION_MISMATCH = "location_mismatch"
    WORK_MODE_MISMATCH = "work_mode_mismatch"
    EMPLOYMENT_TYPE_MISMATCH = "employment_type_mismatch"
    SENIORITY_MISMATCH = "seniority_mismatch"
    LANGUAGE_MISMATCH = "language_mismatch"
    LANGUAGE_LEVEL_MISMATCH = "language_level_mismatch"
    REQUIRED_KEYWORD_MISSING = "required_keyword_missing"
    EXCLUDED_KEYWORD_PRESENT = "excluded_keyword_present"
    UNKNOWN_SENIORITY = "unknown_seniority"
    UNKNOWN_WORK_MODE = "unknown_work_mode"
    UNKNOWN_EMPLOYMENT_TYPE = "unknown_employment_type"
    UNKNOWN_LANGUAGE_LEVEL = "unknown_language_level"

class JobFilterReason(MatchingModel):
    """One explanation produced by the deterministic filter."""
    code: JobFilterReasonCode
    message: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


class DetectedLanguageRequirement(MatchingModel):
    """Language requirement found in a job posting."""
    language: str = Field(min_length=1)
    minimum_level: str | None = None
    evidence: str = Field(min_length=1)

class JobFilterPolicy(MatchingModel):
    """Controls how strictly unknown information is handled."""
    reject_role_mismatch: bool = True
    reject_location_mismatch: bool = True
    reject_work_mode_mismatch: bool = True
    reject_employment_type_mismatch: bool = True
    reject_seniority_mismatch: bool = True
    reject_language_mismatch: bool = True
    reject_language_level_mismatch: bool = True
    reject_missing_required_keywords: bool = True
    reject_excluded_keywords: bool = True
    allow_unknown_seniority: bool = True
    allow_unknown_work_mode: bool = True
    allow_unknown_employment_type: bool = True
    allow_unknown_language_level: bool = True
    remote_overrides_location: bool = True
    require_all_required_keywords: bool = True

class JobFilteringRequest(MatchingModel):
    """Candidate information and jobs submitted for filtering."""
    profile: CandidateProfile
    preferences: JobPreferences
    jobs: list[JobPosting]
    policy: JobFilterPolicy = Field(default_factory=JobFilterPolicy)

class JobFilterDecision(MatchingModel):
    """Filtering result for one job posting."""
    job: JobPosting
    accepted: bool
    rejection_reasons: list[JobFilterReason] = Field(default_factory=list)
    warnings: list[JobFilterReason] = Field(default_factory=list)
    matched_roles: list[str] = Field(default_factory=list)
    detected_seniority: SeniorityLevel | None = None
    detected_work_modes: list[WorkMode] = Field(default_factory=list)
    detected_language_requirements: list[DetectedLanguageRequirement] = Field(default_factory=list)
    matched_required_keywords: list[str] = Field(default_factory=list)
    matched_excluded_keywords: list[str] = Field(default_factory=list)

class JobFilteringStatistics(MatchingModel):
    """Summary counts for a filtering operation."""
    received_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)

class JobFilteringResponse(MatchingModel):
    """Complete deterministic filtering response."""
    accepted_jobs: list[JobFilterDecision]
    rejected_jobs: list[JobFilterDecision]
    statistics: JobFilteringStatistics
