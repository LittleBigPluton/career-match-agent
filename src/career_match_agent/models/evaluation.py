from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator
)

from career_match_agent.models.candidate import (
    CandidateProfile,
    JobPreferences
)
from career_match_agent.models.hiring_agent import CandidateEvidenceSignal
from career_match_agent.models.ranking import RankedJob


def clean_string_list(values: list[str]) -> list[str]:
    """Clean strings while preserving their original order."""
    cleaned_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        cleaned_value = value.strip()

        if not cleaned_value:
            continue

        if cleaned_value in seen_values:
            continue

        seen_values.add(cleaned_value)
        cleaned_values.append(cleaned_value)

    return cleaned_values


class EvaluationModel(BaseModel):
    """Base configuration for evaluation models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceScope(StrEnum):
    """The subject represented by one evidence item."""
    CANDIDATE = "candidate"
    JOB = "job"
    COMPARISON = "comparison"


class GroundingEvidenceItem(EvaluationModel):
    """One evidence item available to the report generator."""
    evidence_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9:_-]*$")
    scope: EvidenceScope
    label: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=2000)


class JobRecommendation(StrEnum):
    """Categorical job recommendation."""
    STRONG_MATCH = "strong_match"
    MATCH = "match"
    POSSIBLE_MATCH = "possible_match"
    WEAK_MATCH = "weak_match"


class EvaluationConfidence(StrEnum):
    """Confidence based on available supporting evidence."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GroundedStatement(EvaluationModel):
    """An evidence-backed statement."""
    text: str = Field(min_length=1, max_length=1200)
    evidence_ids: list[str] = Field(min_length=1, max_length=10)

    @field_validator("evidence_ids")
    @classmethod
    def clean_evidence_ids(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)


class GroundedFinding(EvaluationModel):
    """A titled evidence-backed strength, gap or risk."""
    title: str = Field(min_length=1, max_length=150)
    explanation: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=10)

    @field_validator("evidence_ids")
    @classmethod
    def clean_evidence_ids(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)


class JobSuitabilityReportDraft(EvaluationModel):
    """Structured report returned directly by the LLM."""
    source_id: str = Field(min_length=1)
    recommendation: JobRecommendation
    confidence: EvaluationConfidence
    summary: GroundedStatement
    strengths: list[GroundedFinding] = Field(min_length=1, max_length=5)
    gaps: list[GroundedFinding] = Field(default_factory=list, max_length=5)
    risks: list[GroundedFinding] = Field(default_factory=list, max_length=5)
    interview_focus: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("interview_focus")
    @classmethod
    def clean_interview_focus(cls, values: list[str]) -> list[str]:
        return clean_string_list(values)


class JobEvaluationConfiguration(EvaluationModel):
    """Controls evidence construction and report generation."""
    maximum_jobs: int = Field(default=5, ge=1, le=20)
    maximum_candidate_evidence: int = Field(default=40, ge=5, le=100)
    maximum_job_description_chunks: int = Field(default=8, ge=1, le=30)
    description_chunk_characters: int = Field(default=700, ge=300, le=2000)
    fail_fast: bool = False


class JobEvaluationRequest(EvaluationModel):
    """Ranked jobs and candidate evidence submitted for reports."""
    profile: CandidateProfile
    preferences: JobPreferences
    ranked_jobs: list[RankedJob]
    evidence_signals: list[CandidateEvidenceSignal] = Field(default_factory=list)
    configuration: JobEvaluationConfiguration = Field(default_factory=JobEvaluationConfiguration)

    @model_validator(mode="after")
    def validate_ranked_jobs(self) -> Self:
        source_ids = [ranked_job.decision.job.source_id for ranked_job in self.ranked_jobs]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Ranked jobs must have unique source IDs.")

        invalid_jobs = [ranked_job for ranked_job in self.ranked_jobs if not ranked_job.decision.accepted]
        if invalid_jobs:
            raise ValueError("Only jobs accepted by deterministic filtering can be evaluated.")

        return self


class JobReportGrounding(EvaluationModel):
    """Citation-validation information for one report."""
    available_evidence_count: int = Field(ge=0)
    cited_evidence_count: int = Field(ge=0)
    candidate_citation_count: int = Field(ge=0)
    job_citation_count: int = Field(ge=0)
    comparison_citation_count: int = Field(ge=0)


class EvaluatedJobReport(EvaluationModel):
    """Validated suitability report and cited evidence."""
    rank: int = Field(ge=1)
    hybrid_score: float = Field(ge=0, le=100)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    report: JobSuitabilityReportDraft
    cited_evidence: list[GroundingEvidenceItem]
    grounding: JobReportGrounding


class JobEvaluationFailure(EvaluationModel):
    """One job whose report could not be generated."""
    rank: int = Field(ge=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    error: str = Field(min_length=1)


class JobEvaluationMetadata(EvaluationModel):
    """Information about the report-generation provider."""
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)


class JobEvaluationStatistics(EvaluationModel):
    """Summary counts for one evaluation request."""
    received_count: int = Field(ge=0)
    attempted_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)


class JobEvaluationResponse(EvaluationModel):
    """Complete evidence-grounded report response."""
    generation: JobEvaluationMetadata
    reports: list[EvaluatedJobReport]
    failures: list[JobEvaluationFailure]
    statistics: JobEvaluationStatistics
