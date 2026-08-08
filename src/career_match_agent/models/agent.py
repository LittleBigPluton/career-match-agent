from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)

from career_match_agent.models.candidate import (
    CandidateProfile,
    JobPreferences
)
from career_match_agent.models.evaluation import (
    JobEvaluationConfiguration,
    JobEvaluationResponse
)
from career_match_agent.models.hiring_agent import CandidateEvidenceSignal
from career_match_agent.models.job import JobSearchStatistics
from career_match_agent.models.matching import (
    JobFilterPolicy,
    JobFilteringStatistics
)
from career_match_agent.models.ranking import (
    HybridRankingConfiguration,
    HybridRankingResponse
)


def clean_string_list(values: list[str]) -> list[str]:
    """Remove blank and case-insensitive duplicate strings."""
    cleaned_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        cleaned_value = value.strip()
        if not cleaned_value:
            continue

        comparison_value = cleaned_value.casefold()
        if comparison_value in seen_values:
            continue

        seen_values.add(comparison_value)
        cleaned_values.append(cleaned_value)

    return cleaned_values


class AgentModel(BaseModel):
    """Base configuration for agent workflow models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentSearchPlan(AgentModel):
    """LLM-generated retrieval strategy."""
    keywords: list[str] = Field(min_length=1, max_length=10)
    max_pages: int = Field(default=1, ge=1, le=5)
    maximum_results: int = Field(default=100, ge=1, le=100)
    rationale: str = Field(min_length=1, max_length=800)

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, values: list[str]) -> list[str]:
        cleaned_values = clean_string_list(values)
        if not cleaned_values:
            raise ValueError("At least one search keyword is required.")

        return cleaned_values


class AgentWorkflowConfiguration(AgentModel):
    """Controls bounded agent behavior."""
    minimum_accepted_jobs: int = Field(default=5, ge=1, le=50)
    maximum_search_attempts: int = Field(default=2, ge=1, le=3)
    visa_sponsorship: bool | None = None
    filter_policy: JobFilterPolicy = Field(default_factory=JobFilterPolicy)
    ranking: HybridRankingConfiguration = Field(default_factory=HybridRankingConfiguration)
    evaluation: JobEvaluationConfiguration = Field(default_factory=JobEvaluationConfiguration)

class AgentSearchRequest(AgentModel):
    """Input required by the agentic matching workflow."""
    profile: CandidateProfile
    preferences: JobPreferences
    evidence_signals: list[CandidateEvidenceSignal] = Field(default_factory=list)
    configuration: AgentWorkflowConfiguration = Field(default_factory=AgentWorkflowConfiguration)

class AgentTraceEntry(AgentModel):
    """One observable workflow decision."""
    step: str = Field(min_length=1)
    attempt: int | None = Field(default=None, ge=1)
    message: str = Field(min_length=1)

class AgentSearchResponse(AgentModel):
    """Final result of the complete LangGraph workflow."""
    final_search_plan: AgentSearchPlan
    search_attempts: int
    search_statistics: JobSearchStatistics
    filtering_statistics: JobFilteringStatistics
    ranking: HybridRankingResponse
    evaluation: JobEvaluationResponse
    trace: list[AgentTraceEntry]
