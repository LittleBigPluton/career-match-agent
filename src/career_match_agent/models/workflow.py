from datetime import UTC, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)

from career_match_agent.models.agent import (
    AgentSearchRequest,
    AgentSearchResponse,
    AgentWorkflowConfiguration
)
from career_match_agent.models.candidate import (
    CandidateProfile,
    JobPreferences
)
from career_match_agent.models.hiring_agent import HiringAgentAssessment


LLMProviderValue = Literal["ollama", "openai", "gemini"]
JobProviderValue = Literal["arbeitnow", "adzuna", "jooble"]


class WorkflowModel(BaseModel):
    """Base configuration for automated workflow models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class WorkflowLLMSelection(WorkflowModel):
    """LLM provider and model selected for one workflow run."""
    provider: LLMProviderValue
    model: str = Field(min_length=1)

class WorkflowOptions(WorkflowModel):
    """User-selectable configuration for one workflow run."""
    llm: WorkflowLLMSelection
    job_providers: list[JobProviderValue] = Field(min_length=1)
    hiring_agent_role: str | None = None
    agent: AgentWorkflowConfiguration = Field(default_factory=AgentWorkflowConfiguration)
    record_artifacts: bool = False

    @field_validator("job_providers")
    @classmethod
    def deduplicate_job_providers(cls, values: list[JobProviderValue]) -> list[JobProviderValue]:
        return list(dict.fromkeys(values))

class WorkflowLLMMetadata(WorkflowModel):
    """LLM actually used by the workflow."""
    provider: str
    model: str

class PreparedWorkflowState(WorkflowModel):
    """
    Reusable state produced after candidate preprocessing.

    Uploading this file allows CareerMatch to skip CV/profile,
    preference and HackerRank preprocessing.
    """

    schema_version: Literal["career-match-prepared-v1"] = "career-match-prepared-v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_llm: WorkflowLLMMetadata
    agent_request: AgentSearchRequest
    hiring_agent_assessment: (HiringAgentAssessment | None) = None
    
class AutomatedWorkflowResponse(WorkflowModel):
    """Complete user-facing CareerMatch workflow response."""
    llm: WorkflowLLMMetadata
    profile: CandidateProfile
    preferences: JobPreferences
    hiring_agent_assessment: (HiringAgentAssessment | None) = None
    evidence_signal_count: int = Field(ge=0)
    prepared_state: PreparedWorkflowState
    agent_request: AgentSearchRequest
    agent: AgentSearchResponse
    artifact_run_id: str | None = None

class ProviderCapability(WorkflowModel):
    """Whether one provider has enough local configuration to be selected."""
    name: str
    configured: bool

class WorkflowCapabilities(WorkflowModel):
    """Runtime choices exposed to the web UI."""
    llm_providers: list[ProviderCapability]
    job_providers: list[ProviderCapability]
    default_llm_provider: str
    default_llm_model: str
    default_job_providers: list[str]
