from career_match_agent.core.config import Settings
from career_match_agent.models.workflow import WorkflowLLMSelection
from career_match_agent.providers.base import JobProvider
from career_match_agent.providers.job_factory import create_job_provider
from career_match_agent.providers.llm.base import (
    LLMProviderName,
    StructuredLLMProvider
)
from career_match_agent.providers.llm.factory import create_llm_provider_for_selection


class WorkflowRuntimeFactory:
    """Create request-scoped workflow providers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_llm(self, selection: WorkflowLLMSelection) -> StructuredLLMProvider:
        return create_llm_provider_for_selection(self.settings, provider_name=LLMProviderName(selection.provider), model_name=selection.model)

    def create_jobs(self, provider_names: list[str]) -> JobProvider:
        return create_job_provider(self.settings, provider_names=provider_names)
