from career_match_agent.core.config import get_settings
from career_match_agent.services.profile_extractor import (
    CandidateProfileExtractor,
    StructuredCandidateProfileExtractor
)
from career_match_agent.providers.arbeitnow import ArbeitnowJobProvider
from career_match_agent.providers.base import JobProvider
from functools import lru_cache
from career_match_agent.services.embedding import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider
)
from career_match_agent.services.job_evaluator import (
    JobReportGenerator,
    StructuredJobReportGenerator
)
from career_match_agent.services.search_planner import (
    SearchPlanner,
    StructuredSearchPlanner
)
from career_match_agent.providers.llm.base import StructuredLLMProvider
from career_match_agent.providers.llm.factory import create_llm_provider


@lru_cache(maxsize=1)
def get_llm_provider() -> StructuredLLMProvider:
    settings = get_settings()
    return create_llm_provider(settings)

def get_profile_extractor() -> CandidateProfileExtractor:
    """Create the configured candidate-profile extractor."""
    settings = get_settings()
    return StructuredCandidateProfileExtractor(llm_provider=get_llm_provider(), maximum_cv_characters=(settings.max_cv_text_characters))

def get_job_provider() -> JobProvider:
    """Create the configured job provider."""
    settings = get_settings()
    if settings.job_provider == "arbeitnow":
        return ArbeitnowJobProvider(base_url=settings.arbeitnow_base_url, timeout_seconds=(settings.arbeitnow_timeout_seconds),
                                    maximum_pages=(settings.arbeitnow_max_pages), user_agent=settings.http_user_agent)

    raise RuntimeError(
        f"Unsupported job provider: {settings.job_provider}.")

@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide embedding provider."""
    settings = get_settings()
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(model_name=settings.embedding_model, device=settings.embedding_device, batch_size=settings.embedding_batch_size)

    raise RuntimeError(f"Unsupported embedding provider: {settings.embedding_provider}.")

def get_job_report_generator() -> JobReportGenerator:
    """Create the configured job report generator."""
    return StructuredJobReportGenerator(llm_provider=get_llm_provider())

def get_search_planner() -> SearchPlanner:
    """Create the configured agentic search planner."""
    return StructuredSearchPlanner(llm_provider=get_llm_provider())
