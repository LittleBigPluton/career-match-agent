from career_match_agent.core.config import get_settings
from career_match_agent.services.profile_extractor import (
    CandidateProfileExtractor,
    StructuredCandidateProfileExtractor
)
from career_match_agent.providers.arbeitnow import ArbeitnowJobProvider
from career_match_agent.providers.adzuna import AdzunaJobProvider
from career_match_agent.providers.jooble import JoobleJobProvider
from career_match_agent.providers.base import JobProvider
from career_match_agent.providers.composite import CompositeJobProvider
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
    """Create all configured job providers."""
    settings = get_settings()
    providers: list[JobProvider] = []
    for provider_name in settings.job_providers:
        if provider_name == "arbeitnow":
            providers.append(ArbeitnowJobProvider(base_url=(settings.arbeitnow_base_url),
                                                  timeout_seconds=(settings.arbeitnow_timeout_seconds),
                                                  maximum_pages=(settings.arbeitnow_max_pages),
                                                  user_agent=(settings.http_user_agent)))

        elif provider_name == "adzuna":
            if not settings.adzuna_app_id:
                raise RuntimeError("Adzuna is enabled but CAREER_MATCH_ADZUNA_APP_ID is not configured.")

            if not settings.adzuna_app_key:
                raise RuntimeError("Adzuna is enabled but CAREER_MATCH_ADZUNA_APP_KEY is not configured.")

            providers.append(AdzunaJobProvider(base_url=(settings.adzuna_base_url),
                                               country=(settings.adzuna_country),
                                               app_id=(settings.adzuna_app_id),
                                               app_key=(settings.adzuna_app_key),
                                               timeout_seconds=(settings.adzuna_timeout_seconds),
                                               results_per_page=(settings.adzuna_results_per_page),
                                               maximum_requests=(settings.adzuna_max_requests_per_search),
                                               user_agent=(settings.http_user_agent)))

        elif provider_name == "jooble":
            if not settings.jooble_api_key:
                raise RuntimeError("Jooble is enabled but CAREER_MATCH_JOOBLE_API_KEY is not configured.")

            providers.append(JoobleJobProvider(base_url=(settings.jooble_base_url),
                                               api_key=(settings.jooble_api_key),
                                               default_location=(settings.jooble_default_location),
                                               timeout_seconds=(settings.jooble_timeout_seconds),
                                               results_per_page=(settings.jooble_results_per_page),
                                               maximum_requests=(settings.jooble_max_requests_per_search),
                                               user_agent=(settings.http_user_agent)))

        else:
            raise RuntimeError(f"Unsupported job provider: {provider_name}.")

    if len(providers) == 1:
        return providers[0]

    return CompositeJobProvider(providers)

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
