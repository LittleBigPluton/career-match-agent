from career_match_agent.core.config import get_settings
from career_match_agent.services.profile_extractor import (
    CandidateProfileExtractor,
    OllamaCandidateProfileExtractor,
)

from career_match_agent.providers.arbeitnow import (
    ArbeitnowJobProvider
)
from career_match_agent.providers.base import JobProvider

def get_profile_extractor() -> CandidateProfileExtractor:
    """Create the configured candidate-profile extractor."""
    settings = get_settings()
    return OllamaCandidateProfileExtractor(
        base_url=settings.ollama_base_url,
        model_name=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        maximum_cv_characters=settings.max_cv_text_characters)

def get_job_provider() -> JobProvider:
    """Create the configured job provider."""
    settings = get_settings()
    if settings.job_provider == "arbeitnow":
        return ArbeitnowJobProvider(base_url=settings.arbeitnow_base_url, timeout_seconds=(settings.arbeitnow_timeout_seconds),
                                    maximum_pages=(settings.arbeitnow_max_pages), user_agent=settings.http_user_agent)

    raise RuntimeError(
        f"Unsupported job provider: {settings.job_provider}.")
