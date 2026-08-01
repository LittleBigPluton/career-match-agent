from career_match_agent.core.config import get_settings
from career_match_agent.services.profile_extractor import (
    CandidateProfileExtractor,
    OllamaCandidateProfileExtractor,
)


def get_profile_extractor() -> CandidateProfileExtractor:
    """Create the configured candidate-profile extractor."""
    settings = get_settings()
    return OllamaCandidateProfileExtractor(
        base_url=settings.ollama_base_url,
        model_name=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        maximum_cv_characters=settings.max_cv_text_characters)
