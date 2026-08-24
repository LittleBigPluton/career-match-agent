from career_match_agent.core.config import Settings
from career_match_agent.providers.llm.base import (
    LLMProviderConfigurationError,
    LLMProviderName,
    StructuredLLMProvider
)
from career_match_agent.providers.llm.gemini import GeminiStructuredLLMProvider
from career_match_agent.providers.llm.ollama import OllamaStructuredLLMProvider
from career_match_agent.providers.llm.openai import OpenAIStructuredLLMProvider


def create_llm_provider(settings: Settings) -> StructuredLLMProvider:
    """Create the globally configured LLM provider."""
    if settings.llm_provider == "ollama":
        return OllamaStructuredLLMProvider(base_url=settings.ollama_base_url, model_name=settings.llm_model, timeout_seconds=(settings.llm_timeout_seconds))

    if settings.llm_provider == "openai":
        if settings.openai_api_key is None:
            raise LLMProviderConfigurationError("OpenAI API key is not configured.")

        return OpenAIStructuredLLMProvider(api_key=(settings.openai_api_key.get_secret_value()), model_name=settings.llm_model, timeout_seconds=(settings.llm_timeout_seconds))

    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise LLMProviderConfigurationError("Gemini API key is not configured.")

        return GeminiStructuredLLMProvider(api_key=(settings.gemini_api_key.get_secret_value()), model_name=settings.llm_model, timeout_seconds=(settings.llm_timeout_seconds))

    raise LLMProviderConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}")

def create_llm_provider_for_selection(settings: Settings, *, provider_name: LLMProviderName, model_name: str) -> StructuredLLMProvider:
    """Create an LLM provider selected for one workflow request."""
    cleaned_model_name = model_name.strip()
    if not cleaned_model_name:
        raise LLMProviderConfigurationError("An LLM model name is required.")
    runtime_settings = settings.model_copy(
        update={"llm_provider": provider_name.value, "llm_model": cleaned_model_name})

    return create_llm_provider(runtime_settings)
