import pytest

from career_match_agent.core.config import Settings
from career_match_agent.providers.llm.base import (
    LLMProviderConfigurationError
)
from career_match_agent.providers.llm.factory import create_llm_provider
from career_match_agent.providers.llm.gemini import GeminiStructuredLLMProvider
from career_match_agent.providers.llm.ollama import OllamaStructuredLLMProvider
from career_match_agent.providers.llm.openai import OpenAIStructuredLLMProvider


def test_factory_creates_ollama_provider() -> None:
    settings = Settings(llm_provider="ollama", llm_model="gemma3:4b")
    provider = create_llm_provider(settings=settings)
    assert isinstance(provider, OllamaStructuredLLMProvider)
    assert provider.provider_name == "ollama"
    assert provider.model_name == "gemma3:4b"


def test_factory_creates_openai_provider() -> None:
    settings = Settings(llm_provider="openai", llm_model="test-openai-model", openai_api_key="test-openai-key")
    provider = create_llm_provider(settings=settings)
    assert isinstance(provider, OpenAIStructuredLLMProvider)
    assert provider.provider_name == "openai"
    assert provider.model_name == "test-openai-model"


def test_factory_creates_gemini_provider() -> None:
    settings = Settings(llm_provider="gemini", llm_model="test-gemini-model", gemini_api_key="test-gemini-key")
    provider = create_llm_provider(settings=settings)
    assert isinstance(provider, GeminiStructuredLLMProvider)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "test-gemini-model"


def test_openai_provider_requires_api_key() -> None:
    settings = Settings(llm_provider="openai", llm_model="test-openai-model", openai_api_key=None)
    with pytest.raises(LLMProviderConfigurationError):
        create_llm_provider(settings=settings)


def test_gemini_provider_requires_api_key() -> None:
    settings = Settings(llm_provider="gemini", llm_model="test-gemini-model", gemini_api_key=None)

    with pytest.raises(LLMProviderConfigurationError):
        create_llm_provider(settings=settings)
