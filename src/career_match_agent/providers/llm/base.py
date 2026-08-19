from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import BaseModel


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)

class LLMProviderName(StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMProviderError(RuntimeError):
    """Base error raised by structured LLM providers."""


class LLMProviderUnavailableError(LLMProviderError):
    """Raised when an LLM service cannot be reached."""


class LLMProviderResponseError(LLMProviderError):
    """Raised when an LLM returns an unusable response."""


class LLMProviderConfigurationError(LLMProviderError):
    """Raised when provider configuration is incomplete."""


class StructuredLLMProvider(Protocol):
    """Provider-independent structured language model interface."""

    provider_name: str
    model_name: str

    async def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[StructuredOutputT]) -> StructuredOutputT:
        """Generate output validated against a Pydantic model."""
