from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI
)

from career_match_agent.providers.llm.base import (
    LLMProviderResponseError,
    LLMProviderUnavailableError,
    StructuredOutputT
)


class OpenAIStructuredLLMProvider:
    """Structured-output provider backed by OpenAI."""

    provider_name = "openai"

    def __init__(self, *, api_key: str, model_name: str, timeout_seconds: float) -> None:
        self.model_name = model_name
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

    async def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[StructuredOutputT]) -> StructuredOutputT:
        try:
            response = await self._client.responses.parse(model=self.model_name,
                                                          input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                                                          text_format=response_model)

        except APIConnectionError as error:
            raise LLMProviderUnavailableError("OpenAI could not be reached.") from error

        except APIError as error:
            raise LLMProviderResponseError(f"OpenAI request failed: {error}.") from error

        parsed = response.output_parsed

        if parsed is None:
            raise LLMProviderResponseError("OpenAI returned no structured output.")

        return parsed
