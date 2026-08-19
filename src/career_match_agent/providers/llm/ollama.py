import httpx
from ollama import AsyncClient, ResponseError
from pydantic import ValidationError

from career_match_agent.providers.llm.base import (
    LLMProviderResponseError,
    LLMProviderUnavailableError,
    StructuredOutputT
)


class OllamaStructuredLLMProvider:
    """Structured-output provider backed by Ollama."""

    provider_name = "ollama"

    def __init__(self, *, base_url: str, model_name: str, timeout_seconds: float) -> None:
        self.model_name = model_name
        self._client = AsyncClient(host=base_url, timeout=timeout_seconds)

    async def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[StructuredOutputT],) -> StructuredOutputT:
        schema = response_model.model_json_schema()
        try:
            response = await self._client.chat(model=self.model_name,
                                               messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                                               format=schema,
                                               options={"temperature": 0, "seed": 42})

        except (ResponseError, httpx.HTTPError, OSError) as error:
            raise LLMProviderUnavailableError("Ollama could not be reached.") from error

        response_content = response.message.content
        if not response_content:
            raise LLMProviderResponseError("Ollama returned an empty response.")

        try:
            return response_model.model_validate_json(response_content)

        except ValidationError as error:
            raise LLMProviderResponseError("Ollama returned invalid structured output.") from error
