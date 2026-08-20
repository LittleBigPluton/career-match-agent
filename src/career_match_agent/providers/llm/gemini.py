from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from career_match_agent.providers.llm.base import (
    LLMProviderResponseError,
    LLMProviderUnavailableError,
    StructuredOutputT
)


class GeminiStructuredLLMProvider:
    """Structured-output provider backed by Gemini."""

    provider_name = "gemini"

    def __init__(self, *, api_key: str, model_name: str, timeout_seconds: float) -> None:
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)))

    async def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[StructuredOutputT]) -> StructuredOutputT:
        schema = response_model.model_json_schema()

        try:
            response = await self._client.aio.models.generate_content(model=self.model_name,
                                                                      contents=user_prompt,
                                                                      config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0,
                                                                                                         response_mime_type="application/json", response_json_schema=schema))

        except errors.ClientError as error:
            raise LLMProviderResponseError(f"Gemini rejected the request: {error.message}") from error

        except errors.APIError as error:
            raise LLMProviderUnavailableError("Gemini could not be reached.") from error

        response_content = response.text
        if not response_content:
            raise LLMProviderResponseError("Gemini returned an empty response.")

        try:
            return response_model.model_validate_json(response_content)

        except ValidationError as error:
            raise LLMProviderResponseError("Gemini returned invalid structured output.") from error
