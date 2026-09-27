import logging

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from career_match_agent.providers.llm.base import (
    LLMProviderResponseError,
    LLMProviderUnavailableError,
    StructuredOutputT
)


logger = logging.getLogger(__name__)


class GeminiStructuredLLMProvider:
    """Structured-output provider backed by Gemini."""
    provider_name = "gemini"
    def __init__(self, *, api_key: str, model_name: str, timeout_seconds: float) -> None:
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000),
                                                                                    retry_options=types.HttpRetryOptions(attempts=5,
                                                                                                                         initial_delay=1.0,
                                                                                                                         max_delay=16.0,
                                                                                                                         exp_base=2.0,
                                                                                                                         jitter=1.0,
                                                                                                                         http_status_codes=[408, 429, 500, 502, 503, 504])))

    async def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[StructuredOutputT]) -> StructuredOutputT:
        schema = response_model.model_json_schema()
        try:
            logger.info("Gemini request: model=%r response_model=%s system_chars=%d user_chars=%d", self.model_name, response_model.__name__, len(system_prompt), len(user_prompt))
            response = await self._client.aio.models.generate_content(model=self.model_name,
                                                                      contents=user_prompt,
                                                                      config=types.GenerateContentConfig(system_instruction=system_prompt,
                                                                                                         temperature=0,
                                                                                                         response_mime_type="application/json",
                                                                                                         response_json_schema=schema,
                                                                                                         automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))

        except errors.ClientError as error:
            logger.warning("Gemini client error: code=%s status=%s message=%s model=%s", error.code, error.status, error.message, self.model_name)
            raise LLMProviderResponseError(f"Gemini rejected the request: {error.message}") from error

        except errors.ServerError as error:
            logger.warning("Gemini server error: code=%s status=%s message=%s model=%s", error.code, error.status, error.message, self.model_name)
            raise LLMProviderUnavailableError(f"Gemini is temporarily unavailable ({error.code} {error.status}).") from error

        except errors.APIError as error:
            logger.warning("Gemini API error: code=%s status=%s message=%s model=%s", error.code, error.status, error.message, self.model_name)
            raise LLMProviderUnavailableError("Gemini could not be reached.") from error

        response_content = response.text
        if not response_content:
            raise LLMProviderResponseError("Gemini returned an empty response.")

        try:
            return response_model.model_validate_json(response_content)

        except ValidationError as error:
            raise LLMProviderResponseError("Gemini returned invalid structured output.") from error
