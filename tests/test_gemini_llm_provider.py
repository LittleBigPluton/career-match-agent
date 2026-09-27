import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pydantic import BaseModel
from google.genai import errors

from career_match_agent.providers.llm.gemini import GeminiStructuredLLMProvider
from career_match_agent.providers.llm.base import (
    LLMProviderResponseError,
    LLMProviderUnavailableError
)



class SampleStructuredOutput(BaseModel):
    message: str


def test_gemini_generates_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = GeminiStructuredLLMProvider(api_key="test-api-key", model_name="test-model", timeout_seconds=100.0)
        fake_generate_content = AsyncMock(return_value=SimpleNamespace(text='{"message":"hello from gemini"}'))
        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content,)))
        monkeypatch.setattr(provider, "_client", fake_client)
        result = await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)
        assert isinstance(result, SampleStructuredOutput)
        assert (result.message == "hello from gemini")

        fake_generate_content.assert_awaited_once()
        call_arguments = (fake_generate_content.await_args.kwargs)
        assert (call_arguments["model"] == "test-model")
        assert (call_arguments["contents"] == "User prompt")

        config = call_arguments["config"]
        assert config.system_instruction == "System prompt"
        assert config.temperature == 0
        assert config.response_mime_type == "application/json"
        assert (config.response_json_schema == SampleStructuredOutput.model_json_schema())
        assert config.automatic_function_calling is not None
        assert config.automatic_function_calling.disable is True

    asyncio.run(run_test())


def test_gemini_rejects_invalid_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = GeminiStructuredLLMProvider(api_key="test-api-key", model_name="test-model", timeout_seconds=100.0)
        fake_generate_content = AsyncMock(return_value=SimpleNamespace(text='{"wrong_field":"value"}'))
        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content)))
        monkeypatch.setattr(provider, "_client", fake_client)
        with pytest.raises(LLMProviderResponseError):
            await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)

    asyncio.run(run_test())


def test_gemini_rejects_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = GeminiStructuredLLMProvider(api_key="test-api-key", model_name="test-model", timeout_seconds=100.0)
        fake_generate_content = AsyncMock(return_value=SimpleNamespace(text=""))
        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content)))
        monkeypatch.setattr(provider, "_client", fake_client)
        with pytest.raises(LLMProviderResponseError):
            await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)

    asyncio.run(run_test())

def test_gemini_maps_client_error_to_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = GeminiStructuredLLMProvider(api_key="test-api-key", model_name="test-model", timeout_seconds=100.0)
        client_error = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Invalid request."}})
        fake_generate_content = AsyncMock(side_effect=client_error)
        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content)))
        monkeypatch.setattr(provider, "_client", fake_client)
        with pytest.raises(LLMProviderResponseError, match="Gemini rejected the request"):
            await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)

        fake_generate_content.assert_awaited_once()

    asyncio.run(run_test())


def test_gemini_maps_server_error_to_unavailable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = GeminiStructuredLLMProvider(api_key="test-api-key", model_name="test-model", timeout_seconds=100.0)
        server_error = errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": ("This model is currently experiencing high demand.")}})
        fake_generate_content = AsyncMock(side_effect=server_error)
        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content)))
        monkeypatch.setattr(provider, "_client", fake_client)
        with pytest.raises(LLMProviderUnavailableError, match="Gemini is temporarily unavailable"):
            await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)

        fake_generate_content.assert_awaited_once()

    asyncio.run(run_test())
