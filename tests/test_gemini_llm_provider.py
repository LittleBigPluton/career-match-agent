import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from career_match_agent.providers.llm.base import LLMProviderResponseError
from career_match_agent.providers.llm.gemini import GeminiStructuredLLMProvider


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
