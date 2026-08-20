import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from career_match_agent.providers.llm.base import LLMProviderResponseError
from career_match_agent.providers.llm.ollama import OllamaStructuredLLMProvider


class SampleStructuredOutput(BaseModel):
    message: str


def test_ollama_generates_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = OllamaStructuredLLMProvider(base_url="http://127.0.0.1:11434", model_name="test-model", timeout_seconds=10.0)

        fake_chat = AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content='{"message":"hello from ollama"}')))
        fake_client = SimpleNamespace(chat=fake_chat)
        monkeypatch.setattr(provider, "_client", fake_client)
        result = await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)
        assert isinstance(result, SampleStructuredOutput)
        assert result.message == "hello from ollama"

        fake_chat.assert_awaited_once()
        call_arguments = (fake_chat.await_args.kwargs)
        assert (call_arguments["model"] == "test-model")
        assert (call_arguments["format"] == SampleStructuredOutput.model_json_schema())

    asyncio.run(run_test())


def test_ollama_rejects_invalid_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = OllamaStructuredLLMProvider(base_url="http://127.0.0.1:11434", model_name="test-model", timeout_seconds=10.0)
        fake_chat = AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content='{"wrong_field":"value"}')))
        monkeypatch.setattr(provider, "_client", SimpleNamespace(chat=fake_chat))
        with pytest.raises(LLMProviderResponseError):
            await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)
    asyncio.run(run_test())


def test_ollama_rejects_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = OllamaStructuredLLMProvider(base_url="http://127.0.0.1:11434", model_name="test-model", timeout_seconds=10.0)
        fake_chat = AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content="")))
        monkeypatch.setattr(provider, "_client", SimpleNamespace(chat=fake_chat))
        with pytest.raises(LLMProviderResponseError):
            await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)

    asyncio.run(run_test())
