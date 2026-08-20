import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from career_match_agent.providers.llm.base import LLMProviderResponseError
from career_match_agent.providers.llm.openai import OpenAIStructuredLLMProvider


class SampleStructuredOutput(BaseModel):
    message: str


def test_openai_generates_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = OpenAIStructuredLLMProvider(api_key="test-api-key", model_name="test-model", timeout_seconds=10.0)
        expected_output = SampleStructuredOutput(message="hello from openai")
        fake_parse = AsyncMock(return_value=SimpleNamespace(output_parsed=expected_output))
        fake_client = SimpleNamespace(responses=SimpleNamespace(parse=fake_parse))
        monkeypatch.setattr(provider, "_client", fake_client)
        result = await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)
        assert isinstance(result, SampleStructuredOutput)
        assert (result.message == "hello from openai")

        fake_parse.assert_awaited_once()
        call_arguments = (fake_parse.await_args.kwargs)
        assert (call_arguments["model"] == "test-model")
        assert (call_arguments["text_format"] is SampleStructuredOutput)

    asyncio.run(run_test())


def test_openai_rejects_missing_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        provider = OpenAIStructuredLLMProvider(api_key="test-api-key", model_name="test-model", timeout_seconds=10.0)
        fake_parse = AsyncMock(return_value=SimpleNamespace(output_parsed=None))
        fake_client = SimpleNamespace(responses=SimpleNamespace(parse=fake_parse))
        monkeypatch.setattr(provider, "_client", fake_client)
        with pytest.raises(LLMProviderResponseError):
            await provider.generate_structured(system_prompt="System prompt", user_prompt="User prompt", response_model=SampleStructuredOutput)

    asyncio.run(run_test())
