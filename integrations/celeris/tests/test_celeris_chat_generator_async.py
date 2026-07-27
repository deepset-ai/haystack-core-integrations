import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytz
from haystack.dataclasses import ChatMessage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack_integrations.components.generators.celeris.chat.chat_generator import CelerisChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


@pytest.fixture
def mock_async_chat_completion():
    """
    Mock the async Celeris API completion response and reuse it for async tests.
    """
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
    ) as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="celeris-1",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


class TestCelerisChatGeneratorAsync:
    @pytest.mark.asyncio
    async def test_run_async(self, chat_messages, mock_async_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        response = await component.run_async(chat_messages)

        assert isinstance(response, dict)
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    @pytest.mark.asyncio
    async def test_run_async_quantizes_max_tokens(self, chat_messages, mock_async_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(generation_kwargs={"max_tokens": 100})
        await component.run_async(chat_messages)

        _, kwargs = mock_async_chat_completion.call_args
        assert kwargs["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_run_async_rejects_response_format(self, chat_messages, mock_async_chat_completion, monkeypatch):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator()
        with pytest.raises(ValueError, match="does not support the 'response_format' parameter"):
            await component.run_async(chat_messages, generation_kwargs={"response_format": {"type": "json_object"}})
        mock_async_chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_preserves_warm_ping_max_tokens(
        self, chat_messages, mock_async_chat_completion, monkeypatch
    ):
        monkeypatch.setenv("CELERIS_API_KEY", "fake-api-key")
        component = CelerisChatGenerator(generation_kwargs={"max_tokens": 1})
        await component.run_async(chat_messages)

        _, kwargs = mock_async_chat_completion.call_args
        assert kwargs["max_tokens"] == 1

    @pytest.mark.skipif(
        not os.environ.get("CELERIS_API_KEY", None),
        reason="Export an env var called CELERIS_API_KEY containing the Celeris API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        component = CelerisChatGenerator()
        results = await component.run_async(
            [ChatMessage.from_user("What's the capital of France? Answer with one word.")]
        )

        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0].text
