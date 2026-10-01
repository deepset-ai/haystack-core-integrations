# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from haystack_integrations.components.generators.otari import OtariChatGenerator

from .conftest import OTARI_API_BASE_URL, requires_api_key

DEFAULT_MODEL = "openai:gpt-5-mini"


@pytest.fixture
def mock_async_chat_completion():
    """Mock the async Otari (OpenAI-compatible) chat completion response, including Otari's cost fields."""
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_chat_completion_create:
        mock_chat_completion_create.return_value = ChatCompletion(
            id="foo",
            model="gpt-5-mini-2025-08-07",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=1750162525,
            usage=CompletionUsage(
                prompt_tokens=57,
                completion_tokens=40,
                total_tokens=97,
                cost_usd="0.000279",
                pricing_source="defaults",
            ),
        )
        yield mock_chat_completion_create


@pytest.mark.asyncio
class TestOtariChatGeneratorAsync:
    async def test_run_async(self, chat_messages, mock_async_chat_completion, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "fake-api-key")
        component = OtariChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = await component.run_async(chat_messages)

        # the generation kwargs are passed on to the Otari endpoint
        _, kwargs = mock_async_chat_completion.call_args
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert isinstance(reply, ChatMessage)
        assert reply.text == "Hello world!"
        assert reply.meta["usage"]["cost_usd"] == "0.000279"

    @requires_api_key
    @pytest.mark.integration
    async def test_live_run_async(self):
        component = OtariChatGenerator(api_base_url=OTARI_API_BASE_URL)
        results = await component.run_async([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-5-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @requires_api_key
    @pytest.mark.integration
    async def test_live_run_with_tools_streaming_async(self, tools):
        chunks = []

        async def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        component = OtariChatGenerator(api_base_url=OTARI_API_BASE_URL, tools=tools, streaming_callback=callback)
        results = await component.run_async(
            [ChatMessage.from_user("What's the weather like in Paris?")],
            generation_kwargs={"tool_choice": "auto"},
        )

        assert len(chunks) > 1
        assert any(chunk.tool_calls for chunk in chunks), "No tool calls received in streaming"

        tool_message = results["replies"][0]
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)
        assert tool_message.meta["finish_reason"] == "tool_calls"
        tool_call = tool_message.tool_call
        assert tool_call.id
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
