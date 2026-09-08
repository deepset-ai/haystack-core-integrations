# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, patch

import pytest
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from haystack_integrations.components.generators.hetzner.chat.chat_generator import HetznerChatGenerator

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

requires_api_key = pytest.mark.skipif(
    not os.environ.get("HETZNER_API_KEY", None),
    reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
)


@pytest.fixture
def mock_async_chat_completion():
    """Mock the async Hetzner (OpenAI-compatible) chat completion response and reuse it for tests."""
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_chat_completion_create:
        mock_chat_completion_create.return_value = ChatCompletion(
            id="foo",
            model=DEFAULT_MODEL,
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
            usage=CompletionUsage(prompt_tokens=57, completion_tokens=40, total_tokens=97),
        )
        yield mock_chat_completion_create


@pytest.mark.asyncio
class TestHetznerChatGeneratorAsync:
    async def test_run_async(self, chat_messages, mock_async_chat_completion, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "fake-api-key")
        component = HetznerChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = await component.run_async(chat_messages)

        # the generation kwargs are passed on to the Hetzner endpoint
        _, kwargs = mock_async_chat_completion.call_args
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)
        assert response["replies"][0].text == "Hello world!"

    @requires_api_key
    @pytest.mark.integration
    async def test_live_run_async(self):
        results = await HetznerChatGenerator().run_async([ChatMessage.from_user("What's the capital of France")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert "Paris" in message.text
        assert DEFAULT_MODEL in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @requires_api_key
    @pytest.mark.integration
    async def test_live_run_with_tools_streaming_async(self, tools):
        chunks = []

        async def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        component = HetznerChatGenerator(tools=tools, streaming_callback=callback)
        results = await component.run_async(
            [ChatMessage.from_user("What's the weather like in Paris?")],
            generation_kwargs={"tool_choice": "auto"},
        )

        assert len(chunks) > 1
        assert any(chunk.meta.get("tool_calls") for chunk in chunks), "No tool calls received in streaming"

        tool_message = results["replies"][0]
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)
        assert tool_message.meta["finish_reason"] == "tool_calls"
        tool_call = tool_message.tool_call
        assert tool_call.id
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
