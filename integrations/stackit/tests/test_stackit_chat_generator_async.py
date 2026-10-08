# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils.auth import Secret
from openai import AsyncStream

from haystack_integrations.components.generators.stackit.chat.chat_generator import STACKITChatGenerator


@pytest.mark.asyncio
class TestSTACKITChatGeneratorAsync:
    async def test_run_async_with_string_input(self, mock_async_chat_completion: MagicMock) -> None:

        component = STACKITChatGenerator(model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"))
        response = await component.run_async("What's the capital of France?")

        _, kwargs = mock_async_chat_completion.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "What's the capital of France?"}]

        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)

    async def test_run_with_empty_messages(self, mock_async_chat_completion):
        component = STACKITChatGenerator(model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"))
        assert await component.run_async([]) == {"replies": []}
        mock_async_chat_completion.assert_not_called()

    async def test_run_with_reasoning_async(self, reasoning_completion):
        component = STACKITChatGenerator(model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"))
        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create",
            new_callable=AsyncMock,
            return_value=reasoning_completion,
        ):
            response = await component.run_async([ChatMessage.from_user("What is 2 + 2?")])

        message = response["replies"][0]
        assert message.text == "4"
        assert message.reasoning.reasoning_text == "We need a brief reply. 2+2=4."
        assert message.meta["model"] == "openai/gpt-oss-20b"
        assert message.meta["finish_reason"] == "stop"
        assert message.meta["usage"] == reasoning_completion.usage.model_dump()

    async def test_run_with_reasoning_streaming_async(self, reasoning_chunks):
        async def stream():
            for chunk in reasoning_chunks:
                yield chunk

        chunks = []

        async def streaming_callback(chunk):
            chunks.append(chunk)

        component = STACKITChatGenerator(
            model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock, return_value=stream()
        ):
            response = await component.run_async([ChatMessage.from_user("What is 2 + 2?")])

        assert len(chunks) == 17
        assert chunks[0].content == "" and chunks[0].index is None
        assert [c.reasoning.reasoning_text for c in chunks[1:14]] == [
            "We",
            " need",
            " a",
            " brief",
            " reply",
            ".",
            " ",
            "2",
            "+",
            "2",
            "=",
            "4",
            ".",
        ]
        assert [c.start for c in chunks[1:14]] == [True] + [False] * 12
        assert all(c.index == 0 and c.content == "" and not c.tool_calls for c in chunks[1:14])
        assert chunks[14].content == "4" and chunks[14].start and chunks[14].reasoning is None
        assert chunks[15].finish_reason == "stop"
        assert chunks[16].meta["usage"] == reasoning_chunks[-1].usage.model_dump()
        assert all(c.meta["model"] == "openai/gpt-oss-20b" and c.component_info is not None for c in chunks)
        message = response["replies"][0]
        assert message.text == "4"
        assert message.reasoning.reasoning_text == "We need a brief reply. 2+2=4."
        assert message.meta["finish_reason"] == "stop"
        assert message.meta["usage"] == reasoning_chunks[-1].usage.model_dump()

    async def test_run_with_reasoning_and_tools_streaming_async(self, reasoning_tool_chunk):
        async def stream():
            yield reasoning_tool_chunk

        chunks = []

        async def streaming_callback(chunk):
            chunks.append(chunk)

        component = STACKITChatGenerator(
            model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock, return_value=stream()
        ):
            response = await component.run_async([ChatMessage.from_user("What's the weather in Paris and Berlin?")])

        assert len(chunks) == 2
        assert chunks[0].reasoning.reasoning_text == "Check both cities."
        assert not chunks[0].tool_calls and chunks[0].finish_reason is None
        assert chunks[0].meta["usage"] is None
        assert chunks[1].reasoning is None and chunks[1].finish_reason == "tool_calls"
        assert [call.index for call in chunks[1].tool_calls] == [0, 1]
        message = response["replies"][0]
        assert message.reasoning.reasoning_text == "Check both cities."
        assert message.tool_calls == [
            ToolCall(id="call_1", tool_name="weather", arguments={"city": "Paris"}),
            ToolCall(id="call_2", tool_name="weather", arguments={"city": "Berlin"}),
        ]
        assert message.meta["finish_reason"] == "tool_calls"
        assert message.meta["usage"] == reasoning_tool_chunk.usage.model_dump()

    async def test_async_stream_closes_on_cancellation(self, reasoning_chunks):
        component = STACKITChatGenerator(model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"))
        mock_stream = AsyncMock(spec=AsyncStream)
        received_first_chunk = asyncio.Event()
        keep_stream_open = asyncio.Event()
        chunks = []

        async def stream():
            for chunk in reasoning_chunks:
                yield chunk
                await keep_stream_open.wait()

        def callback(chunk):
            chunks.append(chunk)
            received_first_chunk.set()

        mock_stream.__aiter__ = lambda _: stream()
        task = asyncio.create_task(component._handle_async_stream_response(mock_stream, callback))
        try:
            await asyncio.wait_for(received_first_chunk.wait(), timeout=1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        mock_stream.close.assert_awaited_once()
        assert len(chunks) == 1
        assert chunks[0].component_info is not None
