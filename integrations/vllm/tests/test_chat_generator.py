# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import tool
from haystack.utils.auth import Secret
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, chat_completion_chunk

from haystack_integrations.components.generators.vllm.chat.chat_generator import (
    VLLMChatGenerator,
)

MODEL = "Qwen/Qwen3-0.6B"


class MockStream(Stream[ChatCompletionChunk]):
    def __init__(self, mock_chunks, client=None, *args, **kwargs):
        client = client or MagicMock()
        super().__init__(client=client, *args, **kwargs)  # noqa: B026
        self.mock_chunks = mock_chunks

    def __stream__(self) -> Iterator[ChatCompletionChunk]:
        yield from self.mock_chunks


class AsyncMockStream(AsyncStream[ChatCompletionChunk]):
    def __init__(self, mock_chunks):
        self.mock_chunks = mock_chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index < len(self.mock_chunks):
            chunk = self.mock_chunks[self._index]
            self._index += 1
            return chunk
        raise StopAsyncIteration


def _make_reasoning_chunk(text, content=None, tool_calls=None, role=None, finish_reason=None):
    chunk = ChatCompletionChunk(
        id="test-id",
        model=MODEL,
        object="chat.completion.chunk",
        choices=[
            chat_completion_chunk.Choice(
                finish_reason=finish_reason,
                logprobs=None,
                index=0,
                delta=chat_completion_chunk.ChoiceDelta(content=content, role=role, tool_calls=tool_calls),
            )
        ],
        created=int(datetime.now(tz=UTC).timestamp()),
        usage=None,
    )
    chunk.choices[0].delta.reasoning = text
    return chunk


def _make_content_chunk(text, tool_calls=None, role=None, finish_reason=None):
    return ChatCompletionChunk(
        id="test-id",
        model=MODEL,
        object="chat.completion.chunk",
        choices=[
            chat_completion_chunk.Choice(
                finish_reason=finish_reason,
                logprobs=None,
                index=0,
                delta=chat_completion_chunk.ChoiceDelta(content=text, role=role, tool_calls=tool_calls),
            )
        ],
        created=int(datetime.now(tz=UTC).timestamp()),
        usage=None,
    )


@pytest.fixture
def completion():
    return ChatCompletion(
        id="test-id",
        model=MODEL,
        object="chat.completion",
        choices=[
            {
                "finish_reason": "stop",
                "logprobs": None,
                "index": 0,
                "message": {"content": "Paris is the capital of France.", "role": "assistant"},
            }
        ],
        created=int(datetime.now(tz=UTC).timestamp()),
        usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
    )


@pytest.fixture
def completion_with_reasoning(completion):
    completion.choices[0].message.reasoning = "The user asked about the capital of France. I know it's Paris."
    return completion


@pytest.fixture
def weather_tool():
    @tool
    def weather(city: Annotated[str, "The city to get the weather for"]) -> str:
        """Get the weather in a given city."""
        return f"The weather in {city} is sunny"

    return weather


class TestConvertChatCompletionToChatMessage:
    def test_without_reasoning(self, completion):
        message = VLLMChatGenerator(model=MODEL)._convert_chat_completion_to_chat_message(
            completion, completion.choices[0]
        )

        assert message.text == "Paris is the capital of France."
        assert message.meta["model"] == MODEL
        assert message.reasoning is None

    def test_with_reasoning(self, completion_with_reasoning):
        message = VLLMChatGenerator(model=MODEL)._convert_chat_completion_to_chat_message(
            completion_with_reasoning, completion_with_reasoning.choices[0]
        )

        assert message.text == "Paris is the capital of France."
        assert message.reasoning is not None
        assert "capital of France" in message.reasoning.reasoning_text


class TestPrepareAPICall:
    @pytest.mark.parametrize(
        "last_message",
        [ChatMessage.from_assistant(text=None), ChatMessage.from_assistant(reasoning="only reasoning")],
    )
    def test_drops_trailing_empty_assistant(self, last_message):
        messages = [ChatMessage.from_user("Hi"), last_message]
        assert VLLMChatGenerator(model=MODEL)._prepare_api_call(messages=messages)["messages"] == [
            {"role": "user", "content": "Hi"}
        ]

    def test_keeps_trailing_assistant_with_tool_calls(self):
        tool_call = ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
        messages = [ChatMessage.from_user("Hi"), ChatMessage.from_assistant(tool_calls=[tool_call])]
        assert VLLMChatGenerator(model=MODEL)._prepare_api_call(messages=messages)["messages"][-1] == {
            "role": "assistant",
            "tool_calls": [
                {"id": "123", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
            ],
        }


class TestInitialization:
    def test_init_default(self):
        component = VLLMChatGenerator(model=MODEL)

        assert component.model == MODEL
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.api_base_url == "http://localhost:8000/v1"
        assert component.client is None

    def test_init_with_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "test-vllm-key")
        component = VLLMChatGenerator(model=MODEL)

        assert component.api_key.resolve_value() == "test-vllm-key"

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_VLLM_API_KEY", raising=False)
        component = VLLMChatGenerator(model=MODEL, api_key=Secret.from_env_var("MISSING_VLLM_API_KEY"))

        with pytest.raises(ValueError, match="MISSING_VLLM_API_KEY"):
            component.warm_up()


class TestSerialization:
    def test_to_dict(self):
        component = VLLMChatGenerator(
            model=MODEL,
            generation_kwargs={"max_tokens": 512},
        )
        data = component.to_dict()

        assert data["init_parameters"]["model"] == MODEL
        assert data["init_parameters"]["api_key"] == {"env_vars": ["VLLM_API_KEY"], "strict": False, "type": "env_var"}
        assert data["init_parameters"]["api_base_url"] == "http://localhost:8000/v1"
        assert data["init_parameters"]["generation_kwargs"] == {"max_tokens": 512}
        assert data["init_parameters"]["http_client_kwargs"] is None
        assert data["init_parameters"]["tools"] is None
        assert data["init_parameters"]["timeout"] is None
        assert data["init_parameters"]["max_retries"] is None
        assert data["init_parameters"]["streaming_callback"] is None

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.generators.vllm.chat.chat_generator.VLLMChatGenerator",
            "init_parameters": {
                "model": MODEL,
                "api_key": {"type": "env_var", "env_vars": ["VLLM_API_KEY"], "strict": False},
                "api_base_url": "http://my-server:8000/v1",
                "generation_kwargs": {"max_tokens": 512},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "tools": None,
                "timeout": 30.0,
                "max_retries": 5,
                "http_client_kwargs": None,
            },
        }
        component = VLLMChatGenerator.from_dict(data)

        assert isinstance(component, VLLMChatGenerator)
        assert component.model == MODEL
        assert component.generation_kwargs == {"max_tokens": 512}
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "http://my-server:8000/v1"
        assert component.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert component.timeout == 30.0
        assert component.max_retries == 5
        assert component.http_client_kwargs is None
        assert component.tools is None


class TestRun:
    def test_run_streaming(self):
        openai_chunks = [
            _make_content_chunk("Hello", finish_reason="stop"),
        ]

        chunks_received = []
        component = VLLMChatGenerator(model=MODEL, streaming_callback=chunks_received.append)
        component.warm_up()

        with patch("openai.resources.chat.completions.Completions.create") as mock:
            mock.return_value = MockStream(openai_chunks, cast_to=None, response=None, client=None)
            response = component.run([ChatMessage.from_user("Hello")])

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1
        assert response["replies"][0].text == "Hello"

    def test_run_streaming_with_reasoning(self):
        """Test that streaming with reasoning correctly sets start=True on the first reasoning and content chunks."""
        openai_chunks = [
            _make_reasoning_chunk("Okay"),
            _make_reasoning_chunk(", the"),
            _make_reasoning_chunk(" capital"),
            _make_reasoning_chunk(" is Paris.\n"),
            _make_content_chunk("\n\n"),
            _make_content_chunk("Paris"),
            _make_content_chunk(None, finish_reason="stop"),
        ]

        streaming_chunks = []
        component = VLLMChatGenerator(model=MODEL, streaming_callback=streaming_chunks.append)
        component.warm_up()

        with patch("openai.resources.chat.completions.Completions.create") as mock:
            mock.return_value = MockStream(openai_chunks, cast_to=None, response=None, client=None)
            result = component.run([ChatMessage.from_user("Hello")])

        assert result["replies"][0].text == "\n\nParis"
        assert result["replies"][0].reasoning.reasoning_text == "Okay, the capital is Paris.\n"

        assert len(streaming_chunks) == 7
        # chunk 0: first reasoning chunk -> start=True
        assert streaming_chunks[0].reasoning is not None
        assert streaming_chunks[0].start is True
        # chunks 1-3: subsequent reasoning chunks -> start=False
        for i in range(1, 4):
            assert streaming_chunks[i].reasoning is not None
            assert streaming_chunks[i].start is False
        # chunk 4: first content chunk after reasoning -> start=True
        assert streaming_chunks[4].content == "\n\n"
        assert streaming_chunks[4].start is True
        # chunks 5-6: subsequent content chunks -> start=False
        assert streaming_chunks[5].start is False
        assert streaming_chunks[6].start is False
        assert streaming_chunks[6].meta["finish_reason"] == "stop"
        assert result["replies"][0].meta["finish_reason"] == "stop"

    def test_run_streaming_splits_reasoning_and_content(self):
        openai_chunks = [
            _make_reasoning_chunk("Let me think", content="", role="assistant"),
            _make_reasoning_chunk(" ok.", content="Paris is"),
            _make_content_chunk(" the capital.", finish_reason="stop"),
        ]

        streaming_chunks = []
        component = VLLMChatGenerator(model=MODEL, streaming_callback=streaming_chunks.append)
        component.warm_up()

        with patch("openai.resources.chat.completions.Completions.create") as mock:
            mock.return_value = MockStream(openai_chunks, cast_to=None, response=None, client=None)
            result = component.run([ChatMessage.from_user("What is the capital of France?")])

        reply = result["replies"][0]
        assert reply.text == "Paris is the capital."
        assert reply.reasoning is not None
        assert reply.reasoning.reasoning_text == "Let me think ok."
        assert reply.meta["finish_reason"] == "stop"

        assert len(streaming_chunks) == 4
        assert [c.reasoning.reasoning_text for c in streaming_chunks[:2]] == ["Let me think", " ok."]
        assert [c.start for c in streaming_chunks[:2]] == [True, False]
        assert [c.index for c in streaming_chunks[:2]] == [0, 0]
        assert all(c.content == "" for c in streaming_chunks[:2])
        assert all(c.meta["finish_reason"] is None for c in streaming_chunks[:3])

        assert streaming_chunks[2].reasoning is None
        assert streaming_chunks[2].content == "Paris is"
        assert streaming_chunks[2].start is True
        assert streaming_chunks[2].index == 1

        assert streaming_chunks[3].content == " the capital."
        assert streaming_chunks[3].start is False
        assert streaming_chunks[3].index == 1
        assert streaming_chunks[3].finish_reason == "stop"
        assert streaming_chunks[3].meta["finish_reason"] == "stop"

    def test_run_streaming_splits_reasoning_and_tool_calls(self):
        tool_call = chat_completion_chunk.ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            type="function",
            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(name="weather", arguments='{"city": "Paris"}'),
        )
        openai_chunks = [
            _make_reasoning_chunk("I need the weather.", role="assistant"),
            _make_reasoning_chunk(" Calling the tool.", tool_calls=[tool_call], finish_reason="tool_calls"),
        ]

        streaming_chunks = []
        component = VLLMChatGenerator(model=MODEL, streaming_callback=streaming_chunks.append)
        component.warm_up()

        with patch("openai.resources.chat.completions.Completions.create") as mock:
            mock.return_value = MockStream(openai_chunks, cast_to=None, response=None, client=None)
            result = component.run([ChatMessage.from_user("What is the weather in Paris?")])

        reply = result["replies"][0]
        assert reply.text is None
        assert reply.reasoning is not None
        assert reply.reasoning.reasoning_text == "I need the weather. Calling the tool."
        assert reply.tool_calls == [ToolCall(id="call_1", tool_name="weather", arguments={"city": "Paris"})]
        assert reply.meta["finish_reason"] == "tool_calls"

        assert len(streaming_chunks) == 3
        assert [c.reasoning.reasoning_text for c in streaming_chunks[:2]] == [
            "I need the weather.",
            " Calling the tool.",
        ]
        assert [c.start for c in streaming_chunks[:2]] == [True, False]
        assert [c.index for c in streaming_chunks[:2]] == [0, 0]
        assert all(c.finish_reason is None and c.meta["finish_reason"] is None for c in streaming_chunks[:2])

        assert streaming_chunks[2].reasoning is None
        assert streaming_chunks[2].tool_calls is not None
        assert streaming_chunks[2].tool_calls[0].tool_name == "weather"
        assert streaming_chunks[2].tool_calls[0].arguments == '{"city": "Paris"}'
        assert streaming_chunks[2].tool_calls[0].index == 0
        assert streaming_chunks[2].start is True
        assert streaming_chunks[2].index == 1
        assert streaming_chunks[2].finish_reason == "tool_calls"
        assert streaming_chunks[2].meta["finish_reason"] == "tool_calls"

    def test_run_streaming_splits_reasoning_content_and_tool_calls(self):
        openai_chunks = [
            _make_reasoning_chunk("I need the weather.", role="assistant"),
            _make_reasoning_chunk(
                " in JSON within the specified XML tags.\n",
                content="\n\n",
                tool_calls=[
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(name="weather"),
                    ),
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"city":'),
                    ),
                ],
            ),
            _make_content_chunk(
                None,
                tool_calls=[
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments=' "Paris"}'),
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]

        streaming_chunks = []
        component = VLLMChatGenerator(model=MODEL, streaming_callback=streaming_chunks.append)
        component.warm_up()

        with patch("openai.resources.chat.completions.Completions.create") as mock:
            mock.return_value = MockStream(openai_chunks, cast_to=None, response=None, client=None)
            result = component.run([ChatMessage.from_user("What is the weather in Paris?")])

        reply = result["replies"][0]
        assert reply.text == "\n\n"
        assert reply.reasoning is not None
        assert reply.reasoning.reasoning_text == "I need the weather. in JSON within the specified XML tags.\n"
        assert reply.tool_calls == [ToolCall(id="call_1", tool_name="weather", arguments={"city": "Paris"})]
        assert reply.meta["finish_reason"] == "tool_calls"

        assert len(streaming_chunks) == 5
        assert [c.reasoning.reasoning_text for c in streaming_chunks[:2]] == [
            "I need the weather.",
            " in JSON within the specified XML tags.\n",
        ]
        assert [c.start for c in streaming_chunks[:2]] == [True, False]
        assert [c.index for c in streaming_chunks[:2]] == [0, 0]

        assert streaming_chunks[2].reasoning is None
        assert streaming_chunks[2].content == "\n\n"
        assert streaming_chunks[2].tool_calls is None
        assert streaming_chunks[2].start is True
        assert streaming_chunks[2].index == 1

        assert streaming_chunks[3].reasoning is None
        assert streaming_chunks[3].content == ""
        assert [(tc.index, tc.id, tc.tool_name, tc.arguments) for tc in streaming_chunks[3].tool_calls] == [
            (0, "call_1", "weather", None),
            (0, None, None, '{"city":'),
        ]
        assert streaming_chunks[3].start is True
        assert streaming_chunks[3].index == 2

        assert streaming_chunks[4].reasoning is None
        assert streaming_chunks[4].content == ""
        assert [(tc.index, tc.tool_name, tc.arguments) for tc in streaming_chunks[4].tool_calls] == [
            (0, None, ' "Paris"}'),
        ]
        assert streaming_chunks[4].start is False
        assert streaming_chunks[4].index == 2

        assert all(c.finish_reason is None and c.meta["finish_reason"] is None for c in streaming_chunks[:4])
        assert streaming_chunks[4].finish_reason == "tool_calls"
        assert streaming_chunks[4].meta["finish_reason"] == "tool_calls"

    def test_run_streaming_splits_content_and_tool_calls(self):
        tool_call = chat_completion_chunk.ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            type="function",
            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(name="weather", arguments='{"city": "Paris"}'),
        )
        openai_chunks = [
            _make_content_chunk("\n\n", tool_calls=[tool_call], role="assistant", finish_reason="tool_calls"),
        ]

        streaming_chunks = []
        component = VLLMChatGenerator(model=MODEL, streaming_callback=streaming_chunks.append)
        component.warm_up()

        with patch("openai.resources.chat.completions.Completions.create") as mock:
            mock.return_value = MockStream(openai_chunks, cast_to=None, response=None, client=None)
            result = component.run([ChatMessage.from_user("What is the weather in Paris?")])

        reply = result["replies"][0]
        assert reply.text == "\n\n"
        assert reply.reasoning is None
        assert reply.tool_calls == [ToolCall(id="call_1", tool_name="weather", arguments={"city": "Paris"})]
        assert reply.meta["finish_reason"] == "tool_calls"

        assert len(streaming_chunks) == 2
        assert streaming_chunks[0].content == "\n\n"
        assert streaming_chunks[0].tool_calls is None
        assert streaming_chunks[0].start is True
        assert streaming_chunks[0].index == 0
        assert streaming_chunks[0].finish_reason is None
        assert streaming_chunks[0].meta["finish_reason"] is None

        assert streaming_chunks[1].content == ""
        assert streaming_chunks[1].tool_calls is not None
        assert streaming_chunks[1].tool_calls[0].tool_name == "weather"
        assert streaming_chunks[1].tool_calls[0].arguments == '{"city": "Paris"}'
        assert streaming_chunks[1].tool_calls[0].index == 0
        assert streaming_chunks[1].start is True
        assert streaming_chunks[1].index == 1
        assert streaming_chunks[1].finish_reason == "tool_calls"
        assert streaming_chunks[1].meta["finish_reason"] == "tool_calls"

    def test_run_streaming_tool_calls_with_usage_chunk(self):
        tool_call = chat_completion_chunk.ChoiceDeltaToolCall(
            index=0,
            id="call_1",
            type="function",
            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(name="weather", arguments='{"city": "Paris"}'),
        )
        openai_chunks = [
            _make_content_chunk(None, tool_calls=[tool_call], role="assistant", finish_reason="tool_calls"),
            ChatCompletionChunk(
                id="test-id",
                model=MODEL,
                object="chat.completion.chunk",
                choices=[],
                created=int(datetime.now(tz=UTC).timestamp()),
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            ),
        ]

        streaming_chunks = []
        component = VLLMChatGenerator(model=MODEL, streaming_callback=streaming_chunks.append)
        component.warm_up()

        with patch("openai.resources.chat.completions.Completions.create") as mock:
            mock.return_value = MockStream(openai_chunks, cast_to=None, response=None, client=None)
            result = component.run([ChatMessage.from_user("What is the weather in Paris?")])

        reply = result["replies"][0]
        assert reply.tool_calls == [ToolCall(id="call_1", tool_name="weather", arguments={"city": "Paris"})]
        assert reply.meta["finish_reason"] == "tool_calls"
        assert reply.meta["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }
        assert len(streaming_chunks) == 2
        assert streaming_chunks[0].index == 0
        assert streaming_chunks[0].start is True

    @pytest.mark.asyncio
    async def test_run_async_streaming_with_reasoning(self):
        """Test that async streaming with reasoning sets start=True on first reasoning and content chunks."""
        openai_chunks = [
            _make_reasoning_chunk("Okay"),
            _make_reasoning_chunk(", the"),
            _make_reasoning_chunk(" capital"),
            _make_reasoning_chunk(" is Paris.\n"),
            _make_content_chunk("\n\n"),
            _make_content_chunk("Paris"),
            _make_content_chunk(None, finish_reason="stop"),
        ]

        streaming_chunks = []

        async def callback(chunk):
            streaming_chunks.append(chunk)

        component = VLLMChatGenerator(model=MODEL, streaming_callback=callback)

        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock:
            mock.return_value = AsyncMockStream(openai_chunks)
            result = await component.run_async([ChatMessage.from_user("Hello")])

        assert result["replies"][0].text == "\n\nParis"
        assert result["replies"][0].reasoning.reasoning_text == "Okay, the capital is Paris.\n"

        assert len(streaming_chunks) == 7
        assert streaming_chunks[0].reasoning is not None
        assert streaming_chunks[0].start is True
        for i in range(1, 4):
            assert streaming_chunks[i].reasoning is not None
            assert streaming_chunks[i].start is False
        assert streaming_chunks[4].content == "\n\n"
        assert streaming_chunks[4].start is True
        assert streaming_chunks[5].start is False
        assert streaming_chunks[6].start is False
        assert streaming_chunks[6].meta["finish_reason"] == "stop"
        assert result["replies"][0].meta["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_run_async_streaming_splits_reasoning_content_and_tool_calls(self):
        openai_chunks = [
            _make_reasoning_chunk("I need the weather.", role="assistant"),
            _make_reasoning_chunk(
                " in JSON within the specified XML tags.\n",
                content="\n\n",
                tool_calls=[
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(name="weather"),
                    ),
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"city":'),
                    ),
                ],
            ),
            _make_content_chunk(
                None,
                tool_calls=[
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments=' "Paris"}'),
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]

        streaming_chunks = []

        async def callback(chunk):
            streaming_chunks.append(chunk)

        component = VLLMChatGenerator(model=MODEL, streaming_callback=callback)

        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock:
            mock.return_value = AsyncMockStream(openai_chunks)
            result = await component.run_async([ChatMessage.from_user("What is the weather in Paris?")])

        reply = result["replies"][0]
        assert reply.text == "\n\n"
        assert reply.reasoning is not None
        assert reply.reasoning.reasoning_text == "I need the weather. in JSON within the specified XML tags.\n"
        assert reply.tool_calls == [ToolCall(id="call_1", tool_name="weather", arguments={"city": "Paris"})]
        assert reply.meta["finish_reason"] == "tool_calls"

        assert len(streaming_chunks) == 5
        assert [c.reasoning.reasoning_text for c in streaming_chunks[:2]] == [
            "I need the weather.",
            " in JSON within the specified XML tags.\n",
        ]
        assert [c.start for c in streaming_chunks[:2]] == [True, False]
        assert [c.index for c in streaming_chunks[:2]] == [0, 0]

        assert streaming_chunks[2].reasoning is None
        assert streaming_chunks[2].content == "\n\n"
        assert streaming_chunks[2].tool_calls is None
        assert streaming_chunks[2].start is True
        assert streaming_chunks[2].index == 1

        assert streaming_chunks[3].reasoning is None
        assert streaming_chunks[3].content == ""
        assert [(tc.index, tc.id, tc.tool_name, tc.arguments) for tc in streaming_chunks[3].tool_calls] == [
            (0, "call_1", "weather", None),
            (0, None, None, '{"city":'),
        ]
        assert streaming_chunks[3].start is True
        assert streaming_chunks[3].index == 2

        assert streaming_chunks[4].reasoning is None
        assert streaming_chunks[4].content == ""
        assert [(tc.index, tc.tool_name, tc.arguments) for tc in streaming_chunks[4].tool_calls] == [
            (0, None, ' "Paris"}'),
        ]
        assert streaming_chunks[4].start is False
        assert streaming_chunks[4].index == 2

        assert all(c.finish_reason is None and c.meta["finish_reason"] is None for c in streaming_chunks[:4])
        assert streaming_chunks[4].finish_reason == "tool_calls"
        assert streaming_chunks[4].meta["finish_reason"] == "tool_calls"


NO_THINKING_KWARGS = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
THINKING_KWARGS = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


@pytest.mark.integration
class TestIntegration:
    @pytest.mark.parametrize("generation_kwargs", [NO_THINKING_KWARGS, THINKING_KWARGS])
    def test_live_run(self, generation_kwargs):
        component = VLLMChatGenerator(model=MODEL, generation_kwargs=generation_kwargs)
        response = component.run([ChatMessage.from_user("What is the capital of France?")])

        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert "paris" in reply.text.lower()

        if generation_kwargs == NO_THINKING_KWARGS:
            assert reply.reasoning is None
        else:
            assert reply.reasoning is not None
            assert len(reply.reasoning.reasoning_text) > 0

    @pytest.mark.parametrize("generation_kwargs", [NO_THINKING_KWARGS, THINKING_KWARGS])
    def test_live_run_streaming(self, generation_kwargs):
        chunks_received = []
        component = VLLMChatGenerator(
            model=MODEL,
            generation_kwargs=generation_kwargs,
            streaming_callback=chunks_received.append,
        )
        response = component.run([ChatMessage.from_user("What is the capital of France?")])

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1
        assert "paris" in response["replies"][0].text.lower()
        if generation_kwargs == THINKING_KWARGS:
            assert response["replies"][0].reasoning is not None
            assert len(response["replies"][0].reasoning.reasoning_text) > 0
        else:
            assert response["replies"][0].reasoning is None

    @pytest.mark.parametrize("generation_kwargs", [NO_THINKING_KWARGS, THINKING_KWARGS])
    def test_live_run_after_contentless_assistant_message(self, generation_kwargs):
        # After a discarded tool call, the Agent calls the LLM again with the contentless reply as the last message
        component = VLLMChatGenerator(model=MODEL, generation_kwargs=generation_kwargs)
        messages = [
            ChatMessage.from_user("What is the capital of France? Answer in one short sentence."),
            ChatMessage.from_assistant(text=None),
        ]
        response = component.run(messages)

        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert reply.text
        assert "paris" in reply.text.lower()
        assert "<think>" not in reply.text

    def test_live_run_with_reasoning_and_parallel_tool_calls(self, weather_tool):
        component = VLLMChatGenerator(
            model=MODEL,
            tools=[weather_tool],
            generation_kwargs=THINKING_KWARGS,
        )
        response = component.run([ChatMessage.from_user("What is the weather in Paris? And in Berlin?")])

        assert len(response["replies"]) == 1
        message = response["replies"][0]
        assert message.reasoning is not None
        assert len(message.reasoning.reasoning_text) > 0

        tool_calls = message.tool_calls
        assert tool_calls[0].tool_name == "weather"
        assert tool_calls[0].arguments == {"city": "Paris"}
        assert tool_calls[1].tool_name == "weather"
        assert tool_calls[1].arguments == {"city": "Berlin"}

    def test_live_run_with_structured_output(self):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "capital_info",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"capital": {"type": "string"}, "population": {"type": "number"}},
                    "required": ["capital", "population"],
                },
            },
        }
        component = VLLMChatGenerator(
            model=MODEL,
            # reasoning produces more reliable JSON output
            generation_kwargs={**THINKING_KWARGS, "response_format": response_format, "temperature": 0.0},
        )
        response = component.run(
            [ChatMessage.from_user("What's the capital of France and its population? Respond in JSON.")]
        )

        assert len(response["replies"]) == 1
        response_data = json.loads(response["replies"][0].text)
        assert isinstance(response_data, dict)
        assert "capital" in response_data
        assert "paris" in response_data["capital"].lower()
        assert "population" in response_data

    @pytest.mark.asyncio
    @pytest.mark.parametrize("generation_kwargs", [NO_THINKING_KWARGS, THINKING_KWARGS])
    async def test_live_run_async(self, generation_kwargs):
        component = VLLMChatGenerator(model=MODEL, generation_kwargs=generation_kwargs)
        response = await component.run_async(
            [ChatMessage.from_user("What is the capital of France? Answer in one word.")]
        )

        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert "paris" in reply.text.lower()
        if generation_kwargs == THINKING_KWARGS:
            assert reply.reasoning is not None
            assert len(reply.reasoning.reasoning_text) > 0
        else:
            assert reply.reasoning is None
