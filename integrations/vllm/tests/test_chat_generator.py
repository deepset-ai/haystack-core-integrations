# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from haystack.utils.auth import Secret
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, chat_completion_chunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from haystack_integrations.components.generators.vllm.chat.chat_generator import (
    VLLMChatGenerator,
    _convert_chat_completion_to_chat_message,
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


def _make_reasoning_chunk(text):
    chunk = ChatCompletionChunk(
        id="test-id",
        model=MODEL,
        object="chat.completion.chunk",
        choices=[
            chat_completion_chunk.Choice(
                finish_reason=None,
                logprobs=None,
                index=0,
                delta=chat_completion_chunk.ChoiceDelta(content=None, role=None),
            )
        ],
        created=int(datetime.now(tz=timezone.utc).timestamp()),
        usage=None,
    )
    chunk.choices[0].delta.reasoning = text
    return chunk


def _make_content_chunk(text, finish_reason=None):
    return ChatCompletionChunk(
        id="test-id",
        model=MODEL,
        object="chat.completion.chunk",
        choices=[
            chat_completion_chunk.Choice(
                finish_reason=finish_reason,
                logprobs=None,
                index=0,
                delta=chat_completion_chunk.ChoiceDelta(content=text, role=None),
            )
        ],
        created=int(datetime.now(tz=timezone.utc).timestamp()),
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
        created=int(datetime.now(tz=timezone.utc).timestamp()),
        usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
    )


@pytest.fixture
def completion_with_reasoning(completion):
    completion.choices[0].message.reasoning = "The user asked about the capital of France. I know it's Paris."
    return completion


@pytest.fixture
def mock_chat_completion(completion):
    with patch("openai.resources.chat.completions.Completions.create") as mock:
        mock.return_value = completion
        yield mock


@pytest.fixture
def mock_chat_completion_with_reasoning(completion_with_reasoning):
    with patch("openai.resources.chat.completions.Completions.create") as mock:
        mock.return_value = completion_with_reasoning
        yield mock


@pytest.fixture
def mock_async_chat_completion(completion):
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock:
        mock.return_value = completion
        yield mock


@pytest.fixture
def mock_async_chat_completion_with_reasoning(completion_with_reasoning):
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock:
        mock.return_value = completion_with_reasoning
        yield mock


class TestConvertChatCompletionToChatMessage:
    def test_without_reasoning(self, completion):
        message = _convert_chat_completion_to_chat_message(completion, completion.choices[0])

        assert message.text == "Paris is the capital of France."
        assert message.meta["model"] == MODEL
        assert message.reasoning is None

    def test_with_reasoning(self, completion_with_reasoning):
        message = _convert_chat_completion_to_chat_message(
            completion_with_reasoning, completion_with_reasoning.choices[0]
        )

        assert message.text == "Paris is the capital of France."
        assert message.reasoning is not None
        assert "capital of France" in message.reasoning.reasoning_text

    def test_preserves_tool_calls(self):
        completion = ChatCompletion(
            id="test-id",
            model=MODEL,
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        tool_calls=[
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    ),
                )
            ],
            created=int(datetime.now(tz=timezone.utc).timestamp()),
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        message = _convert_chat_completion_to_chat_message(completion, completion.choices[0])

        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].tool_name == "weather"
        assert message.tool_calls[0].arguments == {"city": "Paris"}

    def test_skips_malformed_tool_call_arguments(self):
        completion = ChatCompletion(
            id="test-id",
            model=MODEL,
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        tool_calls=[
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {"name": "weather", "arguments": "not-valid-json"},
                            }
                        ],
                    ),
                )
            ],
            created=int(datetime.now(tz=timezone.utc).timestamp()),
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        message = _convert_chat_completion_to_chat_message(completion, completion.choices[0])

        assert len(message.tool_calls) == 0


class TestVLLMChatGeneratorInit:
    def test_init_default(self):
        component = VLLMChatGenerator(model=MODEL)

        assert component.model == MODEL
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.api_base_url == "http://localhost:8000/v1"
        assert component._client is None

    def test_init_with_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "test-vllm-key")
        component = VLLMChatGenerator(model=MODEL)

        assert component.api_key.resolve_value() == "test-vllm-key"

    def test_init_with_parameters(self):
        component = VLLMChatGenerator(
            model=MODEL,
            api_key=Secret.from_token("my-key"),
            api_base_url="http://my-server:8000/v1",
            generation_kwargs={"max_tokens": 512, "temperature": 0.7},
            timeout=60.0,
            max_retries=3,
        )

        assert component._client is None
        assert component._async_client is None
        assert not component._is_warmed_up
        assert component.tools is None
        assert component.http_client_kwargs is None
        assert component.streaming_callback is None
        assert component.api_base_url == "http://my-server:8000/v1"
        assert component.model == MODEL
        assert component.api_key.resolve_value() == "my-key"
        assert component.generation_kwargs == {"max_tokens": 512, "temperature": 0.7}
        assert component.timeout == 60.0
        assert component.max_retries == 3


class TestVLLMChatGeneratorWarmUp:
    def test_warm_up_creates_clients(self):
        component = VLLMChatGenerator(model=MODEL)
        assert component._client is None

        component.warm_up()

        assert component._client is not None
        assert component._async_client is not None
        assert component._is_warmed_up is True

    def test_warm_up_is_idempotent(self):
        component = VLLMChatGenerator(model=MODEL)
        component.warm_up()
        first_client = component._client

        component.warm_up()

        assert component._client is first_client


class TestVLLMChatGeneratorSerde:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "test-key")
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

    def test_from_dict_with_streaming_callback(self):
        data = {
            "type": "haystack_integrations.components.generators.vllm.chat.chat_generator.VLLMChatGenerator",
            "init_parameters": {
                "model": MODEL,
                "api_key": {"type": "env_var", "env_vars": ["VLLM_API_KEY"], "strict": False},
                "api_base_url": "http://localhost:8000/v1",
                "generation_kwargs": {},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "tools": None,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }
        component = VLLMChatGenerator.from_dict(data)

        assert component.streaming_callback is not None


class TestVLLMChatGeneratorRun:
    def test_run(self, mock_chat_completion):  # noqa: ARG002
        component = VLLMChatGenerator(model=MODEL)
        response = component.run([ChatMessage.from_user("What's the capital of France")])

        assert len(response["replies"]) == 1
        assert response["replies"][0].text == "Paris is the capital of France."
        assert response["replies"][0].reasoning is None

    def test_run_with_reasoning(self, mock_chat_completion_with_reasoning):  # noqa: ARG002
        component = VLLMChatGenerator(model=MODEL)
        response = component.run([ChatMessage.from_user("What's the capital of France")])

        reply = response["replies"][0]
        assert reply.text == "Paris is the capital of France."
        assert reply.reasoning is not None
        assert "capital of France" in reply.reasoning.reasoning_text

    def test_run_passes_generation_kwargs(self, mock_chat_completion):
        component = VLLMChatGenerator(
            model=MODEL,
            generation_kwargs={"max_tokens": 100, "temperature": 0.5},
        )
        component.run([ChatMessage.from_user("Hello")])

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.5

    def test_run_empty_messages(self):
        component = VLLMChatGenerator(model=MODEL)
        assert component.run([]) == {"replies": []}

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


@pytest.mark.asyncio
class TestVLLMChatGeneratorRunAsync:
    async def test_run_async_empty_messages(self):
        component = VLLMChatGenerator(model=MODEL)
        assert await component.run_async([]) == {"replies": []}

    async def test_run_async(self, mock_async_chat_completion):  # noqa: ARG002
        component = VLLMChatGenerator(model=MODEL)
        response = await component.run_async([ChatMessage.from_user("Hello")])

        assert len(response["replies"]) == 1
        assert response["replies"][0].text == "Paris is the capital of France."
        assert response["replies"][0].reasoning is None

    async def test_run_async_with_reasoning(self, mock_async_chat_completion_with_reasoning):  # noqa: ARG002
        component = VLLMChatGenerator(model=MODEL)
        response = await component.run_async([ChatMessage.from_user("Hello")])

        reply = response["replies"][0]
        assert reply.reasoning is not None
        assert "capital of France" in reply.reasoning.reasoning_text

    async def test_run_async_streaming(self):
        openai_chunks = [
            _make_content_chunk("Hello", finish_reason="stop"),
        ]

        chunks_received = []

        async def callback(chunk):
            chunks_received.append(chunk)

        component = VLLMChatGenerator(model=MODEL, streaming_callback=callback)
        component.warm_up()

        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock:
            mock.return_value = AsyncMockStream(openai_chunks)
            response = await component.run_async([ChatMessage.from_user("Hello")])

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1
        assert response["replies"][0].text == "Hello"

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
        component.warm_up()

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


NO_THINKING_KWARGS = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
THINKING_KWARGS = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


@pytest.mark.integration
class TestVLLMChatGeneratorLiveRun:
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

    def test_live_run_with_reasoning_and_parallel_tool_calls(self):

        @tool
        def weather(city: Annotated[str, "The city to get the weather for"]) -> str:
            """Get the weather in a given city."""
            return f"The weather in {city} is sunny"

        component = VLLMChatGenerator(
            model=MODEL,
            tools=[weather],
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
