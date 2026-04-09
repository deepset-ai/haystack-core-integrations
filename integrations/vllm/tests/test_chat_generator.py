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
from haystack.dataclasses import ChatMessage, ChatRole, ImageContent
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


@pytest.fixture
def completion():
    return ChatCompletion(
        id="test-id",
        model="Qwen/Qwen3.5-0.8B",
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
def streaming_chunks():
    chunks = [
        ChatCompletionChunk(
            id="test-id",
            model="Qwen/Qwen3.5-0.8B",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason=None,
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(content="Hello", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=timezone.utc).timestamp()),
            usage=None,
        ),
        ChatCompletionChunk(
            id="test-id",
            model="Qwen/Qwen3.5-0.8B",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(content=None, role=None),
                )
            ],
            created=int(datetime.now(tz=timezone.utc).timestamp()),
            usage=None,
        ),
    ]
    return chunks


@pytest.fixture
def streaming_chunks_with_reasoning(streaming_chunks):
    reasoning_chunk = ChatCompletionChunk(
        id="test-id",
        model="Qwen/Qwen3.5-0.8B",
        object="chat.completion.chunk",
        choices=[
            chat_completion_chunk.Choice(
                finish_reason=None,
                logprobs=None,
                index=0,
                delta=chat_completion_chunk.ChoiceDelta(content=None, role="assistant"),
            )
        ],
        created=int(datetime.now(tz=timezone.utc).timestamp()),
        usage=None,
    )
    reasoning_chunk.choices[0].delta.reasoning = "Let me think about this."
    return [reasoning_chunk, *streaming_chunks]


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
def mock_streaming_completion(streaming_chunks):
    with patch("openai.resources.chat.completions.Completions.create") as mock:
        mock.return_value = MockStream(streaming_chunks, cast_to=None, response=None, client=None)
        yield mock


@pytest.fixture
def mock_streaming_completion_with_reasoning(streaming_chunks_with_reasoning):
    with patch("openai.resources.chat.completions.Completions.create") as mock:
        mock.return_value = MockStream(streaming_chunks_with_reasoning, cast_to=None, response=None, client=None)
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


@pytest.fixture
def mock_async_streaming_completion(streaming_chunks):
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock:
        mock.return_value = AsyncMockStream(streaming_chunks)
        yield mock


@pytest.fixture
def mock_async_streaming_completion_with_reasoning(streaming_chunks_with_reasoning):
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock:
        mock.return_value = AsyncMockStream(streaming_chunks_with_reasoning)
        yield mock


class TestConvertChatCompletionToChatMessage:
    def test_without_reasoning(self, completion):
        message = _convert_chat_completion_to_chat_message(completion, completion.choices[0])

        assert message.text == "Paris is the capital of France."
        assert message.meta["model"] == "Qwen/Qwen3.5-0.8B"
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
            model="Qwen/Qwen3.5-0.8B",
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
            model="Qwen/Qwen3.5-0.8B",
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
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")

        assert component.model == "Qwen/Qwen3.5-0.8B"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.api_base_url == "http://localhost:8000/v1"
        assert component._client is None

    def test_init_with_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "test-vllm-key")
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")

        assert component.api_key.resolve_value() == "test-vllm-key"

    def test_init_without_api_key_uses_placeholder(self, monkeypatch):
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")

        assert component.api_key.resolve_value() == "placeholder-api-key"

    def test_init_with_parameters(self):
        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
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
        assert component.model == "Qwen/Qwen3.5-0.8B"
        assert component.api_key.resolve_value() == "my-key"
        assert component.generation_kwargs == {"max_tokens": 512, "temperature": 0.7}
        assert component.timeout == 60.0
        assert component.max_retries == 3


class TestVLLMChatGeneratorWarmUp:
    def test_warm_up_creates_clients(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")
        assert component._client is None

        component.warm_up()

        assert component._client is not None
        assert component._async_client is not None
        assert component._is_warmed_up is True

    def test_warm_up_is_idempotent(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")
        component.warm_up()
        first_client = component._client

        component.warm_up()

        assert component._client is first_client


class TestVLLMChatGeneratorSerde:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "test-key")
        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
            generation_kwargs={"max_tokens": 512},
        )
        data = component.to_dict()

        assert data["init_parameters"]["model"] == "Qwen/Qwen3.5-0.8B"
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
                "model": "Qwen/Qwen3.5-0.8B",
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
        assert component.model == "Qwen/Qwen3.5-0.8B"
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
                "model": "Qwen/Qwen3.5-0.8B",
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
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")
        response = component.run([ChatMessage.from_user("What's the capital of France")])

        assert len(response["replies"]) == 1
        assert response["replies"][0].text == "Paris is the capital of France."
        assert response["replies"][0].reasoning is None

    def test_run_with_reasoning(self, mock_chat_completion_with_reasoning):  # noqa: ARG002
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")
        response = component.run([ChatMessage.from_user("What's the capital of France")])

        reply = response["replies"][0]
        assert reply.text == "Paris is the capital of France."
        assert reply.reasoning is not None
        assert "capital of France" in reply.reasoning.reasoning_text

    def test_run_passes_generation_kwargs(self, mock_chat_completion):
        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
            generation_kwargs={"max_tokens": 100, "temperature": 0.5},
        )
        component.run([ChatMessage.from_user("Hello")])

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.5

    def test_run_empty_messages(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")
        assert component.run([]) == {"replies": []}

    def test_run_streaming(self, mock_streaming_completion):  # noqa: ARG002
        chunks_received = []
        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
            streaming_callback=chunks_received.append,
        )
        response = component.run([ChatMessage.from_user("Hello")])

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1

    def test_run_streaming_with_reasoning(self, mock_streaming_completion_with_reasoning):  # noqa: ARG002
        chunks_received = []
        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
            streaming_callback=chunks_received.append,
        )
        response = component.run([ChatMessage.from_user("Hello")])

        reasoning_chunks = [c for c in chunks_received if c.reasoning]
        assert len(reasoning_chunks) > 0
        assert response["replies"][0].reasoning is not None


@pytest.mark.asyncio
class TestVLLMChatGeneratorRunAsync:
    async def test_run_async_empty_messages(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")
        assert await component.run_async([]) == {"replies": []}

    async def test_run_async(self, mock_async_chat_completion):  # noqa: ARG002
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")
        response = await component.run_async([ChatMessage.from_user("Hello")])

        assert len(response["replies"]) == 1
        assert response["replies"][0].text == "Paris is the capital of France."
        assert response["replies"][0].reasoning is None

    async def test_run_async_with_reasoning(self, mock_async_chat_completion_with_reasoning):  # noqa: ARG002
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B")
        response = await component.run_async([ChatMessage.from_user("Hello")])

        reply = response["replies"][0]
        assert reply.reasoning is not None
        assert "capital of France" in reply.reasoning.reasoning_text

    async def test_run_async_streaming(self, mock_async_streaming_completion):  # noqa: ARG002
        chunks_received = []

        async def callback(chunk):
            chunks_received.append(chunk)

        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B", streaming_callback=callback)
        response = await component.run_async([ChatMessage.from_user("Hello")])

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1

    async def test_run_async_streaming_with_reasoning(self, mock_async_streaming_completion_with_reasoning):  # noqa: ARG002
        chunks_received = []

        async def callback(chunk):
            chunks_received.append(chunk)

        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B", streaming_callback=callback)
        response = await component.run_async([ChatMessage.from_user("Hello")])

        reasoning_chunks = [c for c in chunks_received if c.reasoning]
        assert len(reasoning_chunks) > 0
        assert response["replies"][0].reasoning is not None


THINKING_KWARGS = {"temperature": 0.0, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
NO_THINKING_KWARGS = {"temperature": 0.0, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


@pytest.mark.integration
class TestVLLMChatGeneratorLiveRun:
    def test_live_run(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B", generation_kwargs=NO_THINKING_KWARGS)
        response = component.run([ChatMessage.from_user("What is the capital of France?")])

        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert "paris" in reply.text.lower()
        assert reply.reasoning is None

    def test_live_run_streaming(self):
        chunks_received = []
        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
            generation_kwargs=NO_THINKING_KWARGS,
            streaming_callback=chunks_received.append,
        )
        response = component.run([ChatMessage.from_user("What is the capital of France?")])

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1
        assert "paris" in response["replies"][0].text.lower()

    def test_live_run_with_reasoning(self):
        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
            generation_kwargs=THINKING_KWARGS,
        )
        response = component.run([ChatMessage.from_user("What is 2+2? Answer very briefly.")])

        reply = response["replies"][0]
        assert reply.reasoning is not None
        assert len(reply.reasoning.reasoning_text) > 0
        assert reply.text is not None

    def test_live_run_with_reasoning_and_parallel_tool_calls(self):

        @tool
        def weather(city: Annotated[str, "The city to get the weather for"]) -> str:
            """Get the weather in a given city."""
            return f"The weather in {city} is sunny"

        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
            generation_kwargs=THINKING_KWARGS,
            tools=[weather],
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
            model="Qwen/Qwen3.5-0.8B",
            generation_kwargs={**NO_THINKING_KWARGS, "response_format": response_format},
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

    def test_live_run_with_image(self, test_files_path):
        image_content = ImageContent.from_file_path(test_files_path / "image.png")
        message = ChatMessage.from_user(
            content_parts=["What fruit is in this image? Answer in one word.", image_content]
        )
        component = VLLMChatGenerator(
            model="Qwen/Qwen3.5-0.8B",
            generation_kwargs={**NO_THINKING_KWARGS, "max_tokens": 16},
        )
        response = component.run([message])

        first_reply = response["replies"][0]
        assert isinstance(first_reply, ChatMessage)
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT)
        assert first_reply.text
        assert "apple" in first_reply.text.lower()

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3.5-0.8B", generation_kwargs=NO_THINKING_KWARGS)
        response = await component.run_async(
            [ChatMessage.from_user("What is the capital of France? Answer in one word.")]
        )

        assert len(response["replies"]) == 1
        assert "paris" in response["replies"][0].text.lower()
