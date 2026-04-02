# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, chat_completion_chunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

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
        model="Qwen/Qwen3-0.6B",
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
    completion.choices[0].message.reasoning = (
        "The user asked about the capital of France. I know it's Paris."
    )
    return completion


@pytest.fixture
def streaming_chunks():
    chunks = [
        ChatCompletionChunk(
            id="test-id",
            model="Qwen/Qwen3-0.6B",
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
            model="Qwen/Qwen3-0.6B",
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
        model="Qwen/Qwen3-0.6B",
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
        mock.return_value = MockStream(
            streaming_chunks_with_reasoning, cast_to=None, response=None, client=None
        )
        yield mock


@pytest.fixture
def mock_async_chat_completion(completion):
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock:
        mock.return_value = completion
        yield mock


@pytest.fixture
def mock_async_chat_completion_with_reasoning(completion_with_reasoning):
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock:
        mock.return_value = completion_with_reasoning
        yield mock


@pytest.fixture
def mock_async_streaming_completion(streaming_chunks):
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock:
        mock.return_value = AsyncMockStream(streaming_chunks)
        yield mock


@pytest.fixture
def mock_async_streaming_completion_with_reasoning(streaming_chunks_with_reasoning):
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock:
        mock.return_value = AsyncMockStream(streaming_chunks_with_reasoning)
        yield mock


class TestConvertChatCompletionToChatMessage:
    def test_without_reasoning(self, completion):
        message = _convert_chat_completion_to_chat_message(completion, completion.choices[0])

        assert message.text == "Paris is the capital of France."
        assert message.meta["model"] == "Qwen/Qwen3-0.6B"
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
            model="Qwen/Qwen3-0.6B",
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


class TestVLLMChatGeneratorInit:
    def test_init_default(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")

        assert component.model == "Qwen/Qwen3-0.6B"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.client.base_url == "http://localhost:8000/v1/"

    def test_init_with_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "test-vllm-key")
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")

        assert component.client.api_key == "test-vllm-key"

    def test_init_without_api_key_uses_placeholder(self, monkeypatch):
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")

        assert component.client.api_key == "placeholder-api-key"

    def test_init_with_parameters(self):
        component = VLLMChatGenerator(
            model="Qwen/Qwen3-0.6B",
            api_key=Secret.from_token("my-key"),
            api_base_url="http://my-server:8000/v1",
            generation_kwargs={"max_tokens": 512, "temperature": 0.7},
            timeout=60.0,
            max_retries=3,
        )

        assert component.client.api_key == "my-key"
        assert component.generation_kwargs == {"max_tokens": 512, "temperature": 0.7}
        assert component.client.timeout == 60.0
        assert component.client.max_retries == 3


class TestVLLMChatGeneratorSerde:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "test-key")
        component = VLLMChatGenerator(
            model="Qwen/Qwen3-0.6B",
            generation_kwargs={"max_tokens": 512},
        )
        data = component.to_dict()

        assert data["init_parameters"]["model"] == "Qwen/Qwen3-0.6B"
        assert data["init_parameters"]["api_base_url"] == "http://localhost:8000/v1"
        assert data["init_parameters"]["generation_kwargs"] == {"max_tokens": 512}
        assert "organization" not in data["init_parameters"]
        assert "tools_strict" not in data["init_parameters"]

    def test_from_dict(self):
        data = {
            "type": (
                "haystack_integrations.components.generators.vllm.chat.chat_generator.VLLMChatGenerator"
            ),
            "init_parameters": {
                "model": "Qwen/Qwen3-0.6B",
                "api_key": {"type": "env_var", "env_vars": ["VLLM_API_KEY"], "strict": False},
                "api_base_url": "http://my-server:8000/v1",
                "generation_kwargs": {"max_tokens": 512},
                "streaming_callback": None,
                "tools": None,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }
        component = VLLMChatGenerator.from_dict(data)

        assert isinstance(component, VLLMChatGenerator)
        assert component.model == "Qwen/Qwen3-0.6B"
        assert component.generation_kwargs == {"max_tokens": 512}


class TestVLLMChatGeneratorRun:
    def test_run(self, mock_chat_completion):
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")
        response = component.run([ChatMessage.from_user("What's the capital of France")])

        assert len(response["replies"]) == 1
        assert response["replies"][0].text == "Paris is the capital of France."
        assert response["replies"][0].reasoning is None

    def test_run_with_reasoning(self, mock_chat_completion_with_reasoning):
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")
        response = component.run([ChatMessage.from_user("What's the capital of France")])

        reply = response["replies"][0]
        assert reply.text == "Paris is the capital of France."
        assert reply.reasoning is not None
        assert "capital of France" in reply.reasoning.reasoning_text

    def test_run_passes_generation_kwargs(self, mock_chat_completion):
        component = VLLMChatGenerator(
            model="Qwen/Qwen3-0.6B",
            generation_kwargs={"max_tokens": 100, "temperature": 0.5},
        )
        component.run([ChatMessage.from_user("Hello")])

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.5

    def test_run_empty_messages(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")
        assert component.run([]) == {"replies": []}

    def test_run_streaming(self, mock_streaming_completion):
        chunks_received = []
        component = VLLMChatGenerator(
            model="Qwen/Qwen3-0.6B",
            streaming_callback=lambda c: chunks_received.append(c),
        )
        response = component.run([ChatMessage.from_user("Hello")])

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1

    def test_run_streaming_with_reasoning(self, mock_streaming_completion_with_reasoning):
        chunks_received = []
        component = VLLMChatGenerator(
            model="Qwen/Qwen3-0.6B",
            streaming_callback=lambda c: chunks_received.append(c),
        )
        response = component.run([ChatMessage.from_user("Hello")])

        reasoning_chunks = [c for c in chunks_received if c.reasoning]
        assert len(reasoning_chunks) > 0
        assert response["replies"][0].reasoning is not None


class TestVLLMChatGeneratorRunAsync:
    @pytest.mark.asyncio
    async def test_run_async(self, mock_async_chat_completion):
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")
        response = await component.run_async([ChatMessage.from_user("Hello")])

        assert len(response["replies"]) == 1
        assert response["replies"][0].text == "Paris is the capital of France."
        assert response["replies"][0].reasoning is None

    @pytest.mark.asyncio
    async def test_run_async_with_reasoning(self, mock_async_chat_completion_with_reasoning):
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")
        response = await component.run_async([ChatMessage.from_user("Hello")])

        reply = response["replies"][0]
        assert reply.reasoning is not None
        assert "capital of France" in reply.reasoning.reasoning_text

    @pytest.mark.asyncio
    async def test_run_async_streaming(self, mock_async_streaming_completion):
        chunks_received = []

        async def callback(chunk):
            chunks_received.append(chunk)

        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B", streaming_callback=callback)
        response = await component.run_async([ChatMessage.from_user("Hello")])

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1

    @pytest.mark.asyncio
    async def test_run_async_streaming_with_reasoning(self, mock_async_streaming_completion_with_reasoning):
        chunks_received = []

        async def callback(chunk):
            chunks_received.append(chunk)

        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B", streaming_callback=callback)
        response = await component.run_async([ChatMessage.from_user("Hello")])

        reasoning_chunks = [c for c in chunks_received if c.reasoning]
        assert len(reasoning_chunks) > 0
        assert response["replies"][0].reasoning is not None


NO_THINKING_KWARGS = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


@pytest.mark.integration
class TestVLLMChatGeneratorLiveRun:
    def test_live_run(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B", generation_kwargs=NO_THINKING_KWARGS)
        response = component.run(
            [ChatMessage.from_user("What is the capital of France? Answer in one word.")]
        )

        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert "paris" in reply.text.lower()
        assert reply.reasoning is None

    def test_live_run_streaming(self):
        chunks_received = []
        component = VLLMChatGenerator(
            model="Qwen/Qwen3-0.6B",
            generation_kwargs=NO_THINKING_KWARGS,
            streaming_callback=lambda c: chunks_received.append(c),
        )
        response = component.run(
            [ChatMessage.from_user("What is the capital of France? Answer in one word.")]
        )

        assert len(chunks_received) > 0
        assert len(response["replies"]) == 1
        assert "paris" in response["replies"][0].text.lower()

    def test_live_run_with_reasoning(self):
        component = VLLMChatGenerator(
            model="Qwen/Qwen3-0.6B",
            generation_kwargs={"max_tokens": 500},
        )
        response = component.run(
            [ChatMessage.from_user("What is 2+2? Answer briefly.")]
        )

        reply = response["replies"][0]
        assert reply.reasoning is not None
        assert len(reply.reasoning.reasoning_text) > 0
        assert reply.text is not None

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        component = VLLMChatGenerator(model="Qwen/Qwen3-0.6B", generation_kwargs=NO_THINKING_KWARGS)
        response = await component.run_async(
            [ChatMessage.from_user("What is the capital of France? Answer in one word.")]
        )

        assert len(response["replies"]) == 1
        assert "paris" in response["replies"][0].text.lower()
