"""Tests for the LiteLLM Chat Generator integration."""

import json
import sys
import types
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import Tool
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.litellm import LiteLLMChatGenerator


def _make_mock_response(content="Hello!", model="openai/gpt-4o", tool_calls=None):
    """Build a mock litellm ModelResponse."""
    msg = MagicMock()
    msg.content = content
    msg.role = "assistant"
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg
    choice.index = 0
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = model
    return resp


def _make_mock_stream_chunks(content="Hello!"):
    """Build mock streaming chunks."""
    chunks = []
    for char in content:
        delta = MagicMock()
        delta.content = char
        delta.tool_calls = None

        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = None

        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.model = "openai/gpt-4o"
        chunks.append(chunk)

    # Final chunk with finish_reason
    delta = MagicMock()
    delta.content = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = "stop"
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.model = "openai/gpt-4o"
    chunks.append(chunk)
    return chunks


class TestLiteLLMChatGeneratorInit:
    def test_default_init(self):
        gen = LiteLLMChatGenerator()
        assert gen.model == "openai/gpt-4o"
        assert gen.api_key is None
        assert gen.streaming_callback is None
        assert gen.api_base_url is None
        assert gen.generation_kwargs == {}

    def test_init_with_params(self):
        gen = LiteLLMChatGenerator(
            model="anthropic/claude-sonnet-4-20250514",
            api_key=Secret.from_token("sk-test"),
            api_base_url="https://proxy.local",
            generation_kwargs={"temperature": 0.5, "max_tokens": 100},
        )
        assert gen.model == "anthropic/claude-sonnet-4-20250514"
        assert gen.api_key.resolve_value() == "sk-test"
        assert gen.api_base_url == "https://proxy.local"
        assert gen.generation_kwargs["temperature"] == 0.5


class TestSerialization:
    def test_to_dict(self):
        gen = LiteLLMChatGenerator(
            model="openai/gpt-4o",
            generation_kwargs={"temperature": 0.7},
        )
        d = gen.to_dict()
        assert d["init_parameters"]["model"] == "openai/gpt-4o"
        assert d["init_parameters"]["generation_kwargs"]["temperature"] == 0.7

    def test_from_dict(self):
        d = {
            "type": "haystack_integrations.components.generators.litellm.chat.chat_generator.LiteLLMChatGenerator",
            "init_parameters": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "api_key": None,
                "streaming_callback": None,
                "api_base_url": None,
                "generation_kwargs": {"max_tokens": 200},
                "tools": None,
            },
        }
        gen = LiteLLMChatGenerator.from_dict(d)
        assert gen.model == "anthropic/claude-sonnet-4-20250514"
        assert gen.generation_kwargs["max_tokens"] == 200


@pytest.mark.unit
class TestRun:
    def test_basic_completion(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        mock_resp = _make_mock_response("The capital is Paris.")

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            messages = [ChatMessage.from_user("What's the capital of France?")]
            result = gen.run(messages=messages)

            assert len(result["replies"]) == 1
            assert result["replies"][0].text == "The capital is Paris."
            fake_litellm.completion.assert_called_once()

    def test_drop_params_always_set(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        mock_resp = _make_mock_response()

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            gen.run(messages=[ChatMessage.from_user("hi")])
            call_kwargs = fake_litellm.completion.call_args
            assert call_kwargs.kwargs["drop_params"] is True

    def test_api_key_forwarded(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o", api_key=Secret.from_token("sk-test123"))
        mock_resp = _make_mock_response()

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            gen.run(messages=[ChatMessage.from_user("hi")])
            call_kwargs = fake_litellm.completion.call_args
            assert call_kwargs.kwargs["api_key"] == "sk-test123"

    def test_api_key_omitted_when_none(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        mock_resp = _make_mock_response()

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            gen.run(messages=[ChatMessage.from_user("hi")])
            call_kwargs = fake_litellm.completion.call_args
            assert "api_key" not in call_kwargs.kwargs

    def test_base_url_forwarded(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o", api_base_url="https://proxy.local")
        mock_resp = _make_mock_response()

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            gen.run(messages=[ChatMessage.from_user("hi")])
            call_kwargs = fake_litellm.completion.call_args
            assert call_kwargs.kwargs["api_base"] == "https://proxy.local"

    def test_generation_kwargs_merged(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o", generation_kwargs={"temperature": 0.5})
        mock_resp = _make_mock_response()

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            gen.run(messages=[ChatMessage.from_user("hi")], generation_kwargs={"max_tokens": 100})
            call_kwargs = fake_litellm.completion.call_args
            assert call_kwargs.kwargs["temperature"] == 0.5
            assert call_kwargs.kwargs["max_tokens"] == 100

    def test_empty_messages_returns_empty(self):
        gen = LiteLLMChatGenerator()
        result = gen.run(messages=[])
        assert result == {"replies": []}

    def test_usage_in_meta(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        mock_resp = _make_mock_response()

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = gen.run(messages=[ChatMessage.from_user("hi")])
            meta = result["replies"][0].meta
            assert meta["usage"]["prompt_tokens"] == 10
            assert meta["usage"]["completion_tokens"] == 5

    def test_tool_calls_parsed(self):
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "Paris"}'

        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        mock_resp = _make_mock_response(content="", tool_calls=[tc])

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = gen.run(messages=[ChatMessage.from_user("weather?")])
            reply = result["replies"][0]
            assert reply.tool_calls is not None
            assert len(reply.tool_calls) == 1
            assert reply.tool_calls[0].tool_name == "get_weather"
            assert reply.tool_calls[0].arguments == {"city": "Paris"}

    def test_streaming_content_accumulated(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        chunks = _make_mock_stream_chunks("Hi!")

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=iter(chunks))

        collected = []

        def callback(chunk: StreamingChunk) -> None:
            collected.append(chunk)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = gen.run(messages=[ChatMessage.from_user("hi")], streaming_callback=callback)
            assert len(collected) > 0
            assert len(result["replies"]) == 1
            full_text = "".join(c.content for c in collected)
            assert "Hi!" in full_text

    def test_generation_kwargs_runtime_overrides_init(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o", generation_kwargs={"temperature": 0.5})
        mock_resp = _make_mock_response()
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            gen.run(messages=[ChatMessage.from_user("hi")], generation_kwargs={"temperature": 0.9})
            call_kwargs = fake_litellm.completion.call_args
            assert call_kwargs.kwargs["temperature"] == 0.9

    def test_none_content_in_response(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        mock_resp = _make_mock_response(content=None)
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = gen.run(messages=[ChatMessage.from_user("hi")])
            assert result["replies"][0].text == ""

    def test_tools_sent_to_litellm(self):
        def weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        tool = Tool(
            function=weather,
            name="weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        )
        gen = LiteLLMChatGenerator(model="openai/gpt-4o", tools=[tool])
        mock_resp = _make_mock_response()

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            gen.run(messages=[ChatMessage.from_user("weather in Paris?")])
            call_kwargs = fake_litellm.completion.call_args
            assert call_kwargs.kwargs["tools"] is not None
            assert len(call_kwargs.kwargs["tools"]) == 1
            assert call_kwargs.kwargs["tools"][0]["function"]["name"] == "weather"


class TestSerializationRoundTrip:
    def test_round_trip_basic(self):
        gen = LiteLLMChatGenerator(
            model="anthropic/claude-sonnet-4-20250514",
            api_base_url="https://proxy.local",
            generation_kwargs={"temperature": 0.8, "max_tokens": 200},
        )
        restored = LiteLLMChatGenerator.from_dict(gen.to_dict())
        assert restored.model == gen.model
        assert restored.api_base_url == gen.api_base_url
        assert restored.generation_kwargs == gen.generation_kwargs

    def test_round_trip_with_secret(self):
        gen = LiteLLMChatGenerator(
            model="openai/gpt-4o",
            api_key=Secret.from_env_var("MY_TEST_KEY"),
        )
        d = gen.to_dict()
        assert d["init_parameters"]["api_key"]["type"] == "env_var"
        restored = LiteLLMChatGenerator.from_dict(d)
        assert restored.api_key is not None


@pytest.mark.unit
class TestAsync:
    @pytest.mark.asyncio
    async def test_run_async_basic(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        mock_resp = _make_mock_response("async reply")

        fake_litellm = types.ModuleType("litellm")
        fake_litellm.acompletion = MagicMock(return_value=mock_resp)

        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            # acompletion needs to be an awaitable
            import asyncio
            future = asyncio.Future()
            future.set_result(mock_resp)
            fake_litellm.acompletion = MagicMock(return_value=future)

            result = await gen.run_async(messages=[ChatMessage.from_user("hi")])
            assert len(result["replies"]) == 1
            assert result["replies"][0].text == "async reply"

    @pytest.mark.asyncio
    async def test_run_async_empty_messages(self):
        gen = LiteLLMChatGenerator(model="openai/gpt-4o")
        result = await gen.run_async(messages=[])
        assert result == {"replies": []}
