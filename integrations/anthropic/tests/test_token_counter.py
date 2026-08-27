# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool, create_tool_from_function
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.anthropic import AnthropicTokenCounter


def weather_tool(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@pytest.fixture()
def counter(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    return AnthropicTokenCounter(model="claude-sonnet-4-5")


class TestAnthropicTokenCounterInit:
    def test_default_init(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter()
        assert c.model == "claude-sonnet-4-5"
        assert c.timeout is None
        assert c.max_retries is None

    def test_custom_init(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-opus-4-5", timeout=30.0, max_retries=3)
        assert c.model == "claude-opus-4-5"
        assert c.timeout == 30.0
        assert c.max_retries == 3

    def test_explicit_api_key(self):
        c = AnthropicTokenCounter(api_key=Secret.from_token("sk-test"))
        assert c.model == "claude-sonnet-4-5"


class TestAnthropicTokenCounterSerde:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5", timeout=10.0, max_retries=2)
        d = c.to_dict()
        assert d["type"] == "haystack_integrations.components.generators.anthropic.token_counter.AnthropicTokenCounter"
        assert d["init_parameters"]["model"] == "claude-sonnet-4-5"
        assert d["init_parameters"]["timeout"] == 10.0
        assert d["init_parameters"]["max_retries"] == 2
        assert "api_key" in d["init_parameters"]

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5")
        d = c.to_dict()
        restored = AnthropicTokenCounter.from_dict(d)
        assert restored.model == c.model
        assert restored.timeout == c.timeout
        assert restored.max_retries == c.max_retries

    def test_roundtrip(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-haiku-4-5-20251001", timeout=5.0, max_retries=1)
        restored = AnthropicTokenCounter.from_dict(c.to_dict())
        assert restored.model == c.model
        assert restored.timeout == c.timeout
        assert restored.max_retries == c.max_retries


class TestAnthropicTokenCounterCount:
    def test_count_basic_messages(self, counter):
        mock_response = MagicMock()
        mock_response.input_tokens = 42

        with patch.object(counter._client.messages, "count_tokens", return_value=mock_response) as mock_call:
            messages = [
                ChatMessage.from_system("You are helpful."),
                ChatMessage.from_user("Hello!"),
            ]
            result = counter.count(messages)

        assert result == 42
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-5"
        assert "system" in call_kwargs
        assert len(call_kwargs["messages"]) == 1  # only non-system messages

    def test_count_user_only_messages(self, counter):
        mock_response = MagicMock()
        mock_response.input_tokens = 10

        with patch.object(counter._client.messages, "count_tokens", return_value=mock_response) as mock_call:
            messages = [ChatMessage.from_user("How many tokens?")]
            result = counter.count(messages)

        assert result == 10
        call_kwargs = mock_call.call_args.kwargs
        assert "system" not in call_kwargs  # no system message → no system param

    def test_count_with_tools(self, counter, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        mock_response = MagicMock()
        mock_response.input_tokens = 150

        tool = create_tool_from_function(weather_tool)

        with patch.object(counter._client.messages, "count_tokens", return_value=mock_response) as mock_call:
            messages = [ChatMessage.from_user("What's the weather in Paris?")]
            result = counter.count(messages, tools=[tool])

        assert result == 150
        call_kwargs = mock_call.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "weather_tool"

    def test_count_no_tools(self, counter):
        mock_response = MagicMock()
        mock_response.input_tokens = 20

        with patch.object(counter._client.messages, "count_tokens", return_value=mock_response) as mock_call:
            messages = [ChatMessage.from_user("No tools here.")]
            result = counter.count(messages)

        assert result == 20
        call_kwargs = mock_call.call_args.kwargs
        assert "tools" not in call_kwargs

    def test_count_multi_turn(self, counter):
        mock_response = MagicMock()
        mock_response.input_tokens = 75

        with patch.object(counter._client.messages, "count_tokens", return_value=mock_response) as mock_call:
            messages = [
                ChatMessage.from_user("Tell me a joke."),
                ChatMessage.from_assistant("Why did the chicken cross the road?"),
                ChatMessage.from_user("Why?"),
            ]
            result = counter.count(messages)

        assert result == 75
        call_kwargs = mock_call.call_args.kwargs
        assert len(call_kwargs["messages"]) == 3


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
@pytest.mark.integration
class TestAnthropicTokenCounterIntegration:
    def test_count_basic(self):
        counter = AnthropicTokenCounter(model="claude-haiku-4-5-20251001")
        messages = [
            ChatMessage.from_system("You are a scientist."),
            ChatMessage.from_user("Hello, Claude"),
        ]
        count = counter.count(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_with_tools(self):
        counter = AnthropicTokenCounter(model="claude-haiku-4-5-20251001")
        tool = create_tool_from_function(weather_tool)
        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        count = counter.count(messages, tools=[tool])
        assert isinstance(count, int)
        assert count > 0

    def test_count_increases_with_tools(self):
        counter = AnthropicTokenCounter(model="claude-haiku-4-5-20251001")
        messages = [ChatMessage.from_user("What's the weather?")]
        tool = create_tool_from_function(weather_tool)
        count_without = counter.count(messages)
        count_with = counter.count(messages, tools=[tool])
        assert count_with > count_without
