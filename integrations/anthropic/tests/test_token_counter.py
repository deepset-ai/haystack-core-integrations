# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import create_tool_from_function

from haystack_integrations.token_counters.anthropic import AnthropicTokenCounter


def weather_tool(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@pytest.fixture()
def counter(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    c = AnthropicTokenCounter(model="claude-sonnet-4-5")
    c.warm_up()
    return c


class TestAnthropicTokenCounterInit:
    def test_default_init(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5")
        assert c.model == "claude-sonnet-4-5"
        assert c.timeout is None
        assert c.max_retries is None
        assert c.client is None  # client not initialized until warm_up

    def test_custom_init(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-opus-4-5", timeout=30.0, max_retries=3)
        assert c.model == "claude-opus-4-5"
        assert c.timeout == 30.0
        assert c.max_retries == 3

    def test_warm_up_initializes_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5")
        assert c.client is None
        c.warm_up()
        assert c.client is not None

    def test_warm_up_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5")
        c.warm_up()
        first_client = c.client
        c.warm_up()
        assert c.client is first_client  # same instance, not re-created

    def test_close_releases_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5")
        c.warm_up()
        assert c.client is not None
        c.close()
        assert c.client is None

    def test_close_before_warm_up_is_safe(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5")
        c.close()  # should not raise
        assert c.client is None


class TestAnthropicTokenCounterSerde:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5", timeout=10.0, max_retries=2)
        d = c.to_dict()
        assert d["type"] == "haystack_integrations.token_counters.anthropic.token_counter.AnthropicTokenCounter"
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


class TestAnthropicTokenCounterCount:
    def test_count_empty_returns_zero(self, counter):
        assert counter.count([]) == 0
        assert counter.count([], tools=None) == 0

    def test_count_basic_messages(self, counter):
        mock_response = MagicMock()
        mock_response.input_tokens = 42

        with patch.object(counter.client.messages, "count_tokens", return_value=mock_response) as mock_call:
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

        with patch.object(counter.client.messages, "count_tokens", return_value=mock_response) as mock_call:
            messages = [ChatMessage.from_user("How many tokens?")]
            result = counter.count(messages)

        assert result == 10
        call_kwargs = mock_call.call_args.kwargs
        assert "system" not in call_kwargs  # no system message → no system param

    def test_count_with_tools(self, counter):
        mock_response = MagicMock()
        mock_response.input_tokens = 150

        tool = create_tool_from_function(weather_tool)

        with patch.object(counter.client.messages, "count_tokens", return_value=mock_response) as mock_call:
            messages = [ChatMessage.from_user("What's the weather in Paris?")]
            result = counter.count(messages, tools=[tool])

        assert result == 150
        call_kwargs = mock_call.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "weather_tool"

    def test_count_auto_warms_up(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        c = AnthropicTokenCounter(model="claude-sonnet-4-5")
        assert c.client is None

        mock_response = MagicMock()
        mock_response.input_tokens = 5

        with patch("haystack_integrations.token_counters.anthropic.token_counter.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.count_tokens.return_value = mock_response
            result = c.count([ChatMessage.from_user("hi")])

        assert c.client is not None
        assert result == 5


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
@pytest.mark.integration
class TestAnthropicTokenCounterIntegration:
    def test_count_complex(self):
        """Count tokens for a conversation with system, user, assistant, tool result messages and tools."""
        tool = create_tool_from_function(weather_tool)
        tool_call = ToolCall(tool_name="weather_tool", arguments={"city": "Paris"}, id="toolu_01")
        messages = [
            ChatMessage.from_system("You are a helpful weather assistant."),
            ChatMessage.from_user("What's the weather in Paris?"),
            ChatMessage.from_assistant("Let me check the weather for you.", tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="Sunny in Paris", origin=tool_call),
            ChatMessage.from_assistant("The weather in Paris is sunny."),
        ]
        counter = AnthropicTokenCounter(model="claude-haiku-4-5-20251001")
        count = counter.count(messages, tools=[tool])
        assert isinstance(count, int)
        assert count > 20
