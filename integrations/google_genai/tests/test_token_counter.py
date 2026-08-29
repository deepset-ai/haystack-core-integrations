# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from haystack.utils.auth import Secret

from haystack_integrations.token_counters.google_genai import GoogleGenAITokenCounter


def weather(city: str):
    """Get weather information for a city."""
    return f"Weather in {city}: 22°C, sunny"


@pytest.fixture
def tools():
    return [
        Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=weather,
        )
    ]


def _counter_with_mock_client(total_tokens, **kwargs):
    """Build a counter whose client is already set, so `warm_up` never builds a real one."""
    counter = GoogleGenAITokenCounter("gemini-3.7-flash", **kwargs)
    client = Mock()
    client.models.count_tokens.return_value = Mock(total_tokens=total_tokens)
    counter.client = client
    return counter


class TestGoogleGenAITokenCounterInitSerDe:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        counter = GoogleGenAITokenCounter("gemini-3.7-flash")

        assert counter.model == "gemini-3.7-flash"
        assert counter.api == "gemini"
        assert counter.vertex_ai_project is None
        assert counter.vertex_ai_location is None
        assert counter.timeout is None
        assert counter.max_retries is None
        # The client is built lazily, so constructing the counter never touches the network.
        assert counter.client is None

    def test_serde_round_trip(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        counter = GoogleGenAITokenCounter(
            "gemini-3.7-flash",
            api="vertex",
            vertex_ai_project="my-project",
            vertex_ai_location="us-central1",
            timeout=10.0,
            max_retries=2,
        )

        data = counter.to_dict()
        assert data == {
            "type": "haystack_integrations.token_counters.google_genai.token_counter.GoogleGenAITokenCounter",
            "init_parameters": {
                "model": "gemini-3.7-flash",
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "strict": False},
                "api": "vertex",
                "vertex_ai_project": "my-project",
                "vertex_ai_location": "us-central1",
                "timeout": 10.0,
                "max_retries": 2,
            },
        }

        restored = GoogleGenAITokenCounter.from_dict(data)
        assert restored.model == counter.model
        assert restored.api == counter.api
        assert restored.vertex_ai_project == counter.vertex_ai_project
        assert restored.vertex_ai_location == counter.vertex_ai_location
        assert restored.timeout == counter.timeout
        assert restored.max_retries == counter.max_retries
        assert restored.api_key == Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False)


class TestGoogleGenAITokenCounterCount:
    def test_count_without_messages_or_tools_returns_zero(self):
        counter = GoogleGenAITokenCounter("gemini-3.7-flash")

        assert counter.count([]) == 0
        # Nothing to measure means nothing to connect to either.
        assert counter.client is None

    def test_count_sends_messages_and_returns_total(self):
        counter = _counter_with_mock_client(42)

        assert counter.count([ChatMessage.from_user("How many tokens is this?")]) == 42

        _, kwargs = counter.client.models.count_tokens.call_args
        assert kwargs["model"] == "gemini-3.7-flash"
        assert len(kwargs["contents"]) == 1
        assert kwargs["config"] is None

    def test_count_returns_zero_when_the_api_reports_no_total(self):
        counter = _counter_with_mock_client(None)

        assert counter.count([ChatMessage.from_user("Hello")]) == 0

    @pytest.mark.parametrize(
        ("messages", "expected_in_error"),
        [
            ([ChatMessage.from_system("You are helpful."), ChatMessage.from_user("Hi")], "a system message"),
            ([ChatMessage.from_user("Hi")], "tools"),
        ],
        ids=["system_message", "tools"],
    )
    def test_count_rejects_what_the_gemini_developer_api_cannot_measure(self, messages, expected_in_error, tools):
        counter = GoogleGenAITokenCounter("gemini-3.7-flash")
        passed_tools = tools if expected_in_error == "tools" else None

        with pytest.raises(ValueError, match=expected_in_error) as exc_info:
            counter.count(messages, tools=passed_tools)

        # The error has to point at the way out, not just at the failure.
        assert 'api="vertex"' in str(exc_info.value)
        # Rejected before any client is built, so it fails the same way without credentials.
        assert counter.client is None

    def test_count_on_vertex_sends_the_system_instruction_and_tools(self, tools):
        counter = _counter_with_mock_client(99, api="vertex", vertex_ai_project="p", vertex_ai_location="l")
        messages = [
            ChatMessage.from_system("You are helpful."),
            ChatMessage.from_user("What is the weather in Paris?"),
        ]

        assert counter.count(messages, tools=tools) == 99

        _, kwargs = counter.client.models.count_tokens.call_args
        # The system message is measured as the system instruction, not as one of the contents.
        assert len(kwargs["contents"]) == 1
        assert kwargs["config"].system_instruction == "You are helpful."
        assert len(kwargs["config"].tools) == 1

    def test_close_closes_the_client(self):
        counter = _counter_with_mock_client(1)
        client = counter.client

        counter.close()

        # Dropping the reference is not enough: the client holds a pooled HTTP connection.
        client.close.assert_called_once()
        assert counter.client is None


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY", None) and not os.environ.get("GEMINI_API_KEY", None),
    reason="Export an env var called GOOGLE_API_KEY or GEMINI_API_KEY containing the Google API key to run this test.",
)
@pytest.mark.integration
class TestGoogleGenAITokenCounterInference:
    def test_live_count(self, tools):
        """
        Counts a conversation that carries a user turn, an assistant tool call and a tool result.

        A system message and tools are left out on purpose: the Google Gen AI SDK rejects both on the Gemini
        Developer API, which is the backend these tests authenticate against.
        """
        tool_call = ToolCall(tool_name="weather", arguments={"city": "Paris"})
        messages = [
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call),
        ]

        counter = GoogleGenAITokenCounter("gemini-3.7-flash")
        token_count = counter.count(messages)

        assert isinstance(token_count, int)
        assert token_count > 0
