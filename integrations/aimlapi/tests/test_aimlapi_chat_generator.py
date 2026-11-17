# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, ToolCall
from haystack.tools import Tool
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.aimlapi.chat.chat_generator import AIMLAPIChatGenerator


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=weather,
    )

    return [tool]


class TestAIMLAPIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AIMLAPI_API_KEY", "test-api-key")
        component = AIMLAPIChatGenerator()
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "openai/gpt-5-chat-latest"
        assert component.api_base_url == "https://api.aimlapi.com/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_with_parameters(self):
        component = AIMLAPIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="openai/gpt-5-chat-latest",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "openai/gpt-5-chat-latest"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("AIMLAPI_API_KEY", "test-api-key")
        component = AIMLAPIChatGenerator()
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.aimlapi.chat.chat_generator.AIMLAPIChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["AIMLAPI_API_KEY"], "strict": True, "type": "env_var"},
            "model": "openai/gpt-5-chat-latest",
            "streaming_callback": None,
            "api_base_url": "https://api.aimlapi.com/v1",
            "generation_kwargs": {},
            "extra_headers": None,
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AIMLAPIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="openai/gpt-5-chat-latest",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            extra_headers={"test-header": "test-value"},
            timeout=10,
            max_retries=10,
            tools=None,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.aimlapi.chat.chat_generator.AIMLAPIChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "openai/gpt-5-chat-latest",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            "extra_headers": {"test-header": "test-value"},
            "timeout": 10,
            "max_retries": 10,
            "tools": None,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("AIMLAPI_API_KEY", "fake-api-key")
        data = {
            "type": ("haystack_integrations.components.generators.aimlapi.chat.chat_generator.AIMLAPIChatGenerator"),
            "init_parameters": {
                "api_key": {"env_vars": ["AIMLAPI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "openai/gpt-5-chat-latest",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "extra_headers": {"test-header": "test-value"},
                "timeout": 10,
                "max_retries": 10,
                "tools": None,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }
        component = AIMLAPIChatGenerator.from_dict(data)
        assert component.model == "openai/gpt-5-chat-latest"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("AIMLAPI_API_KEY")
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.extra_headers == {"test-header": "test-value"}
        assert component.timeout == 10
        assert component.max_retries == 10

    @pytest.mark.parametrize("extra_headers", [None, {"X-Title": "Custom", "X-Custom": "value"}])
    def test_prepare_api_call_with_params(self, extra_headers):
        component = AIMLAPIChatGenerator(
            generation_kwargs={"max_tokens": 10, "temperature": 0.5}, extra_headers=extra_headers
        )
        args = component._prepare_api_call(
            messages=[ChatMessage.from_user("hi")], generation_kwargs={"temperature": 0.7}
        )

        assert args["extra_body"] == {"max_tokens": 10, "temperature": 0.7}
        assert args["extra_headers"] == {
            "HTTP-Referer": "https://github.com/deepset-ai/haystack-core-integrations",
            "X-Title": "Haystack",
            **(extra_headers or {}),
        }
        assert component.generation_kwargs == {"max_tokens": 10, "temperature": 0.5}

    @pytest.mark.skipif(
        not os.environ.get("AIMLAPI_API_KEY", None),
        reason="Export an env var called AIMLAPI_API_KEY containing the AIMLAPI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = AIMLAPIChatGenerator(model="openai/gpt-5-nano-2025-08-07")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text
        assert "Paris" in message.text
        assert "gpt-5-nano" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("AIMLAPI_API_KEY", None),
        reason="Export an env var called AIMLAPI_API_KEY containing the AIMLAPI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = AIMLAPIChatGenerator(streaming_callback=callback, model="openai/gpt-5-nano-2025-08-07")
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text
        assert "Paris" in message.text

        assert "gpt-5-nano" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("AIMLAPI_API_KEY", None),
        reason="Export an env var called AIMLAPI_API_KEY containing the AIMLAPI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        """
        Integration test that the AIMLAPIChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = AIMLAPIChatGenerator(tools=tools, model="openai/gpt-5-nano-2025-08-07")
        results = component.run(messages=initial_messages, generation_kwargs={"tool_choice": "auto"})

        assert len(results["replies"]) == 1

        # Find the message with tool calls
        tool_message = results["replies"][0]

        assert isinstance(tool_message, ChatMessage)
        tool_calls = tool_message.tool_calls
        assert len(tool_calls) == 2
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)

        for tool_call in tool_calls:
            assert tool_call.id is not None
            assert isinstance(tool_call, ToolCall)
            assert tool_call.tool_name == "weather"

        arguments = [tool_call.arguments for tool_call in tool_calls]
        assert sorted(arguments, key=lambda x: x["city"]) == [{"city": "Berlin"}, {"city": "Paris"}]
        assert tool_message.meta["finish_reason"] == "tool_calls"

        new_messages = [
            initial_messages[0],
            tool_message,
            ChatMessage.from_tool(tool_result="22° C and sunny", origin=tool_calls[0]),
            ChatMessage.from_tool(tool_result="16° C and windy", origin=tool_calls[1]),
        ]
        # Pass the tool result to the model to get the final response
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert final_message.is_from(ChatRole.ASSISTANT)
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.skipif(
        not os.environ.get("AIMLAPI_API_KEY", None),
        reason="Export an env var called AIMLAPI_API_KEY containing the AIMLAPI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the AIMLAPIChatGenerator component can run with tools and streaming.
        """
        component = AIMLAPIChatGenerator(
            tools=tools, streaming_callback=print_streaming_chunk, model="openai/gpt-5-nano-2025-08-07"
        )
        results = component.run(
            [ChatMessage.from_user("What's the weather like in Paris and Berlin?")],
            generation_kwargs={"tool_choice": "auto"},
        )

        assert len(results["replies"]) == 1

        # Find the message with tool calls
        tool_message = results["replies"][0]

        assert isinstance(tool_message, ChatMessage)
        tool_calls = tool_message.tool_calls
        assert len(tool_calls) == 2
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)

        for tool_call in tool_calls:
            assert tool_call.id is not None
            assert isinstance(tool_call, ToolCall)
            assert tool_call.tool_name == "weather"

        arguments = [tool_call.arguments for tool_call in tool_calls]
        assert sorted(arguments, key=lambda x: x["city"]) == [{"city": "Berlin"}, {"city": "Paris"}]
        assert tool_message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("AIMLAPI_API_KEY", None),
        reason="Export an env var called AIMLAPI_API_KEY containing the AIMLAPI API key to run this test.",
    )
    @pytest.mark.integration
    def test_agent_with_aimlapi_chat_generator(self, tools):
        """
        Test that the AIMLAPIChatGenerator component can drive tool calls in an Agent
        """
        agent = Agent(chat_generator=AIMLAPIChatGenerator(model="openai/gpt-5-mini-2025-08-07"), tools=tools)
        result = agent.run(messages=[ChatMessage.from_user("What's the weather like in Paris?")])

        tool_results = [msg.tool_call_result.result for msg in result["messages"] if msg.tool_call_result]
        assert "The weather in Paris is sunny and 32°C" in tool_results
