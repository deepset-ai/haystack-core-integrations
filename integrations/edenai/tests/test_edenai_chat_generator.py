# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import Tool
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.edenai.chat.chat_generator import EdenAIChatGenerator

API_BASE_URL = "https://api.edenai.run/v3"
DEFAULT_MODEL = "openai/gpt-4o-mini"


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


class TestEdenAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "test-api-key")
        component = EdenAIChatGenerator()
        assert component.api_key == Secret.from_env_var("EDENAI_API_KEY")
        assert component.model == DEFAULT_MODEL
        assert component.api_base_url == API_BASE_URL
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_with_parameters(self):
        component = EdenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="mistral/mistral-large-latest",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "mistral/mistral-large-latest"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "test-api-key")
        component = EdenAIChatGenerator()
        data = component.to_dict()

        assert (
            data["type"] == "haystack_integrations.components.generators.edenai.chat.chat_generator.EdenAIChatGenerator"
        )

        # `api_base_url` is fixed to the Eden AI endpoint and `organization` is unused; neither is
        # accepted by EdenAIChatGenerator.__init__, so to_dict drops them to stay round-trippable.
        assert "api_base_url" not in data["init_parameters"]
        assert "organization" not in data["init_parameters"]

        expected_params = {
            "api_key": {"env_vars": ["EDENAI_API_KEY"], "strict": True, "type": "env_var"},
            "model": DEFAULT_MODEL,
            "streaming_callback": None,
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = EdenAIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="mistral/mistral-large-latest",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=10,
            max_retries=10,
            tools=None,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"] == "haystack_integrations.components.generators.edenai.chat.chat_generator.EdenAIChatGenerator"
        )

        assert "api_base_url" not in data["init_parameters"]

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "mistral/mistral-large-latest",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            "timeout": 10,
            "max_retries": 10,
            "tools": None,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "fake-api-key")
        data = {
            "type": ("haystack_integrations.components.generators.edenai.chat.chat_generator.EdenAIChatGenerator"),
            "init_parameters": {
                "api_key": {"env_vars": ["EDENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral/mistral-large-latest",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10,
                "max_retries": 10,
                "tools": None,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }
        component = EdenAIChatGenerator.from_dict(data)
        assert component.model == "mistral/mistral-large-latest"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == API_BASE_URL
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("EDENAI_API_KEY")
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.timeout == 10
        assert component.max_retries == 10

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY", None),
        reason="Export an env var called EDENAI_API_KEY containing the Eden AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = EdenAIChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY", None),
        reason="Export an env var called EDENAI_API_KEY containing the Eden AI API key to run this test.",
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
        component = EdenAIChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY", None),
        reason="Export an env var called EDENAI_API_KEY containing the Eden AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = EdenAIChatGenerator(tools=tools)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.text is None

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
