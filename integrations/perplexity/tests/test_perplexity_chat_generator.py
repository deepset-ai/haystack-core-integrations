# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

import httpx
import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.perplexity.chat import (
    chat_generator as chat_generator_module,
)
from haystack_integrations.components.generators.perplexity.chat.chat_generator import (
    PerplexityChatGenerator,
)


def _make_transport(captured: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": int(datetime.now(tz=UTC).timestamp()),
                "model": "sonar-pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello world!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def chat_messages() -> list[ChatMessage]:
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France?"),
    ]


def weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is sunny."


@pytest.fixture
def tools() -> list[Tool]:
    tool_parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    return [
        Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters=tool_parameters,
            function=weather,
        )
    ]


class TestPerplexityChatGenerator:
    def test_attribution_header_falls_back_when_package_is_not_installed(self, monkeypatch):
        def raise_package_not_found(_package_name: str) -> str:
            raise chat_generator_module.importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(
            chat_generator_module.importlib.metadata,
            "version",
            raise_package_not_found,
        )

        assert chat_generator_module._attribution_header() == "haystack/unknown"

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-api-key")

        component = PerplexityChatGenerator()

        assert component.client.api_key == "test-api-key"
        assert component.model == "sonar-pro"
        assert component.api_base_url == "https://api.perplexity.ai"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_with_parameters(self):
        component = PerplexityChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="custom-model",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "temperature": 0.2},
            extra_headers={"test-header": "test-value"},
        )

        assert component.client.api_key == "test-api-key"
        assert component.model == "custom-model"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "temperature": 0.2}
        assert component.extra_headers == {"test-header": "test-value"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-api-key")

        component = PerplexityChatGenerator()
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.perplexity.chat.chat_generator.PerplexityChatGenerator"
        )
        assert data["init_parameters"]["api_key"] == Secret.from_env_var("PERPLEXITY_API_KEY").to_dict()
        assert data["init_parameters"] == {
            "api_key": Secret.from_env_var("PERPLEXITY_API_KEY").to_dict(),
            "model": "sonar-pro",
            "streaming_callback": None,
            "api_base_url": "https://api.perplexity.ai",
            "generation_kwargs": {},
            "tools": None,
            "extra_headers": None,
            "timeout": None,
            "max_retries": None,
            "http_client_kwargs": None,
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = PerplexityChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="custom-model",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "temperature": 0.2},
            extra_headers={"test-header": "test-value"},
            timeout=10,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )

        data = component.to_dict()

        assert data["init_parameters"] == {
            "api_key": Secret.from_env_var("ENV_VAR").to_dict(),
            "model": "custom-model",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "api_base_url": "test-base-url",
            "generation_kwargs": {"max_tokens": 10, "temperature": 0.2},
            "tools": None,
            "extra_headers": {"test-header": "test-value"},
            "timeout": 10,
            "max_retries": 2,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "fake-api-key")
        data = {
            "type": (
                "haystack_integrations.components.generators.perplexity.chat.chat_generator.PerplexityChatGenerator"
            ),
            "init_parameters": {
                "api_key": Secret.from_env_var("PERPLEXITY_API_KEY").to_dict(),
                "model": "sonar-pro",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "temperature": 0.2},
                "tools": None,
                "extra_headers": {"test-header": "test-value"},
                "timeout": 10,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }

        component = PerplexityChatGenerator.from_dict(data)

        assert component.model == "sonar-pro"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "temperature": 0.2}
        assert component.api_key == Secret.from_env_var("PERPLEXITY_API_KEY")
        assert component.extra_headers == {"test-header": "test-value"}
        assert component.timeout == 10
        assert component.max_retries == 2
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}

    def test_run_sends_attribution_header(self, chat_messages):
        captured: list[httpx.Request] = []
        component = PerplexityChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            http_client_kwargs={"transport": _make_transport(captured)},
        )

        result = component.run(chat_messages)

        assert result["replies"][0].text == "Hello world!"
        assert len(captured) == 1
        request = captured[0]
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert request.headers["X-Pplx-Integration"].startswith("haystack/")

    def test_prepare_api_call_merges_extra_headers(self, chat_messages):
        component = PerplexityChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            extra_headers={"test-header": "test-value"},
        )

        api_args = component._prepare_api_call(messages=chat_messages)

        assert api_args["extra_headers"]["test-header"] == "test-value"
        assert api_args["extra_headers"]["X-Pplx-Integration"].startswith("haystack/")

    def test_prepare_api_call_raises_when_streaming_with_multiple_responses(self, chat_messages):
        component = PerplexityChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"n": 2},
        )

        with pytest.raises(ValueError, match="Cannot stream multiple responses"):
            component._prepare_api_call(messages=chat_messages, streaming_callback=print_streaming_chunk)

    def test_prepare_api_call_with_tools_strict(self, chat_messages, tools):
        component = PerplexityChatGenerator(api_key=Secret.from_token("test-api-key"))

        api_args = component._prepare_api_call(messages=chat_messages, tools=tools, tools_strict=True)

        assert api_args["tools"][0]["type"] == "function"
        function_spec = api_args["tools"][0]["function"]
        assert function_spec["name"] == "weather"
        assert function_spec["strict"] is True
        assert function_spec["parameters"]["additionalProperties"] is False

    def test_prepare_api_call_with_response_format(self, chat_messages):
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "Foo", "schema": {"type": "object"}},
        }
        component = PerplexityChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"response_format": response_format},
        )

        api_args = component._prepare_api_call(messages=chat_messages)

        assert api_args["openai_endpoint"] == "parse"
        assert api_args["response_format"] == response_format

    def test_prepare_api_call_with_response_format_and_streaming(self, chat_messages):
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "Foo", "schema": {"type": "object"}},
        }
        component = PerplexityChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"response_format": response_format},
        )

        api_args = component._prepare_api_call(messages=chat_messages, streaming_callback=print_streaming_chunk)

        assert api_args["openai_endpoint"] == "create"
        assert api_args["stream"] is True
        assert api_args["response_format"] == response_format
