# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret
from openai.types.responses import Response

from haystack_integrations.components.generators.perplexity.chat import (
    chat_generator as chat_generator_module,
)
from haystack_integrations.components.generators.perplexity.chat.chat_generator import (
    PerplexityChatGenerator,
)


def _make_response() -> Response:
    response_data = {
        "id": "resp_test",
        "created_at": datetime.now(tz=timezone.utc).timestamp(),
        "model": "openai/gpt-5.4",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }

    if hasattr(Response, "model_construct"):
        return Response.model_construct(**response_data)
    return Response.construct(**response_data)


@pytest.fixture
def chat_messages() -> list[ChatMessage]:
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France?"),
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
        assert component.model == "openai/gpt-5.4"
        assert component.api_base_url == "https://api.perplexity.ai/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_with_parameters(self):
        component = PerplexityChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="anthropic/claude-sonnet-4-6",
            streaming_callback=print_streaming_chunk,
            api_base_url="https://example.perplexity.test/v1",
            generation_kwargs={"max_output_tokens": 10, "temperature": 0.2},
            tools=[{"type": "web_search"}],
            tools_strict=True,
            extra_headers={"test-header": "test-value"},
            timeout=10,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )

        assert component.client.api_key == "test-api-key"
        assert component.model == "anthropic/claude-sonnet-4-6"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {
            "max_output_tokens": 10,
            "temperature": 0.2,
        }
        assert component.tools == [{"type": "web_search"}]
        assert component.tools_strict
        assert component.extra_headers == {"test-header": "test-value"}
        assert component.timeout == 10
        assert component.max_retries == 2
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}

    def test_to_dict_default_round_trip(self, monkeypatch):
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
            "model": "openai/gpt-5.4",
            "organization": None,
            "streaming_callback": None,
            "api_base_url": "https://api.perplexity.ai/v1",
            "generation_kwargs": {},
            "tools": None,
            "tools_strict": False,
            "extra_headers": None,
            "timeout": None,
            "max_retries": None,
            "http_client_kwargs": None,
        }

        deserialized = PerplexityChatGenerator.from_dict(data)
        assert deserialized.model == "openai/gpt-5.4"
        assert deserialized.api_base_url == "https://api.perplexity.ai/v1"
        assert deserialized.api_key == Secret.from_env_var("PERPLEXITY_API_KEY")
        assert deserialized.extra_headers is None

    def test_to_dict_with_parameters_round_trip(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = PerplexityChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="xai/grok-4-1",
            streaming_callback=print_streaming_chunk,
            api_base_url="https://example.perplexity.test/v1",
            generation_kwargs={"max_output_tokens": 10, "temperature": 0.2},
            tools=[{"type": "web_search"}],
            tools_strict=True,
            extra_headers={"test-header": "test-value"},
            timeout=10,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )

        data = component.to_dict()

        assert data["init_parameters"] == {
            "api_key": Secret.from_env_var("ENV_VAR").to_dict(),
            "model": "xai/grok-4-1",
            "organization": None,
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "api_base_url": "https://example.perplexity.test/v1",
            "generation_kwargs": {"max_output_tokens": 10, "temperature": 0.2},
            "tools": [{"type": "web_search"}],
            "tools_strict": True,
            "extra_headers": {"test-header": "test-value"},
            "timeout": 10,
            "max_retries": 2,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
        }

        deserialized = PerplexityChatGenerator.from_dict(data)
        assert deserialized.model == "xai/grok-4-1"
        assert deserialized.streaming_callback is print_streaming_chunk
        assert deserialized.api_base_url == "https://example.perplexity.test/v1"
        assert deserialized.generation_kwargs == {
            "max_output_tokens": 10,
            "temperature": 0.2,
        }
        assert deserialized.tools == [{"type": "web_search"}]
        assert deserialized.tools_strict
        assert deserialized.api_key == Secret.from_env_var("ENV_VAR")
        assert deserialized.extra_headers == {"test-header": "test-value"}
        assert deserialized.timeout == 10
        assert deserialized.max_retries == 2
        assert deserialized.http_client_kwargs == {"proxy": "http://localhost:8080"}

    def test_run_uses_responses_create(self, chat_messages):
        component = PerplexityChatGenerator(api_key=Secret.from_token("test-api-key"))

        with patch("openai.resources.responses.Responses.create", return_value=_make_response()) as mock_create:
            result = component.run(chat_messages)

        assert len(result["replies"]) == 1
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-5.4"
        assert "input" in call_kwargs
        assert "messages" not in call_kwargs
        assert call_kwargs["stream"] is False

    def test_default_headers_include_perplexity_attribution(self):
        component = PerplexityChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            extra_headers={"test-header": "test-value"},
        )

        assert component.client.default_headers["X-Pplx-Integration"].startswith("haystack/")
        assert component.client.default_headers["test-header"] == "test-value"
        assert component.async_client.default_headers["X-Pplx-Integration"].startswith("haystack/")
        assert component.async_client.default_headers["test-header"] == "test-value"


@pytest.mark.skipif(
    not os.environ.get("PERPLEXITY_API_KEY"),
    reason="Export PERPLEXITY_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestPerplexityChatGeneratorInference:
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France? Reply in one word.")]
        component = PerplexityChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France? Reply in one word.")]
        component = PerplexityChatGenerator()
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
