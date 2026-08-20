# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret
from openai.types.responses import Response

from haystack_integrations.components.generators.parallel.chat import (
    chat_generator as chat_generator_module,
)
from haystack_integrations.components.generators.parallel.chat.chat_generator import (
    ParallelChatGenerator,
)


def _make_response() -> Response:
    response_data = {
        "id": "resp_test",
        "created_at": datetime.now(tz=timezone.utc).timestamp(),
        "model": "parallel",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }

    if hasattr(Response, "model_construct"):
        return Response.model_construct(**response_data)
    return Response.construct(**response_data)


def _make_transport(captured: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status_code=200, json=_make_response().model_dump(mode="json"))

    return httpx.MockTransport(handler)


@pytest.fixture
def chat_messages() -> list[ChatMessage]:
    return [ChatMessage.from_user("What did Parallel Web Systems announce this year?")]


class TestParallelChatGenerator:
    def test_attribution_header_falls_back_when_package_is_not_installed(self, monkeypatch):
        monkeypatch.setattr(chat_generator_module, "_PACKAGE_NAME", "definitely-not-installed-package")
        assert chat_generator_module._attribution_header() == "haystack/unknown"

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "test-api-key")
        component = ParallelChatGenerator()
        assert component.api_key == Secret.from_env_var("PARALLEL_API_KEY")
        assert component.model == "parallel"
        assert component.api_base_url == "https://api.parallel.ai/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_missing_api_key_raises_on_use(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        # haystack-ai 2.x resolves the secret at init, >= 3.0 at warm-up
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            component = ParallelChatGenerator()
            component.warm_up()

    def test_init_with_parameters(self):
        component = ParallelChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"reasoning": {"effort": "high"}},
            timeout=10,
            max_retries=2,
            extra_headers={"test-header": "test-value"},
        )
        assert component.model == "parallel"
        assert component.generation_kwargs == {"reasoning": {"effort": "high"}}
        assert component.extra_headers == {"test-header": "test-value"}

    def test_supported_models_listed(self):
        assert ParallelChatGenerator.SUPPORTED_MODELS == ["parallel"]

    def test_to_dict_default_round_trip(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "test-api-key")
        component = ParallelChatGenerator()
        data = component.to_dict()

        assert data["type"].endswith("chat_generator.ParallelChatGenerator")
        init_parameters = data["init_parameters"]
        assert init_parameters["model"] == "parallel"
        assert init_parameters["api_base_url"] == "https://api.parallel.ai/v1"
        assert "organization" not in init_parameters
        assert "tools" not in init_parameters

        deserialized = ParallelChatGenerator.from_dict(data)
        assert deserialized.model == "parallel"
        assert deserialized.api_base_url == "https://api.parallel.ai/v1"

    def test_to_dict_with_parameters_round_trip(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = ParallelChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            generation_kwargs={"reasoning": {"effort": "low"}},
            timeout=10,
            max_retries=2,
            extra_headers={"test-header": "test-value"},
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()
        # the serialized http_client_kwargs is the user-provided value, without attribution headers
        assert data["init_parameters"]["http_client_kwargs"] == {"proxy": "http://localhost:8080"}

        deserialized = ParallelChatGenerator.from_dict(data)
        assert deserialized.api_key == Secret.from_env_var("ENV_VAR")
        assert deserialized.generation_kwargs == {"reasoning": {"effort": "low"}}
        assert deserialized.extra_headers == {"test-header": "test-value"}
        assert deserialized.timeout == 10
        assert deserialized.max_retries == 2
        assert deserialized._http_client_kwargs == {"proxy": "http://localhost:8080"}

    def test_run_uses_responses_create(self, chat_messages):
        component = ParallelChatGenerator(api_key=Secret.from_token("test-api-key"))

        with patch("openai.resources.responses.Responses.create", return_value=_make_response()) as mock_create:
            result = component.run(chat_messages)

        assert len(result["replies"]) == 1
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "parallel"
        assert "input" in call_kwargs
        assert "messages" not in call_kwargs
        assert call_kwargs["stream"] is False

    def test_run_forwards_reasoning_effort(self, chat_messages):
        component = ParallelChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"reasoning": {"effort": "high"}},
        )

        with patch("openai.resources.responses.Responses.create", return_value=_make_response()) as mock_create:
            component.run(chat_messages)

        assert mock_create.call_args.kwargs["reasoning"] == {"effort": "high"}

    def test_http_client_kwargs_with_headers_merges_extra_and_attribution(self):
        kwargs = chat_generator_module._http_client_kwargs_with_headers(
            {"headers": {"existing-header": "existing-value"}},
            {"test-header": "test-value"},
        )

        assert kwargs["headers"]["existing-header"] == "existing-value"
        assert kwargs["headers"]["test-header"] == "test-value"
        assert kwargs["headers"]["x-parallel-integration"].startswith("haystack/")

    def test_run_sends_attribution_header(self, chat_messages):
        captured: list[httpx.Request] = []
        component = ParallelChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            extra_headers={"test-header": "test-value"},
            http_client_kwargs={"transport": _make_transport(captured)},
        )

        component.run(chat_messages)

        assert len(captured) == 1
        request = captured[0]
        assert str(request.url) == "https://api.parallel.ai/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert request.headers["x-parallel-integration"].startswith("haystack/")
        assert request.headers["test-header"] == "test-value"

    @pytest.mark.asyncio
    async def test_run_async_sends_attribution_header(self, chat_messages):
        captured: list[httpx.Request] = []
        component = ParallelChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            extra_headers={"test-header": "test-value"},
            http_client_kwargs={"transport": _make_transport(captured)},
        )

        await component.run_async(chat_messages)

        assert len(captured) == 1
        request = captured[0]
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert request.headers["x-parallel-integration"].startswith("haystack/")


@pytest.mark.skipif(
    not os.environ.get("PARALLEL_API_KEY"),
    reason="Export PARALLEL_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestParallelChatGeneratorInference:
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("In one sentence, what does Parallel Web Systems do?")]
        component = ParallelChatGenerator(generation_kwargs={"reasoning": {"effort": "low"}})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("In one sentence, what does Parallel Web Systems do?")]
        component = ParallelChatGenerator(generation_kwargs={"reasoning": {"effort": "low"}})
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        assert results["replies"][0].text
