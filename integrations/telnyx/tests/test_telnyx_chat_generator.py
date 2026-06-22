# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from haystack_integrations.components.generators.telnyx.chat.chat_generator import TelnyxChatGenerator


class CalendarEvent(BaseModel):
    event_name: str
    event_date: str
    event_location: str


@pytest.fixture
def calendar_event_model():
    return CalendarEvent


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France?"),
    ]


@pytest.fixture
def mock_chat_completion():
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="openai/gpt-5.2",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=timezone.utc).timestamp()),
            usage=CompletionUsage(prompt_tokens=57, completion_tokens=40, total_tokens=97),
        )
        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


class TestTelnyxChatGenerator:
    def test_supported_models(self):
        models = TelnyxChatGenerator.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "test-api-key")
        component = TelnyxChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "openai/gpt-5.2"
        assert component.api_base_url == "https://api.telnyx.com/v2/ai/openai"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("TELNYX_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            TelnyxChatGenerator()

    def test_init_with_parameters(self):
        component = TelnyxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="zai-org/GLM-5.1-FP8",
            streaming_callback=print_streaming_chunk,
            api_base_url="https://custom-api-base-url.com",
            generation_kwargs={"max_tokens": 10, "temperature": 0.2},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "zai-org/GLM-5.1-FP8"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "https://custom-api-base-url.com"
        assert component.generation_kwargs == {"max_tokens": 10, "temperature": 0.2}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "test-api-key")
        component = TelnyxChatGenerator()
        data = component.to_dict()

        assert data["type"] == (
            "haystack_integrations.components.generators.telnyx.chat.chat_generator.TelnyxChatGenerator"
        )
        assert data["init_parameters"] == {
            "api_key": {"env_vars": ["TELNYX_API_KEY"], "strict": True, "type": "env_var"},
            "model": "openai/gpt-5.2",
            "streaming_callback": None,
            "api_base_url": "https://api.telnyx.com/v2/ai/openai",
            "generation_kwargs": {},
            "tools": None,
            "timeout": None,
            "max_retries": None,
            "http_client_kwargs": None,
        }

    def test_to_dict_with_parameters(self, monkeypatch, calendar_event_model):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = TelnyxChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="zai-org/GLM-5.1-FP8",
            streaming_callback=print_streaming_chunk,
            api_base_url="https://custom-api-base-url.com",
            generation_kwargs={"response_format": calendar_event_model},
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com:8080"},
        )
        data = component.to_dict()

        assert data["init_parameters"]["api_key"] == {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"}
        assert data["init_parameters"]["model"] == "zai-org/GLM-5.1-FP8"
        assert data["init_parameters"]["streaming_callback"] == (
            "haystack.components.generators.utils.print_streaming_chunk"
        )
        assert data["init_parameters"]["api_base_url"] == "https://custom-api-base-url.com"
        assert data["init_parameters"]["generation_kwargs"]["response_format"]["type"] == "json_schema"
        assert data["init_parameters"]["timeout"] == 10.0
        assert data["init_parameters"]["max_retries"] == 2
        assert data["init_parameters"]["http_client_kwargs"] == {"proxy": "https://proxy.example.com:8080"}

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.telnyx.chat.chat_generator.TelnyxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["TELNYX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "zai-org/GLM-5.1-FP8",
                "api_base_url": "https://custom-api-base-url.com",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10},
                "tools": None,
            },
        }
        component = TelnyxChatGenerator.from_dict(data)
        assert component.model == "zai-org/GLM-5.1-FP8"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "https://custom-api-base-url.com"
        assert component.generation_kwargs == {"max_tokens": 10}
        assert component.api_key == Secret.from_env_var("TELNYX_API_KEY")

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "fake-api-key")
        component = TelnyxChatGenerator()
        response = component.run(chat_messages)

        assert response["replies"][0].text == "Hello world!"
        mock_chat_completion.assert_called_once()
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["model"] == "openai/gpt-5.2"

    @pytest.mark.skipif(
        not os.environ.get("TELNYX_API_KEY", None),
        reason="Export TELNYX_API_KEY to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        component = TelnyxChatGenerator(generation_kwargs={"max_tokens": 32})
        result = component.run([ChatMessage.from_user("Reply with the word haystack.")])
        assert result["replies"][0].text
