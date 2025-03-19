# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from unittest.mock import patch

import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from openai.types.chat import ChatCompletion
from requests_mock import Mocker

from haystack_integrations.components.generators.nvidia.chat.alternative_chat_generator import (
    AlternativeNvidiaChatGenerator,
)


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("What is the answer to life, the universe, and everything?"),
    ]


@pytest.fixture
def openai_mock_chat_completion():
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="gpt-4",
            object="chat.completion",
            choices=[
                {
                    "finish_reason": "stop",
                    "logprobs": None,
                    "index": 0,
                    "message": {"content": "Hello world!", "role": "assistant"},
                }
            ],
            created=int(datetime.now().timestamp()),  # noqa: DTZ005
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture
def mock_local_chat_completion(requests_mock: Mocker) -> None:
    requests_mock.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "choices": [
                {
                    "message": {"content": "The answer is 42.", "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "total_tokens": 30,
                "completion_tokens": 5,
            },
            "model": "meta/llama3-70b-instruct",
        },
    )


class TestAlternativeNvidiaChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = AlternativeNvidiaChatGenerator("meta/llama3-70b-instruct")

        assert generator.api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert generator.model == "meta/llama3-70b-instruct"
        assert generator.generation_kwargs == {}

    def test_init_with_parameters(self):
        generator = AlternativeNvidiaChatGenerator(
            api_key=Secret.from_token("fake-api-key"),
            model="meta/llama3-70b-instruct",
            generation_kwargs={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
            },
        )
        assert generator.api_key == Secret.from_token("fake-api-key")
        assert generator.model == "meta/llama3-70b-instruct"
        assert generator.generation_kwargs == {
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
        }

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AlternativeNvidiaChatGenerator("meta/llama3-70b-instruct")

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = AlternativeNvidiaChatGenerator("meta/llama3-70b-instruct")
        data = generator.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.nvidia.chat."
            "alternative_chat_generator.AlternativeNvidiaChatGenerator",
            "init_parameters": {
                "api_base_url": "https://integrate.api.nvidia.com/v1",
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "meta/llama3-70b-instruct",
                "generation_kwargs": {},
                "streaming_callback": None,
                "timeout": 60.0,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.nvidia.chat."
            "alternative_chat_generator.AlternativeNvidiaChatGenerator",
            "init_parameters": {
                "api_base_url": "https://my.url.com/v1",
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "meta/llama3-70b-instruct",
                "generation_kwargs": {
                    "temperature": 0.2,
                    "top_p": 0.7,
                    "max_tokens": 1024,
                },
                "streaming_callback": None,
                "timeout": 60.0,
            },
        }
        generator = AlternativeNvidiaChatGenerator.from_dict(data)
        assert generator.api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert generator.model == "meta/llama3-70b-instruct"
        assert generator.generation_kwargs == {
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
        }
        assert generator.api_base_url == "https://my.url.com/v1"

    def test_run(self, chat_messages, openai_mock_chat_completion):
        generator = AlternativeNvidiaChatGenerator(
            model="meta/llama3-70b-instruct", api_key=Secret.from_token("test-api-key")
        )

        response = generator.run(chat_messages)

        _, kwargs = openai_mock_chat_completion.call_args
        assert kwargs["model"] == "meta/llama3-70b-instruct"

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_generation_kwargs(self, chat_messages, openai_mock_chat_completion):
        generator = AlternativeNvidiaChatGenerator(
            model="meta/llama3-70b-instruct",
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"temperature": 0.7},
        )

        generator.run(messages=chat_messages, generation_kwargs={"max_tokens": 100, "temperature": 0.2})

        # Verify parameters are merged correctly (run kwargs override init kwargs)
        # and the component calls the API with the correct parameters
        _, kwargs = openai_mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.2

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        generator = AlternativeNvidiaChatGenerator(
            model="meta/llama3-70b-instruct",
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            generation_kwargs={"temperature": 0.2, "max_tokens": 50},
        )

        messages = [
            ChatMessage.from_system("You are a helpful assistant. Keep your answers brief."),
            ChatMessage.from_user("What is the answer to life, the universe, and everything?"),
        ]

        result = generator.run(messages=messages)

        assert "replies" in result
        assert len(result["replies"]) == 1
        assert isinstance(result["replies"][0], ChatMessage)
        assert len(result["replies"][0].text) > 0
