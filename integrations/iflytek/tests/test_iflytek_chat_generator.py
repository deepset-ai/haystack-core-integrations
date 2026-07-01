import os
from unittest.mock import patch

import pytest
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils.auth import Secret
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from haystack_integrations.components.generators.iflytek import IFlytekChatGenerator

TYPE_NAME = "haystack_integrations.components.generators.iflytek.chat.chat_generator.IFlytekChatGenerator"


@pytest.fixture
def chat_messages():
    return [ChatMessage.from_user("What's the capital of France")]


@pytest.fixture
def mock_chat_completion():
    with patch("openai.resources.chat.completions.Completions.create") as mock_create:
        completion = ChatCompletion(
            id="chatcmpl-spark-1",
            object="chat.completion",
            created=1677652288,
            model="generalv3.5",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content="Paris"),
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        )
        mock_create.return_value = completion
        yield mock_create


class TestIFlytekChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("IFLYTEK_API_KEY", "test-api-key")
        component = IFlytekChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "generalv3.5"
        assert component.api_base_url == "https://spark-api-open.xf-yun.com/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("IFLYTEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            IFlytekChatGenerator()

    def test_init_with_parameters(self):
        component = IFlytekChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="4.0Ultra",
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "temperature": 0.3},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "4.0Ultra"
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "temperature": 0.3}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("IFLYTEK_API_KEY", "test-api-key")
        component = IFlytekChatGenerator()
        data = component.to_dict()

        assert data["type"] == TYPE_NAME

        expected_params = {
            "api_key": {"env_vars": ["IFLYTEK_API_KEY"], "strict": True, "type": "env_var"},
            "model": "generalv3.5",
            "streaming_callback": None,
            "api_base_url": "https://spark-api-open.xf-yun.com/v1",
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
        }
        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("IFLYTEK_API_KEY", "test-api-key")
        data = {
            "type": TYPE_NAME,
            "init_parameters": {
                "api_key": {"env_vars": ["IFLYTEK_API_KEY"], "strict": True, "type": "env_var"},
                "model": "4.0Ultra",
                "api_base_url": "https://spark-api-open.xf-yun.com/v1",
                "streaming_callback": None,
                "generation_kwargs": {"max_tokens": 10},
            },
        }
        component = IFlytekChatGenerator.from_dict(data)
        assert isinstance(component, IFlytekChatGenerator)
        assert component.model == "4.0Ultra"
        assert component.api_base_url == "https://spark-api-open.xf-yun.com/v1"
        assert component.generation_kwargs == {"max_tokens": 10}
        assert component.api_key == Secret.from_env_var("IFLYTEK_API_KEY")

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("IFLYTEK_API_KEY", "test-api-key")
        component = IFlytekChatGenerator()
        response = component.run(chat_messages)

        assert isinstance(response, dict)
        assert "replies" in response
        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert reply.role == ChatRole.ASSISTANT
        assert reply.text == "Paris"
        assert mock_chat_completion.called

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("IFLYTEK_API_KEY"),
        reason="Export an env var called IFLYTEK_API_KEY to run this test.",
    )
    def test_live_run(self):
        component = IFlytekChatGenerator()
        results = component.run([ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0].text
