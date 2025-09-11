import os
from datetime import datetime
from unittest.mock import patch

import pytest
import pytz
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils.auth import Secret
from openai import OpenAIError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack_integrations.components.generators.stackit.chat.chat_generator import STACKITChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage=CompletionUsage(prompt_tokens=57, completion_tokens=40, total_tokens=97),
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


class TestSTACKITChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")
        component = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8")
        assert component.client.api_key == "test-api-key"
        assert component.model == "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
        assert component.api_base_url == "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("STACKIT_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8")

    def test_init_with_parameters(self):
        component = STACKITChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")
        component = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8")
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
            "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
            "streaming_callback": None,
            "api_base_url": "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = STACKITChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            "timeout": 10.0,
            "max_retries": 2,
            "http_client_kwargs": {"proxy": "https://proxy.example.com:8080"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
                "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = STACKITChatGenerator.from_dict(data)
        assert component.model == "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("STACKIT_API_KEY")

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("STACKIT_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
                "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            STACKITChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("STACKIT_API_KEY", "fake-api-key")
        component = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8")
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "fake-api-key")
        component = STACKITChatGenerator(
            model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8", generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self) -> None:
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = STACKITChatGenerator(model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = STACKITChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the API key to run this test.",
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
        component = STACKITChatGenerator(
            model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8", streaming_callback=callback
        )
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses
