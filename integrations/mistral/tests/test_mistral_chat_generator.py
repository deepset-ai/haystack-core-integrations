import os
from datetime import datetime
from typing import Iterator
from unittest.mock import patch

import pytest
import pytz
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils.auth import Secret
from haystack_integrations.components.generators.mistral.chat.chat_generator import MistralChatGenerator
from openai import OpenAIError, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


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
            model="mistral-tiny",
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
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture
def mock_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    class MockStream(Stream[ChatCompletionChunk]):
        def __init__(self, mock_chunk: ChatCompletionChunk, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mock_chunk = mock_chunk

        def __stream__(self) -> Iterator[ChatCompletionChunk]:
            # Yielding only one ChatCompletionChunk object
            yield self.mock_chunk

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    finish_reason="stop", logprobs=None, index=0, delta=ChoiceDelta(content="Hello", role="assistant")
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )
        mock_chat_completion_create.return_value = MockStream(completion, cast_to=None, response=None, client=None)
        yield mock_chat_completion_create


class TestMistralChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        component = MistralChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "mistral-tiny"
        assert component.api_base_url == "https://api.mistral.ai/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            MistralChatGenerator()

    def test_init_with_parameters(self):
        component = MistralChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-small",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "mistral-small"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        component = MistralChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-tiny",
                "organization": None,
                "streaming_callback": None,
                "api_base_url": "https://api.mistral.ai/v1",
                "generation_kwargs": {},
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = MistralChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="mistral-small",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "model": "mistral-small",
                "api_base_url": "test-base-url",
                "organization": None,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-small",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = MistralChatGenerator.from_dict(data)
        assert component.model == "mistral-small"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("MISTRAL_API_KEY")

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-small",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            MistralChatGenerator.from_dict(data)

    def test_run(self, chat_messages):
        component = MistralChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion):
        component = MistralChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
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

    def test_run_with_params_streaming(self, chat_messages):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:  # noqa: ARG001
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = MistralChatGenerator(streaming_callback=streaming_callback)
        response = component.run(chat_messages)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Paris" in response["replies"][0].content  # see mock_chat_completion_chunk

    def test_check_abnormal_completions(self, caplog):
        component = MistralChatGenerator(api_key=Secret.from_token("test-api-key"))
        messages = [
            ChatMessage.from_assistant(
                "", meta={"finish_reason": "content_filter" if i % 2 == 0 else "length", "index": i}
            )
            for i, _ in enumerate(range(4))
        ]

        for m in messages:
            component._check_finish_reason(m)

        # check truncation warning
        message_template = (
            "The completion for index {index} has been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions."
        )

        for index in [1, 3]:
            assert caplog.records[index].message == message_template.format(index=index)

        # check content filter warning
        message_template = "The completion for index {index} has been truncated due to the content filter."
        for index in [0, 2]:
            assert caplog.records[index].message == message_template.format(index=index)

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = MistralChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content
        assert "mistral-tiny" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = MistralChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the OpenAI API key to run this test.",
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
        component = MistralChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content

        assert "mistral-tiny" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses
