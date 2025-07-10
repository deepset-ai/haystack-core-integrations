from datetime import datetime
from unittest.mock import patch

import pytest
import pytz
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils.auth import Secret
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack_integrations.components.generators.llama_stack.chat.chat_generator import LlamaStackChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


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


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="openai/gpt-4o-mini",
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


class TestLlamaStackChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = LlamaStackChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "llama3.2:3b"
        assert component.api_base_url == "http://localhost:8321/v1/openai/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            LlamaStackChatGenerator()

    def test_init_with_parameters(self):
        component = LlamaStackChatGenerator(
            model="llama3.2:3b",
            api_key=Secret.from_token("test-api-key"),
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.model == "llama3.2:3b"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self):
        component = LlamaStackChatGenerator()
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator"
        )

        expected_params = {
            "model": "llama3.2:3b",
            "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
            "streaming_callback": None,
            "api_base_url": "http://localhost:8321/v1/openai/v1",
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = LlamaStackChatGenerator(
            model="llama3.2:3b",
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=10,
            max_retries=10,
            tools=None,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator"
        )

        expected_params = {
            "model": "llama3.2:3b",
            "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
            "api_base_url": "test-base-url",
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
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        data = {
            "type": (
                "haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator"
            ),
            "init_parameters": {
                "model": "llama3.2:3b",
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10,
                "max_retries": 10,
                "tools": None,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }
        component = LlamaStackChatGenerator.from_dict(data)
        assert component.model == "llama3.2:3b"
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.timeout == 10
        assert component.max_retries == 10

    def test_run(self, chat_messages, mock_chat_completion):  # noqa: ARG002
        component = LlamaStackChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion):
        component = LlamaStackChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        # for LlamaStack
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5
        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
