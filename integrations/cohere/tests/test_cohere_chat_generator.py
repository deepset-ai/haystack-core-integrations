import os
from unittest.mock import Mock, patch

import cohere
import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.utils import Secret
from haystack_integrations.components.generators.cohere import CohereChatGenerator

pytestmark = pytest.mark.chat_generators


@pytest.fixture
def mock_chat_response():
    """
    Mock the CohereI API response and reuse it for tests
    """
    with patch("cohere.Client.chat", autospec=True) as mock_chat_response:
        # mimic the response from the Cohere API

        mock_response = Mock()
        mock_response.text = "I'm fine, thanks."
        mock_response.token_count = {
            "prompt_tokens": 66,
            "response_tokens": 78,
            "total_tokens": 144,
            "billed_tokens": 133,
        }
        mock_response.meta = {
            "api_version": {"version": "1"},
            "billed_units": {"input_tokens": 55, "output_tokens": 78},
        }
        mock_chat_response.return_value = mock_response
        yield mock_chat_response


def streaming_chunk(text: str):
    """
    Mock chunks of streaming responses from the Cohere API
    """
    # mimic the chunk response from the OpenAI API
    mock_chunks = Mock()
    mock_chunks.index = 0
    mock_chunks.text = text
    mock_chunks.event_type = "text-generation"
    return mock_chunks


@pytest.fixture
def chat_messages():
    return [ChatMessage(content="What's the capital of France", role=ChatRole.ASSISTANT, name=None)]


class TestCohereChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")

        component = CohereChatGenerator()
        assert component.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert component.model == "command"
        assert component.streaming_callback is None
        assert component.api_base_url == cohere.COHERE_API_URL
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        with pytest.raises(ValueError):
            CohereChatGenerator()

    def test_init_with_parameters(self):
        component = CohereChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="command-nightly",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.api_key == Secret.from_token("test-api-key")
        assert component.model == "command-nightly"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component = CohereChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command",
                "streaming_callback": None,
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "https://api.cohere.ai",
                "generation_kwargs": {},
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        monkeypatch.setenv("CO_API_KEY", "fake-api-key")
        component = CohereChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="command-nightly",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-nightly",
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "api_base_url": "test-base-url",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component = CohereChatGenerator(
            model="command",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command",
                "api_base_url": "test-base-url",
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "streaming_callback": "tests.test_cohere_chat_generator.<lambda>",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "fake-api-key")
        monkeypatch.setenv("CO_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command",
                "api_base_url": "test-base-url",
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = CohereChatGenerator.from_dict(data)
        assert component.model == "command"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command",
                "api_base_url": "test-base-url",
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError):
            CohereChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_response):  # noqa: ARG002
        component = CohereChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_message_to_dict(self, chat_messages):
        obj = CohereChatGenerator(api_key=Secret.from_token("test-api-key"))
        dictionary = [obj._message_to_dict(message) for message in chat_messages]
        assert dictionary == [{"user_name": "Chatbot", "text": "What's the capital of France"}]

    def test_run_with_params(self, chat_messages, mock_chat_response):
        component = CohereChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(chat_messages)

        # check that the component calls the Cohere API with the correct parameters
        _, kwargs = mock_chat_response.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_streaming(self, chat_messages, mock_chat_response):
        streaming_call_count = 0

        # Define the streaming callback function and assert that it is called with StreamingChunk objects
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        generator = CohereChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback_fn
        )

        # Create a fake streamed response
        # self needed here, don't remove
        def mock_iter(self):  # noqa: ARG001
            yield streaming_chunk("Hello")
            yield streaming_chunk("How are you?")

        mock_response = Mock(**{"__iter__": mock_iter})
        mock_chat_response.return_value = mock_response

        response = generator.run(chat_messages)

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage(content="What's the capital of France", role=ChatRole.USER, name="", meta={})]
        component = CohereChatGenerator(generation_kwargs={"temperature": 0.8})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = CohereChatGenerator(model="something-obviously-wrong")
        with pytest.raises(cohere.CohereAPIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
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
        component = CohereChatGenerator(streaming_callback=callback)
        results = component.run(
            [ChatMessage(content="What's the capital of France? answer in a word", role=ChatRole.USER, name=None)]
        )

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content[0]

        assert message.meta["finish_reason"] == "COMPLETE"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_connector(self):
        chat_messages = [ChatMessage(content="What's the capital of France", role=ChatRole.USER, name="", meta={})]
        component = CohereChatGenerator(generation_kwargs={"temperature": 0.8})
        results = component.run(chat_messages, generation_kwargs={"connectors": [{"id": "web-search"}]})
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content
        assert message.meta["documents"] is not None
        assert "citations" in message.meta  # Citations might be None

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming_with_connector(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        chat_messages = [ChatMessage(content="What's the capital of France? answer in a word", role=None, name=None)]
        component = CohereChatGenerator(streaming_callback=callback)
        results = component.run(chat_messages, generation_kwargs={"connectors": [{"id": "web-search"}]})

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content[0]

        assert message.meta["finish_reason"] == "COMPLETE"

        assert "Paris" in callback.responses

        assert message.meta["documents"] is not None
        assert message.meta["citations"] is not None
