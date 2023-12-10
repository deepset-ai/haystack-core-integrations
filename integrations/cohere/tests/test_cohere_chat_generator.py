import os
from unittest.mock import patch, Mock

import pytest
import cohere

from cohere_haystack.chat.chat_generator import CohereChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.components.generators.utils import default_streaming_callback

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
        mock_response.token_count =  {"prompt_tokens": 66, "response_tokens": 78, "total_tokens": 144, "billed_tokens": 133}
        mock_response.meta = {"api_version": {"version": "1"}, "billed_units": {"input_tokens": 55, "output_tokens": 78}}
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
    return [
        ChatMessage(content="What's the capital of France", role=None, name=None)
    ]

class TestCohereChatGenerator:
    @pytest.mark.unit
    def test_init_default(self):
        component = CohereChatGenerator(api_key="test-api-key")
        assert component.api_key == "test-api-key"
        assert component.model_name == "command"
        assert component.streaming_callback is None
        assert component.api_base_url == cohere.COHERE_API_URL
        assert not component.generation_kwargs

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="CohereChatGenerator needs an API key to run. Either provide it as init parameter or set the env var COHERE_API_KEY."):
            CohereChatGenerator()

    @pytest.mark.unit
    def test_init_with_parameters(self):
        component = CohereChatGenerator(
            api_key="test-api-key",
            model_name="command-nightly",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"}
        )
        assert component.api_key == "test-api-key"
        assert component.model_name == "command-nightly"
        assert component.streaming_callback is default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.unit
    def test_to_dict_default(self):
        component = CohereChatGenerator(api_key="test-api-key")
        data = component.to_dict()
        assert data == {
            "type": "cohere_haystack.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model_name":"command",
                "streaming_callback": None,
                "api_base_url": "https://api.cohere.ai",
                "generation_kwargs": {}
            }
        }

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        component = CohereChatGenerator(
            api_key="test-api-key",
            model_name="command-nightly",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "cohere_haystack.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model_name": "command-nightly",
                "streaming_callback": "haystack.components.generators.utils.default_streaming_callback",
                "api_base_url": "test-base-url",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"}
            }
        }

    @pytest.mark.unit
    def test_to_dict_with_lambda_streaming_callback(self):
        component = CohereChatGenerator(
            api_key="test-api-key",
            model_name="command",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "cohere_haystack.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model_name": "command",
                "api_base_url": "test-base-url",
                "streaming_callback": "tests.test_cohere_chat_generator.<lambda>",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "fake-api-key")
        data = {
            "type": "cohere_haystack.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model_name": "command",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.default_streaming_callback",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = CohereChatGenerator.from_dict(data)
        assert component.model_name == "command"
        assert component.streaming_callback is default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.unit
    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        data = {
            "type": "cohere_haystack.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model_name": "command",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.default_streaming_callback",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError, match="CohereChatGenerator needs an API key to run. Either provide it as init parameter or set the env var COHERE_API_KEY."):
            CohereChatGenerator.from_dict(data)

    @pytest.mark.unit
    def test_run(self, chat_messages, mock_chat_response):
        component = CohereChatGenerator(api_key="test-api-key")
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.unit
    def test_run_with_params(self, chat_messages, mock_chat_response):
        component = CohereChatGenerator(api_key="test-api-key", generation_kwargs={"max_tokens": 10, "temperature": 0.5})
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

    @pytest.mark.unit
    def test_run_streaming(self, chat_messages, mock_chat_response):
        streaming_call_count = 0

        # Define the streaming callback function and assert that it is called with StreamingChunk objects
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        generator = CohereChatGenerator(api_key="test-api-key", streaming_callback=streaming_callback_fn)

        # Create a fake streamed response
        # self needed here, don't remove
        def mock_iter(self):
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
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage(content="What's the capital of France", role=None, name="", metadata={})]
        component = CohereChatGenerator(
            api_key=os.environ.get("COHERE_API_KEY"),
            generation_kwargs={"temperature": 0.8}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = CohereChatGenerator(model_name="something-obviously-wrong", api_key=os.environ.get("COHERE_API_KEY"))
        with pytest.raises(cohere.CohereAPIError, match="model not found, make sure the correct model ID was used and that you have access to the model."):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
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
        component = CohereChatGenerator(os.environ.get("COHERE_API_KEY"), streaming_callback=callback)
        results = component.run([ChatMessage(content="What's the capital of France? answer in a word", role=None, name=None)])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content[0]

        assert message.metadata["finish_reason"] == "COMPLETE"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_connector(self):
        chat_messages = [ChatMessage(content="What's the capital of France", role=None, name="", metadata={})]
        component = CohereChatGenerator(
            api_key=os.environ.get("COHERE_API_KEY"),
            generation_kwargs={"temperature": 0.8}
            )
        results = component.run(chat_messages, generation_kwargs={"connectors":[{"id": "web-search"}]})
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content
        assert message.metadata["documents"] is not None
        assert message.metadata["citations"] is not None

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
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
        component = CohereChatGenerator(os.environ.get("COHERE_API_KEY"), streaming_callback=callback)
        results = component.run(chat_messages, generation_kwargs={"connectors":[{"id": "web-search"}]})

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content[0]

        assert message.metadata["finish_reason"] == "COMPLETE"

        assert callback.counter > 1
        assert "Paris" in callback.responses

        assert message.metadata["documents"] is not None
        assert message.metadata["citations"] is not None
