import json
import os
from unittest.mock import Mock

import pytest
from cohere.core import ApiError
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.utils import Secret
from haystack_integrations.components.generators.cohere import CohereChatGenerator

pytestmark = pytest.mark.chat_generators


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
        assert component.model == "command-r"
        assert component.streaming_callback is None
        assert component.api_base_url == "https://api.cohere.com"
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
                "model": "command-r",
                "streaming_callback": None,
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "https://api.cohere.com",
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
            model="command-r",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-r",
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
                "model": "command-r",
                "api_base_url": "test-base-url",
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = CohereChatGenerator.from_dict(data)
        assert component.model == "command-r"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-r",
                "api_base_url": "test-base-url",
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError):
            CohereChatGenerator.from_dict(data)

    def test_message_to_dict(self, chat_messages):
        obj = CohereChatGenerator(api_key=Secret.from_token("test-api-key"))
        dictionary = [obj._message_to_dict(message) for message in chat_messages]
        assert dictionary == [{"user_name": "Chatbot", "text": "What's the capital of France"}]

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
        with pytest.raises(ApiError):
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
        assert "Paris" in message.content

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
        assert "Paris" in message.content

        assert message.meta["finish_reason"] == "COMPLETE"

        assert "Paris" in callback.responses

        assert message.meta["documents"] is not None
        assert message.meta["citations"] is not None

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_tools_use(self):
        # See https://docs.anthropic.com/en/docs/tool-use for more information
        tools_schema = {
            "name": "get_stock_price",
            "description": "Retrieves the current stock price for a given ticker symbol.",
            "parameter_definitions": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                    "required": True,
                }
            },
        }
        client = CohereChatGenerator(model="command-r")
        response = client.run(
            messages=[ChatMessage.from_user("What is the current price of AAPL?")],
            generation_kwargs={"tools": [tools_schema]},
        )
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.content, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert "get_stock_price" in first_reply.content.lower(), "First reply does not contain get_stock_price"
        assert first_reply.meta, "First reply has no metadata"
        fc_response = json.loads(first_reply.content)
        assert "name" in fc_response, "First reply does not contain name of the tool"
        assert "parameters" in fc_response, "First reply does not contain parameters of the tool"
