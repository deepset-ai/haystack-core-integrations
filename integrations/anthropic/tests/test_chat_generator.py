import json
import os

import anthropic
import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("\\nYou are a helpful assistant, be super brief in your responses."),
        ChatMessage.from_user("What's the capital of France?"),
    ]


class TestAnthropicChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-3-5-sonnet-20240620"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.ignore_tools_thinking_messages

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            AnthropicChatGenerator()

    def test_init_with_parameters(self):
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="claude-3-5-sonnet-20240620",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            ignore_tools_thinking_messages=False,
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-3-5-sonnet-20240620"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.ignore_tools_thinking_messages is False

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "strict": True, "type": "env_var"},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": None,
                "generation_kwargs": {},
                "ignore_tools_thinking_messages": True,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator(
            model="claude-3-5-sonnet-20240620",
            streaming_callback=lambda x: x,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "strict": True, "type": "env_var"},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": "tests.test_chat_generator.<lambda>",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "strict": True, "type": "env_var"},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
            },
        }
        component = AnthropicChatGenerator.from_dict(data)
        assert component.model == "claude-3-5-sonnet-20240620"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("ANTHROPIC_API_KEY")

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "strict": True, "type": "env_var"},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
            },
        }
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            AnthropicChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion):
        component = AnthropicChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion):
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(chat_messages)

        # check that the component calls the Anthropic API with the correct parameters
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
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = AnthropicChatGenerator(model="something-obviously-wrong")
        with pytest.raises(anthropic.NotFoundError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_default_inference_params(self, chat_messages):
        client = AnthropicChatGenerator()
        response = client.run(chat_messages)

        assert "replies" in response, "Response does not contain 'replies' key"
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.content, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert "paris" in first_reply.content.lower(), "First reply does not contain 'paris'"
        assert first_reply.meta, "First reply has no metadata"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_default_inference_with_streaming(self, chat_messages):
        streaming_callback_called = False
        paris_found_in_response = False

        def streaming_callback(chunk: StreamingChunk):
            nonlocal streaming_callback_called, paris_found_in_response
            streaming_callback_called = True
            assert isinstance(chunk, StreamingChunk)
            assert chunk.content is not None
            if not paris_found_in_response:
                paris_found_in_response = "paris" in chunk.content.lower()

        client = AnthropicChatGenerator(streaming_callback=streaming_callback)
        response = client.run(chat_messages)

        assert streaming_callback_called, "Streaming callback was not called"
        assert paris_found_in_response, "The streaming callback response did not contain 'paris'"
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.content, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert "paris" in first_reply.content.lower(), "First reply does not contain 'paris'"
        assert first_reply.meta, "First reply has no metadata"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_tools_use(self):
        # See https://docs.anthropic.com/en/docs/tool-use for more information
        tools_schema = {
            "name": "get_stock_price",
            "description": "Retrieves the current stock price for a given ticker symbol.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."}
                },
                "required": ["ticker"],
            },
        }
        client = AnthropicChatGenerator()
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
        assert "input" in fc_response, "First reply does not contain input of the tool"

    def test_prompt_caching_enabled(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly passed to the Anthropic API when prompt
        caching is enabled
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator(
            generation_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}}
        )
        assert component.generation_kwargs.get("extra_headers", {}).get("anthropic-beta") == "prompt-caching-2024-07-31"

    def test_prompt_caching_cache_control_without_extra_headers(self, monkeypatch, mock_chat_completion, caplog):
        """
        Test that the cache_control is removed from the messages when prompt caching is not enabled via extra_headers
        This is to avoid Anthropic errors when prompt caching is not enabled
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()

        messages = [ChatMessage.from_system("System message"), ChatMessage.from_user("User message")]

        # Add cache_control to messages
        for msg in messages:
            msg.meta["cache_control"] = {"type": "ephemeral"}

        # Invoke run with messages
        component.run(messages)

        # Check caplog for the warning message that should have been logged
        assert any("Prompt caching" in record.message for record in caplog.records)

        # Check that the Anthropic API was called without cache_control in messages so that it does not raise an error
        _, kwargs = mock_chat_completion.call_args
        for msg in kwargs["messages"]:
            assert "cache_control" not in msg

    @pytest.mark.parametrize("enable_caching", [True, False])
    def test_run_with_prompt_caching(self, monkeypatch, mock_chat_completion, enable_caching):
        """
        Test that the generation_kwargs extra_headers are correctly passed to the Anthropic API in both cases of
        prompt caching being enabled or not
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        generation_kwargs = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} if enable_caching else {}
        component = AnthropicChatGenerator(generation_kwargs=generation_kwargs)

        messages = [ChatMessage.from_system("System message"), ChatMessage.from_user("User message")]

        component.run(messages)

        # Check that the Anthropic API was called with the correct headers
        _, kwargs = mock_chat_completion.call_args
        headers = kwargs.get("extra_headers", {})
        if enable_caching:
            assert "anthropic-beta" in headers
        else:
            assert "anthropic-beta" not in headers

    def test_to_dict_with_prompt_caching(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly serialized to a dictionary
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator(
            generation_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}}
        )
        data = component.to_dict()
        assert (
            data["init_parameters"]["generation_kwargs"]["extra_headers"]["anthropic-beta"]
            == "prompt-caching-2024-07-31"
        )

    def test_from_dict_with_prompt_caching(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly deserialized from a dictionary
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "strict": True, "type": "env_var"},
                "model": "claude-3-5-sonnet-20240620",
                "generation_kwargs": {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}},
            },
        }
        component = AnthropicChatGenerator.from_dict(data)
        assert component.generation_kwargs["extra_headers"]["anthropic-beta"] == "prompt-caching-2024-07-31"

    def test_convert_messages_to_anthropic_format(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        generator = AnthropicChatGenerator()

        # Test scenario 1: Regular user and assistant messages
        messages = [
            ChatMessage.from_user("Hello"),
            ChatMessage.from_assistant("Hi there!"),
        ]
        result = generator._convert_to_anthropic_format(messages)
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Test scenario 2: System message
        messages = [ChatMessage.from_system("You are a helpful assistant.")]
        result = generator._convert_to_anthropic_format(messages)
        assert result == [{"type": "text", "text": "You are a helpful assistant."}]

        # Test scenario 3: Mixed message types
        messages = [
            ChatMessage.from_system("Be concise."),
            ChatMessage.from_user("What's AI?"),
            ChatMessage.from_assistant("Artificial Intelligence."),
        ]
        result = generator._convert_to_anthropic_format(messages)
        assert result == [
            {"type": "text", "text": "Be concise."},
            {"role": "user", "content": "What's AI?"},
            {"role": "assistant", "content": "Artificial Intelligence."},
        ]

        # Test scenario 4: metadata
        messages = [
            ChatMessage.from_user("What's AI?"),
            ChatMessage.from_assistant("Artificial Intelligence.", meta={"confidence": 0.9}),
        ]
        result = generator._convert_to_anthropic_format(messages)
        assert result == [
            {"role": "user", "content": "What's AI?"},
            {"role": "assistant", "content": "Artificial Intelligence.", "confidence": 0.9},
        ]

        # Test scenario 5: Empty message list
        assert generator._convert_to_anthropic_format([]) == []

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY", None), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.parametrize("cache_enabled", [True, False])
    def test_prompt_caching(self, cache_enabled):
        generation_kwargs = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} if cache_enabled else {}

        claude_llm = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY"), generation_kwargs=generation_kwargs
        )

        # see https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#cache-limitations
        system_message = ChatMessage.from_system("This is the cached, here we make it at least 1024 tokens long." * 70)
        if cache_enabled:
            system_message.meta["cache_control"] = {"type": "ephemeral"}

        messages = [system_message, ChatMessage.from_user("What's in cached content?")]
        result = claude_llm.run(messages)

        assert "replies" in result
        assert len(result["replies"]) == 1
        token_usage = result["replies"][0].meta.get("usage")

        if cache_enabled:
            # either we created cache or we read it (depends on how you execute this integration test)
            assert (
                token_usage.get("cache_creation_input_tokens") > 1024
                or token_usage.get("cache_read_input_tokens") > 1024
            )
        else:
            assert "cache_creation_input_tokens" not in token_usage
            assert "cache_read_input_tokens" not in token_usage
