import os
from unittest.mock import AsyncMock, patch

import anthropic
import pytest
from anthropic.types import Message
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole

from haystack_integrations.components.generators.anthropic import AnthropicFoundryChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("\nYou are a helpful assistant, be super brief in your responses."),
        ChatMessage.from_user("What's the capital of France?"),
    ]


class TestAnthropicFoundryChatGenerator:
    def test_supported_models(self):
        assert isinstance(AnthropicFoundryChatGenerator.SUPPORTED_MODELS, list)
        assert len(AnthropicFoundryChatGenerator.SUPPORTED_MODELS) > 0
        assert all(isinstance(m, str) for m in AnthropicFoundryChatGenerator.SUPPORTED_MODELS)
        # No old claude-3 models
        assert not any("claude-3" in m for m in AnthropicFoundryChatGenerator.SUPPORTED_MODELS)

    def test_init_with_api_key_and_resource(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        component = AnthropicFoundryChatGenerator(resource="my-resource")
        assert component.resource == "my-resource"
        assert component.endpoint is None
        assert component.model == "claude-sonnet-4-5"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.ignore_tools_thinking_messages
        assert component.timeout is None
        assert component.max_retries is None
        assert component.azure_ad_token_provider is None
        # Clients are not created until warm_up
        assert component.client is None
        assert component.async_client is None

    def test_init_with_endpoint(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        component = AnthropicFoundryChatGenerator(endpoint="https://my-resource.openai.azure.com/anthropic")
        assert component.resource is None
        assert component.endpoint == "https://my-resource.openai.azure.com/anthropic"

    def test_init_resource_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "env-resource")
        component = AnthropicFoundryChatGenerator()
        assert component.resource == "env-resource"

    def test_init_with_azure_ad_token_provider_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
        token_provider = lambda: "my-token"  # noqa: E731
        component = AnthropicFoundryChatGenerator(
            resource="my-resource",
            azure_ad_token_provider=token_provider,
        )
        assert component.azure_ad_token_provider is token_provider

    def test_init_no_auth_raises(self):
        with pytest.raises(ValueError, match="Please provide an API key or an azure_ad_token_provider"):
            AnthropicFoundryChatGenerator(api_key=None, resource="my-resource")

    def test_init_missing_resource_and_endpoint_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
        with pytest.raises(ValueError, match="Either 'resource' or 'endpoint' must be provided"):
            AnthropicFoundryChatGenerator()

    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        component = AnthropicFoundryChatGenerator(resource="my-resource")
        assert component.client is None
        assert component.async_client is None
        component.warm_up()
        assert component.client is not None
        assert component.async_client is not None

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        component = AnthropicFoundryChatGenerator(resource="my-resource")
        data = component.to_dict()
        assert data == {
            "type": (
                "haystack_integrations.components.generators."
                "anthropic.chat.foundry_chat_generator.AnthropicFoundryChatGenerator"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_FOUNDRY_API_KEY"], "strict": True, "type": "env_var"},
                "resource": "my-resource",
                "endpoint": None,
                "model": "claude-sonnet-4-5",
                "streaming_callback": None,
                "generation_kwargs": {},
                "ignore_tools_thinking_messages": True,
                "tools": None,
                "timeout": None,
                "max_retries": None,
                "azure_ad_token_provider": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        component = AnthropicFoundryChatGenerator(
            resource="my-resource",
            model="claude-opus-4-6",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            ignore_tools_thinking_messages=False,
            timeout=10.0,
            max_retries=1,
        )
        data = component.to_dict()
        assert data == {
            "type": (
                "haystack_integrations.components.generators."
                "anthropic.chat.foundry_chat_generator.AnthropicFoundryChatGenerator"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_FOUNDRY_API_KEY"], "strict": True, "type": "env_var"},
                "resource": "my-resource",
                "endpoint": None,
                "model": "claude-opus-4-6",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": False,
                "tools": None,
                "timeout": 10.0,
                "max_retries": 1,
                "azure_ad_token_provider": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        data = {
            "type": (
                "haystack_integrations.components.generators."
                "anthropic.chat.foundry_chat_generator.AnthropicFoundryChatGenerator"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_FOUNDRY_API_KEY"], "strict": True, "type": "env_var"},
                "resource": "my-resource",
                "endpoint": None,
                "model": "claude-sonnet-4-5",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
                "tools": None,
                "timeout": None,
                "max_retries": None,
                "azure_ad_token_provider": None,
            },
        }
        component = AnthropicFoundryChatGenerator.from_dict(data)
        assert component.model == "claude-sonnet-4-5"
        assert component.resource == "my-resource"
        assert component.endpoint is None
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.ignore_tools_thinking_messages is True
        assert component.timeout is None
        assert component.max_retries is None
        assert component.azure_ad_token_provider is None

    def test_to_dict_from_dict_roundtrip(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        original = AnthropicFoundryChatGenerator(
            resource="my-resource",
            model="claude-sonnet-4-6",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 500},
            timeout=15.0,
            max_retries=2,
        )
        restored = AnthropicFoundryChatGenerator.from_dict(original.to_dict())
        assert restored.model == original.model
        assert restored.resource == original.resource
        assert restored.streaming_callback is original.streaming_callback
        assert restored.generation_kwargs == original.generation_kwargs
        assert restored.timeout == original.timeout
        assert restored.max_retries == original.max_retries

    def test_run_triggers_warm_up(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        component = AnthropicFoundryChatGenerator(resource="my-resource")
        assert not component._is_warmed_up
        response = component.run(chat_messages)
        assert component._is_warmed_up
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        component = AnthropicFoundryChatGenerator(
            resource="my-resource",
            generation_kwargs={"max_tokens": 10, "temperature": 0.5},
        )
        response = component.run(chat_messages)

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        assert isinstance(response, dict)
        assert "replies" in response
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") or not os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"),
        reason="Set ANTHROPIC_FOUNDRY_API_KEY and ANTHROPIC_FOUNDRY_RESOURCE env variables to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = AnthropicFoundryChatGenerator(
            model="something-obviously-wrong",
            resource=os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"),
        )
        with pytest.raises(anthropic.NotFoundError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") or not os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"),
        reason="Set ANTHROPIC_FOUNDRY_API_KEY and ANTHROPIC_FOUNDRY_RESOURCE env variables to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self, chat_messages):
        client = AnthropicFoundryChatGenerator(
            resource=os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"), model="claude-sonnet-4-5"
        )
        response = client.run(chat_messages)

        assert "replies" in response
        replies = response["replies"]
        assert isinstance(replies, list)
        assert len(replies) > 0

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage)
        assert first_reply.text
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT)
        assert "paris" in first_reply.text.lower()
        assert first_reply.meta

    # Anthropic messages API is similar for AnthropicFoundry and Anthropic endpoint;
    # remaining tests are skipped as they are already tested in AnthropicChatGenerator.


class TestAnthropicFoundryChatGeneratorAsync:
    @pytest.mark.asyncio
    async def test_run_async_triggers_warm_up(self, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
        component = AnthropicFoundryChatGenerator(resource="my-resource")
        assert not component._is_warmed_up
        completion = Message(
            id="foo",
            content=[{"type": "text", "text": "Hello!"}],
            model="claude-sonnet-4-5",
            role="assistant",
            type="message",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        with patch(
            "anthropic.resources.messages.AsyncMessages.create", new_callable=AsyncMock, return_value=completion
        ):
            response = await component.run_async([ChatMessage.from_user("hi")])
        assert component._is_warmed_up
        assert "replies" in response

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") or not os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"),
        reason="Set ANTHROPIC_FOUNDRY_API_KEY and ANTHROPIC_FOUNDRY_RESOURCE env variables to run this test.",
    )
    @pytest.mark.integration
    async def test_live_run_async(self):
        component = AnthropicFoundryChatGenerator(
            resource=os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"),
            model="claude-sonnet-4-5",
        )
        results = await component.run_async(messages=[ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert message.meta["finish_reason"] == "end_turn"

    # Anthropic messages API is similar for AnthropicFoundry and Anthropic endpoint;
    # remaining tests are skipped as they are already tested in AnthropicChatGenerator.
