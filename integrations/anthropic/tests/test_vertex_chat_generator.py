import os
from unittest.mock import AsyncMock, patch

import anthropic
import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole

from haystack_integrations.components.generators.anthropic import AnthropicVertexChatGenerator


class TestUnit:
    def test_supported_models(self):
        """SUPPORTED_MODELS is a non-empty list of strings."""
        assert isinstance(AnthropicVertexChatGenerator.SUPPORTED_MODELS, list)
        assert len(AnthropicVertexChatGenerator.SUPPORTED_MODELS) > 0
        assert all(isinstance(m, str) for m in AnthropicVertexChatGenerator.SUPPORTED_MODELS)

    def test_init_default(self):
        component = AnthropicVertexChatGenerator(region="us-central1", project_id="test-project-id")
        assert component.region == "us-central1"
        assert component.project_id == "test-project-id"
        assert component.model == "claude-sonnet-4@20250514"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.ignore_tools_thinking_messages
        assert component.client is None
        assert component.async_client is None

    def test_init_with_parameters(self):
        component = AnthropicVertexChatGenerator(
            region="us-central1",
            project_id="test-project-id",
            model="claude-sonnet-4@20250514",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            ignore_tools_thinking_messages=False,
        )
        assert component.region == "us-central1"
        assert component.project_id == "test-project-id"
        assert component.model == "claude-sonnet-4@20250514"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.ignore_tools_thinking_messages is False

    def test_to_dict_default(self):
        component = AnthropicVertexChatGenerator(region="us-central1", project_id="test-project-id")
        data = component.to_dict()
        assert data == {
            "type": (
                "haystack_integrations.components.generators."
                "anthropic.chat.vertex_chat_generator.AnthropicVertexChatGenerator"
            ),
            "init_parameters": {
                "region": "us-central1",
                "project_id": "test-project-id",
                "model": "claude-sonnet-4@20250514",
                "streaming_callback": None,
                "generation_kwargs": {},
                "ignore_tools_thinking_messages": True,
                "tools": None,
                "anthropic_server_tools": None,
                "timeout": None,
                "max_retries": None,
            },
        }

    def test_to_dict_with_parameters(self):
        component = AnthropicVertexChatGenerator(
            region="us-central1",
            project_id="test-project-id",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            ignore_tools_thinking_messages=False,
            anthropic_server_tools=[{"type": "web_search_20250305", "name": "web_search"}],
            timeout=10.0,
            max_retries=1,
        )
        data = component.to_dict()
        assert data == {
            "type": (
                "haystack_integrations.components.generators."
                "anthropic.chat.vertex_chat_generator.AnthropicVertexChatGenerator"
            ),
            "init_parameters": {
                "region": "us-central1",
                "project_id": "test-project-id",
                "model": "claude-sonnet-4@20250514",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": False,
                "tools": None,
                "anthropic_server_tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "timeout": 10.0,
                "max_retries": 1,
            },
        }

    def test_from_dict(self):
        data = {
            "type": (
                "haystack_integrations.components.generators."
                "anthropic.chat.vertex_chat_generator.AnthropicVertexChatGenerator"
            ),
            "init_parameters": {
                "region": "us-central1",
                "project_id": "test-project-id",
                "model": "claude-sonnet-4@20250514",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
                "tools": None,
                "anthropic_server_tools": None,
                "timeout": None,
                "max_retries": None,
            },
        }
        component = AnthropicVertexChatGenerator.from_dict(data)
        assert component.model == "claude-sonnet-4@20250514"
        assert component.region == "us-central1"
        assert component.project_id == "test-project-id"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.ignore_tools_thinking_messages is True
        assert component.timeout is None
        assert component.max_retries is None
        assert component.anthropic_server_tools is None

    def test_run(self, chat_messages, mock_chat_completion):
        component = AnthropicVertexChatGenerator(region="us-central1", project_id="test-project-id")
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion):
        component = AnthropicVertexChatGenerator(
            region="us-central1", project_id="test-project-id", generation_kwargs={"max_tokens": 10, "temperature": 0.5}
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


class TestComponentLifecycle:
    @patch("haystack_integrations.components.generators.anthropic.chat.vertex_chat_generator.AnthropicVertex")
    def test_sync_lifecycle(self, mock_client_cls):
        component = AnthropicVertexChatGenerator(region="us-central1", project_id="test-project-id")
        client = mock_client_cls.return_value

        component.warm_up()
        assert component.client is client
        assert component.async_client is None
        component.close()
        client.close.assert_called_once_with()
        assert component.client is None
        component.warm_up()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.generators.anthropic.chat.vertex_chat_generator.AsyncAnthropicVertex")
    async def test_async_lifecycle(self, mock_client_cls):
        component = AnthropicVertexChatGenerator(region="us-central1", project_id="test-project-id")
        client = mock_client_cls.return_value
        client.close = AsyncMock()

        await component.warm_up_async()
        assert component.async_client is client
        assert component.client is None
        await component.close_async()
        client.close.assert_awaited_once_with()
        assert component.async_client is None
        await component.warm_up_async()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.generators.anthropic.chat.vertex_chat_generator.AnthropicVertex")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        component = AnthropicVertexChatGenerator(
            region="us-central1", project_id="test-project-id", timeout=10.0, max_retries=1
        )
        component.warm_up()
        component.warm_up()
        mock_client_cls.assert_called_once_with(
            region="us-central1", project_id="test-project-id", timeout=10.0, max_retries=1
        )

    @patch("haystack_integrations.components.generators.anthropic.chat.vertex_chat_generator.AsyncAnthropicVertex")
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        component = AnthropicVertexChatGenerator(region="us-central1", project_id="test-project-id")
        await component.warm_up_async()
        await component.warm_up_async()
        mock_client_cls.assert_called_once_with(region="us-central1", project_id="test-project-id")


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("REGION", None) and os.environ.get("PROJECT_ID", None)),
    reason="Authenticate with GCP and set env variables REGION and PROJECT_ID to run this test.",
)
class TestIntegration:
    def test_live_run_wrong_model(self, chat_messages):
        component = AnthropicVertexChatGenerator(
            model="something-obviously-wrong", region=os.environ.get("REGION"), project_id=os.environ.get("PROJECT_ID")
        )
        with pytest.raises(anthropic.NotFoundError):
            component.run(chat_messages)

    def test_default_inference_params(self, chat_messages):
        client = AnthropicVertexChatGenerator(
            region=os.environ.get("REGION"), project_id=os.environ.get("PROJECT_ID"), model="claude-sonnet-4@20250514"
        )
        response = client.run(chat_messages)

        assert "replies" in response, "Response does not contain 'replies' key"
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply has no text"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert "paris" in first_reply.text.lower(), "First reply does not contain 'paris'"
        assert first_reply.meta, "First reply has no metadata"

    async def test_live_run_async(self):
        """
        Integration test that the async run method of AnthropicVertexChatGenerator works correctly
        """
        component = AnthropicVertexChatGenerator(
            region=os.environ.get("REGION"),
            project_id=os.environ.get("PROJECT_ID"),
            model="claude-sonnet-4@20250514",
        )
        results = await component.run_async(messages=[ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "claude-sonnet-4-5" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
