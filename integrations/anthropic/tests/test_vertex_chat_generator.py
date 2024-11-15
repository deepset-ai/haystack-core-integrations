import os

import anthropic
import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole

from haystack_integrations.components.generators.anthropic import AnthropicVertexChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("\\nYou are a helpful assistant, be super brief in your responses."),
        ChatMessage.from_user("What's the capital of France?"),
    ]


class TestAnthropicVertexChatGenerator:
    def test_init_default(self):
        component = AnthropicVertexChatGenerator(region="us-central1", project_id="test-project-id")
        assert component.region == "us-central1"
        assert component.project_id == "test-project-id"
        assert component.model == "claude-3-5-sonnet@20240620"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.ignore_tools_thinking_messages

    def test_init_with_parameters(self):
        component = AnthropicVertexChatGenerator(
            region="us-central1",
            project_id="test-project-id",
            model="claude-3-5-sonnet@20240620",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            ignore_tools_thinking_messages=False,
        )
        assert component.region == "us-central1"
        assert component.project_id == "test-project-id"
        assert component.model == "claude-3-5-sonnet@20240620"
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
                "model": "claude-3-5-sonnet@20240620",
                "streaming_callback": None,
                "generation_kwargs": {},
                "ignore_tools_thinking_messages": True,
            },
        }

    def test_to_dict_with_parameters(self):
        component = AnthropicVertexChatGenerator(
            region="us-central1",
            project_id="test-project-id",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
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
                "model": "claude-3-5-sonnet@20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self):
        component = AnthropicVertexChatGenerator(
            region="us-central1",
            project_id="test-project-id",
            model="claude-3-5-sonnet@20240620",
            streaming_callback=lambda x: x,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
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
                "model": "claude-3-5-sonnet@20240620",
                "streaming_callback": "tests.test_vertex_chat_generator.<lambda>",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
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
                "model": "claude-3-5-sonnet@20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "ignore_tools_thinking_messages": True,
            },
        }
        component = AnthropicVertexChatGenerator.from_dict(data)
        assert component.model == "claude-3-5-sonnet@20240620"
        assert component.region == "us-central1"
        assert component.project_id == "test-project-id"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

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

    @pytest.mark.skipif(
        not (os.environ.get("REGION", None) or os.environ.get("PROJECT_ID", None)),
        reason="Authenticate with GCP and set env variables REGION and PROJECT_ID to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = AnthropicVertexChatGenerator(
            model="something-obviously-wrong", region=os.environ.get("REGION"), project_id=os.environ.get("PROJECT_ID")
        )
        with pytest.raises(anthropic.NotFoundError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not (os.environ.get("REGION", None) or os.environ.get("PROJECT_ID", None)),
        reason="Authenticate with GCP and set env variables REGION and PROJECT_ID to run this test.",
    )
    @pytest.mark.integration
    def test_default_inference_params(self, chat_messages):
        client = AnthropicVertexChatGenerator(
            region=os.environ.get("REGION"), project_id=os.environ.get("PROJECT_ID"), model="claude-3-sonnet@20240229"
        )
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

    # Anthropic messages API is similar for AnthropicVertex and Anthropic endpoint,
    # remaining tests are skipped for AnthropicVertexChatGenerator as they are already tested in AnthropicChatGenerator.
