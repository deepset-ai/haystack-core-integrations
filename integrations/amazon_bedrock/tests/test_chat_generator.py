import json
from typing import Any, Dict, Optional

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

KLASS = "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
MODELS_TO_TEST = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "cohere.command-r-plus-v1:0",
    "mistral.mistral-large-2402-v1:0",
]
MODELS_TO_TEST_WITH_TOOLS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "cohere.command-r-plus-v1:0",
    "mistral.mistral-large-2402-v1:0",
]

# so far we've discovered these models support streaming and tool use
STREAMING_TOOL_MODELS = ["anthropic.claude-3-5-sonnet-20240620-v1:0", "cohere.command-r-plus-v1:0"]


@pytest.fixture
def chat_messages():
    messages = [
        ChatMessage.from_system("\\nYou are a helpful assistant, be super brief in your responses."),
        ChatMessage.from_user("What's the capital of France?"),
    ]
    return messages


@pytest.mark.parametrize(
    "boto3_config",
    [
        None,
        {
            "read_timeout": 1000,
        },
    ],
)
def test_to_dict(mock_boto3_session, boto3_config):
    """
    Test that the to_dict method returns the correct dictionary without aws credentials
    """
    generator = AmazonBedrockChatGenerator(
        model="cohere.command-r-plus-v1:0",
        generation_kwargs={"temperature": 0.7},
        streaming_callback=print_streaming_chunk,
        boto3_config=boto3_config,
    )
    expected_dict = {
        "type": KLASS,
        "init_parameters": {
            "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
            "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
            "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
            "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
            "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
            "model": "cohere.command-r-plus-v1:0",
            "generation_kwargs": {"temperature": 0.7},
            "stop_words": [],
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "boto3_config": boto3_config,
        },
    }

    assert generator.to_dict() == expected_dict


@pytest.mark.parametrize(
    "boto3_config",
    [
        None,
        {
            "read_timeout": 1000,
        },
    ],
)
def test_from_dict(mock_boto3_session: Any, boto3_config: Optional[Dict[str, Any]]):
    """
    Test that the from_dict method returns the correct object
    """
    generator = AmazonBedrockChatGenerator.from_dict(
        {
            "type": KLASS,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "generation_kwargs": {"temperature": 0.7},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "boto3_config": boto3_config,
            },
        }
    )
    assert generator.model == "anthropic.claude-3-5-sonnet-20240620-v1:0"
    assert generator.streaming_callback == print_streaming_chunk
    assert generator.boto3_config == boto3_config


def test_default_constructor(mock_boto3_session, set_env_variables):
    """
    Test that the default constructor sets the correct values
    """

    layer = AmazonBedrockChatGenerator(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    assert layer.model == "anthropic.claude-3-5-sonnet-20240620-v1:0"

    # assert mocked boto3 client called exactly once
    mock_boto3_session.assert_called_once()

    # assert mocked boto3 client was called with the correct parameters
    mock_boto3_session.assert_called_with(
        aws_access_key_id="some_fake_id",
        aws_secret_access_key="some_fake_key",
        aws_session_token="some_fake_token",
        profile_name="some_fake_profile",
        region_name="fake_region",
    )


def test_constructor_with_generation_kwargs(mock_boto3_session):
    """
    Test that model_kwargs are correctly set in the constructor
    """
    generation_kwargs = {"temperature": 0.7}

    layer = AmazonBedrockChatGenerator(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0", generation_kwargs=generation_kwargs
    )
    assert layer.generation_kwargs == generation_kwargs


def test_constructor_with_empty_model():
    """
    Test that the constructor raises an error when the model is empty
    """
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        AmazonBedrockChatGenerator(model="")


class TestAmazonBedrockChatGeneratorInference:

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    @pytest.mark.integration
    def test_default_inference_params(self, model_name, chat_messages):
        client = AmazonBedrockChatGenerator(model=model_name)
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

        if first_reply.meta and "usage" in first_reply.meta:
            assert "prompt_tokens" in first_reply.meta["usage"]
            assert "completion_tokens" in first_reply.meta["usage"]

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    @pytest.mark.integration
    def test_default_inference_with_streaming(self, model_name, chat_messages):
        streaming_callback_called = False
        paris_found_in_response = False

        def streaming_callback(chunk: StreamingChunk):
            nonlocal streaming_callback_called, paris_found_in_response
            streaming_callback_called = True
            assert isinstance(chunk, StreamingChunk)
            assert chunk.content is not None
            if not paris_found_in_response:
                paris_found_in_response = "paris" in chunk.content.lower()

        client = AmazonBedrockChatGenerator(model=model_name, streaming_callback=streaming_callback)
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

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_TOOLS)
    @pytest.mark.integration
    def test_tools_use(self, model_name):
        """
        Test function calling with AWS Bedrock Anthropic adapter
        """
        # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolConfiguration.html
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "top_song",
                        "description": "Get the most popular song played on a radio station.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station "
                                        "for which you want the most popular song. "
                                        "Example calls signs are WZPZ and WKRP.",
                                    }
                                },
                                "required": ["sign"],
                            }
                        },
                    }
                }
            ],
            # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            "toolChoice": {"auto": {}},
        }

        messages = []
        messages.append(ChatMessage.from_user("What is the most popular song on WZPZ?"))
        client = AmazonBedrockChatGenerator(model=model_name)
        response = client.run(messages=messages, generation_kwargs={"toolConfig": tool_config})
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.content, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert first_reply.meta, "First reply has no metadata"

        # Some models return thinking message as first and the second one as the tool call
        if len(replies) > 1:
            second_reply = replies[1]
            assert isinstance(second_reply, ChatMessage), "Second reply is not a ChatMessage instance"
            assert second_reply.content, "Second reply has no content"
            assert ChatMessage.is_from(second_reply, ChatRole.ASSISTANT), "Second reply is not from the assistant"
            tool_call = json.loads(second_reply.content)
            assert "toolUseId" in tool_call, "Tool call does not contain 'toolUseId' key"
            assert tool_call["name"] == "top_song", f"Tool call {tool_call} does not contain the correct 'name' value"
            assert "input" in tool_call, f"Tool call {tool_call} does not contain 'input' key"
            assert (
                tool_call["input"]["sign"] == "WZPZ"
            ), f"Tool call {tool_call} does not contain the correct 'input' value"
        else:
            # case where the model returns the tool call as the first message
            # double check that the tool call is correct
            tool_call = json.loads(first_reply.content)
            assert "toolUseId" in tool_call, "Tool call does not contain 'toolUseId' key"
            assert tool_call["name"] == "top_song", f"Tool call {tool_call} does not contain the correct 'name' value"
            assert "input" in tool_call, f"Tool call {tool_call} does not contain 'input' key"
            assert (
                tool_call["input"]["sign"] == "WZPZ"
            ), f"Tool call {tool_call} does not contain the correct 'input' value"

    @pytest.mark.parametrize("model_name", STREAMING_TOOL_MODELS)
    @pytest.mark.integration
    def test_tools_use_with_streaming(self, model_name):
        """
        Test function calling with AWS Bedrock Anthropic adapter
        """
        # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolConfiguration.html
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "top_song",
                        "description": "Get the most popular song played on a radio station.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station "
                                        "for which you want the most popular song. Example "
                                        "calls signs are WZPZ and WKRP.",
                                    }
                                },
                                "required": ["sign"],
                            }
                        },
                    }
                }
            ],
            # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            "toolChoice": {"auto": {}},
        }

        messages = []
        messages.append(ChatMessage.from_user("What is the most popular song on WZPZ?"))
        client = AmazonBedrockChatGenerator(model=model_name, streaming_callback=print_streaming_chunk)
        response = client.run(messages=messages, generation_kwargs={"toolConfig": tool_config})
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.content, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert first_reply.meta, "First reply has no metadata"

        # Some models return thinking message as first and the second one as the tool call
        if len(replies) > 1:
            second_reply = replies[1]
            assert isinstance(second_reply, ChatMessage), "Second reply is not a ChatMessage instance"
            assert second_reply.content, "Second reply has no content"
            assert ChatMessage.is_from(second_reply, ChatRole.ASSISTANT), "Second reply is not from the assistant"
            tool_call = json.loads(second_reply.content)
            assert "toolUseId" in tool_call, "Tool call does not contain 'toolUseId' key"
            assert tool_call["name"] == "top_song", f"Tool call {tool_call} does not contain the correct 'name' value"
            assert "input" in tool_call, f"Tool call {tool_call} does not contain 'input' key"
            assert (
                tool_call["input"]["sign"] == "WZPZ"
            ), f"Tool call {tool_call} does not contain the correct 'input' value"
        else:
            # case where the model returns the tool call as the first message
            # double check that the tool call is correct
            tool_call = json.loads(first_reply.content)
            assert "toolUseId" in tool_call, "Tool call does not contain 'toolUseId' key"
            assert tool_call["name"] == "top_song", f"Tool call {tool_call} does not contain the correct 'name' value"
            assert "input" in tool_call, f"Tool call {tool_call} does not contain 'input' key"
            assert (
                tool_call["input"]["sign"] == "WZPZ"
            ), f"Tool call {tool_call} does not contain the correct 'input' value"
