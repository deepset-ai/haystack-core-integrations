import os
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from haystack.utils.auth import Secret

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.token_counters.amazon_bedrock import AmazonBedrockTokenCounter

CLASS_TYPE = "haystack_integrations.token_counters.amazon_bedrock.token_counter.AmazonBedrockTokenCounter"
MODEL = "anthropic.claude-sonnet-4-20250514-v1:0"

_GET_AWS_SESSION = "haystack_integrations.token_counters.amazon_bedrock.token_counter.get_aws_session"


def _weather_tool() -> Tool:
    return Tool(
        name="weather",
        description="Get the weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=lambda city: f"Weather in {city}: sunny",
    )


class TestAmazonBedrockTokenCounter:
    def test_init(self):
        counter = AmazonBedrockTokenCounter(model=MODEL)
        assert counter.model == MODEL
        # the client is created lazily, on warm_up
        assert counter.client is None

    def test_empty_model(self):
        with pytest.raises(ValueError, match="cannot be None or empty string"):
            AmazonBedrockTokenCounter(model="")

    def test_empty_input_does_not_initialize_the_client(self):
        counter = AmazonBedrockTokenCounter(model=MODEL)
        assert counter.count([]) == 0
        assert counter.client is None

    def test_warm_up_initializes_the_client_once(self, mock_boto3_session):
        counter = AmazonBedrockTokenCounter(model=MODEL)
        counter.warm_up()
        counter.warm_up()
        mock_boto3_session.return_value.client.assert_called_once()
        assert mock_boto3_session.return_value.client.call_args[0][0] == "bedrock-runtime"

    def test_warm_up_connection_error(self):
        counter = AmazonBedrockTokenCounter(model=MODEL)
        with patch(_GET_AWS_SESSION, side_effect=Exception("boom")):
            with pytest.raises(AmazonBedrockConfigurationError):
                counter.warm_up()

    def test_count(self, mock_boto3_session):
        counter = AmazonBedrockTokenCounter(model=MODEL)
        counter.warm_up()
        counter.client.count_tokens.return_value = {"inputTokens": 42}

        messages = [ChatMessage.from_system("You are a helpful assistant."), ChatMessage.from_user("Hello!")]
        assert counter.count(messages) == 42

        kwargs = counter.client.count_tokens.call_args.kwargs
        assert kwargs["modelId"] == MODEL
        converse = kwargs["input"]["converse"]
        assert converse["system"] == [{"text": "You are a helpful assistant."}]
        assert converse["messages"][0]["role"] == "user"
        assert "toolConfig" not in converse

    def test_count_with_tools(self, mock_boto3_session):
        counter = AmazonBedrockTokenCounter(model=MODEL)
        counter.warm_up()
        counter.client.count_tokens.return_value = {"inputTokens": 100}

        result = counter.count([ChatMessage.from_user("What is the weather?")], tools=[_weather_tool()])
        assert result == 100

        converse = counter.client.count_tokens.call_args.kwargs["input"]["converse"]
        assert converse["toolConfig"]["tools"][0]["toolSpec"]["name"] == "weather"

    def test_count_raises_inference_error(self, mock_boto3_session):
        counter = AmazonBedrockTokenCounter(model=MODEL)
        counter.warm_up()
        counter.client.count_tokens.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "boom"}}, "CountTokens"
        )
        with pytest.raises(AmazonBedrockInferenceError, match="Could not count tokens"):
            counter.count([ChatMessage.from_user("Hello!")])

    def test_close_releases_the_client(self, mock_boto3_session):
        counter = AmazonBedrockTokenCounter(model=MODEL)
        counter.warm_up()
        client = counter.client
        counter.close()
        client.close.assert_called_once_with()
        assert counter.client is None
        counter.close()  # closing again is a no-op

    def test_to_dict(self):
        counter = AmazonBedrockTokenCounter(model=MODEL, boto3_config={"read_timeout": 1000})
        assert counter.to_dict() == {
            "type": CLASS_TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": MODEL,
                "boto3_config": {"read_timeout": 1000},
            },
        }

    def test_from_dict(self):
        counter = AmazonBedrockTokenCounter.from_dict(
            {
                "type": CLASS_TYPE,
                "init_parameters": {
                    "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                    "aws_secret_access_key": {
                        "type": "env_var",
                        "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                        "strict": False,
                    },
                    "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                    "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                    "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                    "model": MODEL,
                    "boto3_config": {"read_timeout": 1000},
                },
            }
        )
        assert counter.model == MODEL
        assert counter.boto3_config == {"read_timeout": 1000}


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("AWS_REGION"),
    reason="AWS_REGION must be set (with AWS credentials available) for a live Bedrock call",
)
class TestAmazonBedrockTokenCounterInference:
    def test_count_complex(self):
        """Count tokens for a conversation with system, user, assistant and tool result messages, plus tools."""
        # Requires a model that supports Bedrock's CountTokens API (e.g. a current Claude model) and ambient AWS
        # credentials (a profile, static keys, or a Bedrock bearer token).
        tool_call = ToolCall(tool_name="weather", arguments={"city": "Paris"}, id="tooluse_01")
        messages = [
            ChatMessage.from_system("You are a helpful weather assistant."),
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant("Let me check the weather for you.", tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="Weather in Paris: sunny", origin=tool_call),
            ChatMessage.from_assistant("The weather in Paris is sunny."),
        ]
        counter = AmazonBedrockTokenCounter(
            model="anthropic.claude-sonnet-4-20250514-v1:0",
            aws_region_name=Secret.from_token(os.environ["AWS_REGION"]),
        )
        count = counter.count(messages, tools=[_weather_tool()])
        assert isinstance(count, int)
        assert count > 20
