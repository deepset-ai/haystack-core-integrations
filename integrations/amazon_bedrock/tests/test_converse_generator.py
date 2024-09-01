from unittest.mock import Mock, patch, MagicMock
import pytest

from haystack_integrations.components.generators.amazon_bedrock import (
    ConverseMessage,
    AmazonBedrockConverseGenerator,
    MODEL_CAPABILITIES,
    ToolConfig,
)
from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
)


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location"""
    # This is a mock function, replace with actual API call
    return f"The weather in {location} is 22 degrees {unit}."


def get_current_time(timezone: str) -> str:
    """Get the current time in a given timezone"""
    # This is a mock function, replace with actual time lookup
    return f"The current time in {timezone} is 14:30."


def test_to_dict(mock_boto3_session):
    """
    Test that the to_dict method returns the correct dictionary without aws credentials
    """
    tool_config = ToolConfig.from_functions([get_current_weather, get_current_time])

    generator = AmazonBedrockConverseGenerator(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        inference_config={
            "temperature": 0.1,
            "maxTokens": 256,
            "topP": 0.1,
            "stopSequences": ["\\n"],
        },
        tool_config=tool_config,
    )

    expected_dict = {
        "type": "haystack_integrations.components.generators.amazon_bedrock.converse.converse_generator.AmazonBedrockConverseGenerator",
        "init_parameters": {
            "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
            "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
            "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
            "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
            "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "inference_config": {
                "temperature": 0.1,
                "maxTokens": 256,
                "topP": 0.1,
                "stopSequences": ["\\n"],
            },
            "tool_config": tool_config.to_dict(),
            "streaming_callback": None,
            "system_prompt": None,
        },
    }

    assert generator.to_dict() == expected_dict


def test_from_dict(mock_boto3_session):
    """
    Test that the from_dict method returns the correct object
    """
    tool_config = ToolConfig.from_functions([get_current_weather, get_current_time])

    generator = AmazonBedrockConverseGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.amazon_bedrock.converse.converse_generator.AmazonBedrockConverseGenerator",
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "inference_config": {
                    "temperature": 0.1,
                    "maxTokens": 256,
                    "topP": 0.1,
                    "stopSequences": ["\\n"],
                },
                "tool_config": tool_config.to_dict(),
                "streaming_callback": None,
                "system_prompt": None,
            },
        }
    )

    assert generator.inference_config["temperature"] == 0.1
    assert generator.inference_config["maxTokens"] == 256
    assert generator.inference_config["topP"] == 0.1
    assert generator.inference_config["stopSequences"] == ["\\n"]
    assert generator.tool_config.to_dict() == tool_config.to_dict()
    assert generator.model == "anthropic.claude-3-5-sonnet-20240620-v1:0"
