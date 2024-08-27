from unittest.mock import Mock, patch

import pytest
from capabilities import MODEL_CAPABILITIES, ModelCapability
from converse_generator import AmazonBedrockConfigurationError, AmazonBedrockConverseGenerator
from haystack.dataclasses import ChatMessage
from utils import ContentBlock, ConverseMessage, ConverseRole, ToolConfig


@pytest.fixture
def generator():
    model = "anthropic.claude-3-haiku-20240307-v1:0"
    return AmazonBedrockConverseGenerator(model=model, streaming_callback=print)


def test_init(generator):
    assert generator.model == "anthropic.claude-3-haiku-20240307-v1:0"
    assert generator.streaming_callback == print
    assert generator.client is not None


def test_get_model_capabilities(generator):
    capabilities = generator._get_model_capabilities(generator.model)
    expected_capabilities = MODEL_CAPABILITIES["anthropic.claude-3.*"]
    assert capabilities == expected_capabilities


def test_get_model_capabilities_unsupported_model(generator):
    with pytest.raises(ValueError):
        generator._get_model_capabilities("unsupported_model")


@patch("converse_generator.get_aws_session")
def test_init_aws_error(mock_get_aws_session):
    mock_get_aws_session.side_effect = Exception("AWS Error")
    with pytest.raises(AmazonBedrockConfigurationError):
        AmazonBedrockConverseGenerator(model="anthropic.claude-3-haiku-20240307-v1:0")


@patch.object(AmazonBedrockConverseGenerator, "client")
def test_run_streaming(mock_client, generator):
    mock_stream = Mock()
    mock_client.converse_stream.return_value = {"stream": mock_stream}

    messages = [ConverseMessage.from_user(["Hello"])]
    streaming_callback = Mock()

    result = generator.run(messages, streaming_callback=streaming_callback)

    mock_client.converse_stream.assert_called_once()
    assert "message" in result
    assert "usage" in result
    assert "metrics" in result
    assert "guardrail_trace" in result
    assert "stop_reason" in result


@patch.object(AmazonBedrockConverseGenerator, "client")
def test_run_non_streaming(mock_client, generator):
    mock_response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "Hello, how can I help you?"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 20},
        "metrics": {"firstByteLatency": 0.5},
    }
    mock_client.converse.return_value = mock_response

    messages = [ConverseMessage.from_user(["Hello"])]

    result = generator.run(messages)

    mock_client.converse.assert_called_once()
    assert isinstance(result["message"], ConverseMessage)
    assert result["usage"] == mock_response["usage"]
    assert result["metrics"] == mock_response["metrics"]


def test_run_invalid_messages(generator):
    with pytest.raises(ValueError):
        generator.run(["invalid message"])


def test_to_dict(generator):
    serialized = generator.to_dict()
    assert "model" in serialized
    assert serialized["model"] == generator.model


def test_from_dict():
    data = {
        "init_parameters": {"model": "anthropic.claude-3-haiku-20240307-v1:0", "streaming_callback": "builtins.print"}
    }
    deserialized = AmazonBedrockConverseGenerator.from_dict(data)
    assert deserialized.model == "anthropic.claude-3-haiku-20240307-v1:0"
    assert deserialized.streaming_callback == print
