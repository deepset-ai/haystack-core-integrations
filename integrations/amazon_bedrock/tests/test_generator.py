from typing import Any, Dict, Optional, Type
from unittest.mock import MagicMock, call

import pytest
from haystack.dataclasses import StreamingChunk

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
)
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator
from haystack_integrations.components.generators.amazon_bedrock.adapters import (
    AI21LabsJurassic2Adapter,
    AmazonTitanAdapter,
    AnthropicClaudeAdapter,
    BedrockModelAdapter,
    CohereCommandAdapter,
    CohereCommandRAdapter,
    MetaLlamaAdapter,
    MistralAdapter,
)


@pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 1000}])
def test_to_dict(mock_boto3_session: Any, boto3_config: Optional[Dict[str, Any]]):
    """
    Test that the to_dict method returns the correct dictionary without aws credentials
    """
    generator = AmazonBedrockGenerator(
        model="anthropic.claude-v2", max_length=99, temperature=10, boto3_config=boto3_config
    )

    expected_dict = {
        "type": "haystack_integrations.components.generators.amazon_bedrock.generator.AmazonBedrockGenerator",
        "init_parameters": {
            "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
            "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
            "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
            "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
            "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
            "model": "anthropic.claude-v2",
            "max_length": 99,
            "temperature": 10,
            "streaming_callback": None,
            "boto3_config": boto3_config,
            "model_family": None,
        },
    }

    assert generator.to_dict() == expected_dict


@pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 1000}])
def test_from_dict(mock_boto3_session: Any, boto3_config: Optional[Dict[str, Any]]):
    """
    Test that the from_dict method returns the correct object
    """
    generator = AmazonBedrockGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.amazon_bedrock.generator.AmazonBedrockGenerator",
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "anthropic.claude-v2",
                "max_length": 99,
                "boto3_config": boto3_config,
                "model_family": "anthropic.claude",
            },
        }
    )

    assert generator.max_length == 99
    assert generator.model == "anthropic.claude-v2"
    assert generator.boto3_config == boto3_config
    assert generator.model_family == "anthropic.claude"


def test_default_constructor(mock_boto3_session, set_env_variables):
    """
    Test that the default constructor sets the correct values
    """

    layer = AmazonBedrockGenerator(
        model="anthropic.claude-v2",
        max_length=99,
    )

    assert layer.max_length == 99
    assert layer.model == "anthropic.claude-v2"

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


def test_constructor_with_model_kwargs(mock_boto3_session):
    """
    Test that model_kwargs are correctly set in the constructor
    """
    model_kwargs = {"temperature": 0.7}

    layer = AmazonBedrockGenerator(model="anthropic.claude-v2", **model_kwargs)
    assert "temperature" in layer.model_adapter.model_kwargs
    assert layer.model_adapter.model_kwargs["temperature"] == 0.7


def test_constructor_with_empty_model():
    """
    Test that the constructor raises an error when the model is empty
    """
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        AmazonBedrockGenerator(model="")


@pytest.mark.parametrize(
    "model, expected_model_adapter",
    [
        ("anthropic.claude-v1", AnthropicClaudeAdapter),
        ("anthropic.claude-v2", AnthropicClaudeAdapter),
        ("eu.anthropic.claude-v1", AnthropicClaudeAdapter),  # cross-region inference
        ("us.anthropic.claude-v2", AnthropicClaudeAdapter),  # cross-region inference
        ("anthropic.claude-instant-v1", AnthropicClaudeAdapter),
        ("anthropic.claude-super-v5", AnthropicClaudeAdapter),  # artificial
        ("cohere.command-text-v14", CohereCommandAdapter),
        ("cohere.command-light-text-v14", CohereCommandAdapter),
        ("cohere.command-text-v21", CohereCommandAdapter),  # artificial
        ("cohere.command-r-v1:0", CohereCommandRAdapter),
        ("cohere.command-r-plus-v1:0", CohereCommandRAdapter),
        ("cohere.command-r-v8:9", CohereCommandRAdapter),  # artificial
        ("ai21.j2-mid-v1", AI21LabsJurassic2Adapter),
        ("ai21.j2-ultra-v1", AI21LabsJurassic2Adapter),
        ("ai21.j2-mega-v5", AI21LabsJurassic2Adapter),  # artificial
        ("amazon.titan-text-lite-v1", AmazonTitanAdapter),
        ("amazon.titan-text-express-v1", AmazonTitanAdapter),
        ("us.amazon.titan-text-express-v1", AmazonTitanAdapter),  # cross-region inference
        ("amazon.titan-text-agile-v1", AmazonTitanAdapter),
        ("amazon.titan-text-lightning-v8", AmazonTitanAdapter),  # artificial
        ("meta.llama2-13b-chat-v1", MetaLlamaAdapter),
        ("meta.llama2-70b-chat-v1", MetaLlamaAdapter),
        ("eu.meta.llama2-13b-chat-v1", MetaLlamaAdapter),  # cross-region inference
        ("us.meta.llama2-70b-chat-v1", MetaLlamaAdapter),  # cross-region inference
        ("meta.llama2-130b-v5", MetaLlamaAdapter),  # artificial
        ("meta.llama3-8b-instruct-v1:0", MetaLlamaAdapter),
        ("meta.llama3-70b-instruct-v1:0", MetaLlamaAdapter),
        ("meta.llama3-130b-instruct-v5:9", MetaLlamaAdapter),  # artificial
        ("mistral.mistral-7b-instruct-v0:2", MistralAdapter),
        ("mistral.mixtral-8x7b-instruct-v0:1", MistralAdapter),
        ("mistral.mistral-large-2402-v1:0", MistralAdapter),
        ("eu.mistral.mixtral-8x7b-instruct-v0:1", MistralAdapter),  # cross-region inference
        ("us.mistral.mistral-large-2402-v1:0", MistralAdapter),  # cross-region inference
        ("mistral.mistral-medium-v8:0", MistralAdapter),  # artificial
    ],
)
def test_get_model_adapter(model: str, expected_model_adapter: Optional[Type[BedrockModelAdapter]]):
    """
    Test that the correct model adapter is returned for a given model
    """
    model_adapter = AmazonBedrockGenerator.get_model_adapter(model=model)
    assert model_adapter == expected_model_adapter


@pytest.mark.parametrize(
    "model_family, expected_model_adapter",
    [
        ("anthropic.claude", AnthropicClaudeAdapter),
        ("cohere.command", CohereCommandAdapter),
        ("cohere.command-r", CohereCommandRAdapter),
        ("ai21.j2", AI21LabsJurassic2Adapter),
        ("amazon.titan-text", AmazonTitanAdapter),
        ("meta.llama", MetaLlamaAdapter),
        ("mistral", MistralAdapter),
    ],
)
def test_get_model_adapter_with_model_family(
    model_family: str, expected_model_adapter: Optional[Type[BedrockModelAdapter]]
):
    """
    Test that the correct model adapter is returned for a given model model_family
    """
    model_adapter = AmazonBedrockGenerator.get_model_adapter(model="arn:123435423", model_family=model_family)
    assert model_adapter == expected_model_adapter


def test_get_model_adapter_with_invalid_model_family():
    """
    Test that an error is raised when an invalid model_family is provided
    """
    with pytest.raises(AmazonBedrockConfigurationError):
        AmazonBedrockGenerator.get_model_adapter(model="arn:123435423", model_family="invalid")


def test_get_model_adapter_auto_detect_family_fails():
    """
    Test that an error is raised when auto-detection of model_family fails
    """
    with pytest.raises(AmazonBedrockConfigurationError):
        AmazonBedrockGenerator.get_model_adapter(model="arn:123435423")


def test_get_model_adapter_model_family_over_auto_detection():
    """
    Test that the model_family is used over auto-detection
    """
    model_adapter = AmazonBedrockGenerator.get_model_adapter(
        model="cohere.command-text-v14", model_family="anthropic.claude"
    )
    assert model_adapter == AnthropicClaudeAdapter


class TestAnthropicClaudeAdapter:
    def test_default_init(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=100)
        assert adapter.use_messages_api is True

    def test_use_messages_api_false(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=100)
        assert adapter.use_messages_api is False


class TestAnthropicClaudeAdapterMessagesAPI:
    def test_prepare_body_with_default_params(self) -> None:
        layer = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 99,
            "anthropic_version": "bedrock-2023-05-31",
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        layer = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
            "stop_sequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
            "system": "system prompt",
            "anthropic_version": "custom_version",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            top_p=0.8,
            top_k=5,
            max_tokens=50,
            stop_sequences=["CUSTOM_STOP"],
            system="system prompt",
            anthropic_version="custom_version",
            unknown_arg="unknown_value",
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        layer = AnthropicClaudeAdapter(
            model_kwargs={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 5,
                "max_tokens": 50,
                "stop_sequences": ["CUSTOM_STOP"],
                "system": "system prompt",
                "anthropic_version": "custom_version",
                "unknown_arg": "unknown_value",
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
            "stop_sequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
            "system": "system prompt",
            "anthropic_version": "custom_version",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        layer = AnthropicClaudeAdapter(
            model_kwargs={
                "temperature": 0.6,
                "top_p": 0.7,
                "top_k": 4,
                "max_tokens": 49,
                "stop_sequences": ["CUSTOM_STOP_MODEL_KWARGS"],
                "system": "system prompt",
                "anthropic_version": "custom_version",
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
            "stop_sequences": ["CUSTOM_STOP_MODEL_KWARGS"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
            "system": "new system prompt",
            "anthropic_version": "new_custom_version",
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            top_p=0.8,
            top_k=5,
            max_tokens=50,
            system="new system prompt",
            anthropic_version="new_custom_version",
            thinking={"type": "enabled", "budget_tokens": 2048},
        )

        assert body == expected_body

    def test_get_responses(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        response_body = {"content": [{"text": "This is a single response."}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        response_body = {"content": [{"text": "\n\t This is a single response."}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_with_thinking(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        response_body = {
            "content": [
                {"thinking": "This is a thinking part.", "type": "thinking"},
                {"text": "This is a single response."},
            ]
        }
        expected_responses = ["<thinking>This is a thinking part.</thinking>\n\nThis is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_with_thinking_include_thinking_false(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={"include_thinking": False}, max_length=99)
        response_body = {
            "content": [
                {"thinking": "This is a thinking part.", "type": "thinking"},
                {"text": "This is a single response."},
            ]
        }
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_with_thinking_custom_thinking_tag(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={"thinking_tag": "custom"}, max_length=99)
        response_body = {
            "content": [
                {"thinking": "This is a thinking part.", "type": "thinking"},
                {"text": "This is a single response."},
            ]
        }
        expected_responses = ["<custom>This is a thinking part.</custom>\n\nThis is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_with_thinking_no_thinking_tag(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={"thinking_tag": None}, max_length=99)
        response_body = {
            "content": [
                {"thinking": "This is a thinking part.", "type": "thinking"},
                {"text": "This is a single response."},
            ]
        }
        expected_responses = ["This is a thinking part.\n\nThis is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_with_thinking_redacted_thinking_is_ignored(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        response_body = {
            "content": [
                {"thinking": "This is a thinking part.", "type": "thinking"},
                {"data": "[REDACTED]", "type": "redacted_thinking"},
                {"text": "This is a single response."},
            ]
        }
        expected_responses = ["<thinking>This is a thinking part.</thinking>\n\nThis is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "text"}, "index": 0}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " single"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " response."}}'}},
        ]

        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(StreamingChunk(content=" This", meta={"delta": {"text": " This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"text": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"text": " a"}})),
                call(StreamingChunk(content=" single", meta={"delta": {"text": " single"}})),
                call(StreamingChunk(content=" response.", meta={"delta": {"text": " response."}})),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_not_called()

    def test_get_stream_responses_with_thinking(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 0}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " thinking"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " part."}}'}},
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "text"}, "index": 1}'}},
            {"chunk": {"bytes": b'{"delta": {"text": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " single"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " response."}}'}},
        ]

        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["<thinking>This is a thinking part.</thinking>\n\nThis is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(
                    StreamingChunk(
                        content="<thinking>",
                        meta={"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 0},
                    )
                ),
                call(StreamingChunk(content="This", meta={"delta": {"thinking": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"thinking": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"thinking": " a"}})),
                call(StreamingChunk(content=" thinking", meta={"delta": {"thinking": " thinking"}})),
                call(StreamingChunk(content=" part.", meta={"delta": {"thinking": " part."}})),
                call(
                    StreamingChunk(
                        content="</thinking>\n\n",
                        meta={"type": "content_block_start", "content_block": {"type": "text"}, "index": 1},
                    )
                ),
                call(StreamingChunk(content="This", meta={"delta": {"text": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"text": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"text": " a"}})),
                call(StreamingChunk(content=" single", meta={"delta": {"text": " single"}})),
                call(StreamingChunk(content=" response.", meta={"delta": {"text": " response."}})),
            ]
        )

    def test_get_stream_responses_with_thinking_include_thinking_false(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 0}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " thinking"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " part."}}'}},
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "text"}, "index": 1}'}},
            {"chunk": {"bytes": b'{"delta": {"text": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " single"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " response."}}'}},
        ]

        adapter = AnthropicClaudeAdapter(model_kwargs={"include_thinking": False}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(StreamingChunk(content="This", meta={"delta": {"text": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"text": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"text": " a"}})),
                call(StreamingChunk(content=" single", meta={"delta": {"text": " single"}})),
                call(StreamingChunk(content=" response.", meta={"delta": {"text": " response."}})),
            ]
        )

    def test_get_stream_responses_with_thinking_custom_thinking_tag(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 0}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " thinking"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " part."}}'}},
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "text"}, "index": 1}'}},
            {"chunk": {"bytes": b'{"delta": {"text": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " single"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " response."}}'}},
        ]

        adapter = AnthropicClaudeAdapter(model_kwargs={"thinking_tag": "custom"}, max_length=99)
        expected_responses = ["<custom>This is a thinking part.</custom>\n\nThis is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(
                    StreamingChunk(
                        content="<custom>",
                        meta={"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 0},
                    )
                ),
                call(StreamingChunk(content="This", meta={"delta": {"thinking": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"thinking": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"thinking": " a"}})),
                call(StreamingChunk(content=" thinking", meta={"delta": {"thinking": " thinking"}})),
                call(StreamingChunk(content=" part.", meta={"delta": {"thinking": " part."}})),
                call(
                    StreamingChunk(
                        content="</custom>\n\n",
                        meta={"type": "content_block_start", "content_block": {"type": "text"}, "index": 1},
                    )
                ),
                call(StreamingChunk(content="This", meta={"delta": {"text": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"text": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"text": " a"}})),
                call(StreamingChunk(content=" single", meta={"delta": {"text": " single"}})),
                call(StreamingChunk(content=" response.", meta={"delta": {"text": " response."}})),
            ]
        )

    def test_get_stream_responses_with_thinking_no_thinking_tag(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 0}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " thinking"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " part."}}'}},
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "text"}, "index": 1}'}},
            {"chunk": {"bytes": b'{"delta": {"text": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " single"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " response."}}'}},
        ]

        adapter = AnthropicClaudeAdapter(model_kwargs={"thinking_tag": None}, max_length=99)
        expected_responses = ["This is a thinking part.\n\nThis is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(
                    StreamingChunk(
                        content="",
                        meta={"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 0},
                    )
                ),
                call(StreamingChunk(content="This", meta={"delta": {"thinking": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"thinking": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"thinking": " a"}})),
                call(StreamingChunk(content=" thinking", meta={"delta": {"thinking": " thinking"}})),
                call(StreamingChunk(content=" part.", meta={"delta": {"thinking": " part."}})),
                call(
                    StreamingChunk(
                        content="\n\n",
                        meta={"type": "content_block_start", "content_block": {"type": "text"}, "index": 1},
                    )
                ),
                call(StreamingChunk(content="This", meta={"delta": {"text": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"text": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"text": " a"}})),
                call(StreamingChunk(content=" single", meta={"delta": {"text": " single"}})),
                call(StreamingChunk(content=" response.", meta={"delta": {"text": " response."}})),
            ]
        )

    def test_get_stream_responses_with_thinking_redacted_thinking_is_ignored(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {
                "chunk": {
                    "bytes": (
                        b'{"type": "content_block_start", "content_block": {"type": "redacted_thinking"}, "index": 0}'
                    )
                }
            },
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 1}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " thinking"}}'}},
            {"chunk": {"bytes": b'{"delta": {"thinking": " part."}}'}},
            {"chunk": {"bytes": b'{"type": "content_block_start", "content_block": {"type": "text"}, "index": 2}'}},
            {"chunk": {"bytes": b'{"delta": {"text": "This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " single"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " response."}}'}},
        ]

        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["<thinking>This is a thinking part.</thinking>\n\nThis is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(
                    StreamingChunk(
                        content="<thinking>",
                        meta={"type": "content_block_start", "content_block": {"type": "thinking"}, "index": 1},
                    )
                ),
                call(StreamingChunk(content="This", meta={"delta": {"thinking": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"thinking": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"thinking": " a"}})),
                call(StreamingChunk(content=" thinking", meta={"delta": {"thinking": " thinking"}})),
                call(StreamingChunk(content=" part.", meta={"delta": {"thinking": " part."}})),
                call(
                    StreamingChunk(
                        content="</thinking>\n\n",
                        meta={"type": "content_block_start", "content_block": {"type": "text"}, "index": 2},
                    )
                ),
                call(StreamingChunk(content="This", meta={"delta": {"text": "This"}})),
                call(StreamingChunk(content=" is", meta={"delta": {"text": " is"}})),
                call(StreamingChunk(content=" a", meta={"delta": {"text": " a"}})),
                call(StreamingChunk(content=" single", meta={"delta": {"text": " single"}})),
                call(StreamingChunk(content=" response.", meta={"delta": {"text": " response."}})),
            ]
        )


class TestAnthropicClaudeAdapterNoMessagesAPI:
    def test_prepare_body_with_default_params(self) -> None:
        layer = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "\n\nHuman: Hello, how are you?\n\nAssistant:",
            "max_tokens_to_sample": 99,
            "stop_sequences": ["\n\nHuman:"],
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        layer = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "\n\nHuman: Hello, how are you?\n\nAssistant:",
            "max_tokens_to_sample": 50,
            "stop_sequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            top_p=0.8,
            top_k=5,
            max_tokens_to_sample=50,
            stop_sequences=["CUSTOM_STOP"],
            unknown_arg="unknown_value",
        )

        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        layer = AnthropicClaudeAdapter(
            model_kwargs={
                "use_messages_api": False,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 5,
                "max_tokens_to_sample": 50,
                "stop_sequences": ["CUSTOM_STOP"],
                "unknown_arg": "unknown_value",
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "\n\nHuman: Hello, how are you?\n\nAssistant:",
            "max_tokens_to_sample": 50,
            "stop_sequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        layer = AnthropicClaudeAdapter(
            model_kwargs={
                "use_messages_api": False,
                "temperature": 0.6,
                "top_p": 0.7,
                "top_k": 4,
                "max_tokens_to_sample": 49,
                "stop_sequences": ["CUSTOM_STOP_MODEL_KWARGS"],
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "\n\nHuman: Hello, how are you?\n\nAssistant:",
            "max_tokens_to_sample": 50,
            "stop_sequences": ["CUSTOM_STOP_MODEL_KWARGS"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
        }

        body = layer.prepare_body(prompt, temperature=0.7, top_p=0.8, top_k=5, max_tokens_to_sample=50)

        assert body == expected_body

    def test_get_responses(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=99)
        response_body = {"completion": "This is a single response."}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        adapter = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=99)
        response_body = {"completion": "\n\t This is a single response."}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"completion": " This"}'}},
            {"chunk": {"bytes": b'{"completion": " is"}'}},
            {"chunk": {"bytes": b'{"completion": " a"}'}},
            {"chunk": {"bytes": b'{"completion": " single"}'}},
            {"chunk": {"bytes": b'{"completion": " response."}'}},
        ]

        adapter = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(StreamingChunk(content=" This", meta={"completion": " This"})),
                call(StreamingChunk(content=" is", meta={"completion": " is"})),
                call(StreamingChunk(content=" a", meta={"completion": " a"})),
                call(StreamingChunk(content=" single", meta={"completion": " single"})),
                call(StreamingChunk(content=" response.", meta={"completion": " response."})),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        adapter = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_not_called()


class TestMistralAdapter:
    def test_prepare_body_with_default_params(self) -> None:
        layer = MistralAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {"prompt": "<s>[INST] Hello, how are you? [/INST]", "max_tokens": 99, "stop": []}

        body = layer.prepare_body(prompt)
        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        layer = MistralAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "<s>[INST] Hello, how are you? [/INST]",
            "max_tokens": 50,
            "stop": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            top_p=0.8,
            top_k=5,
            max_tokens=50,
            stop=["CUSTOM_STOP"],
            unknown_arg="unknown_value",
        )

        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        layer = MistralAdapter(
            model_kwargs={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 5,
                "max_tokens": 50,
                "stop": ["CUSTOM_STOP"],
                "unknown_arg": "unknown_value",
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "<s>[INST] Hello, how are you? [/INST]",
            "max_tokens": 50,
            "stop": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        layer = MistralAdapter(
            model_kwargs={
                "temperature": 0.6,
                "top_p": 0.7,
                "top_k": 4,
                "max_tokens": 49,
                "stop": ["CUSTOM_STOP_MODEL_KWARGS"],
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "<s>[INST] Hello, how are you? [/INST]",
            "max_tokens": 50,
            "stop": ["CUSTOM_STOP_MODEL_KWARGS"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
        }

        body = layer.prepare_body(prompt, temperature=0.7, top_p=0.8, top_k=5, max_tokens=50)

        assert body == expected_body

    def test_get_responses(self) -> None:
        adapter = MistralAdapter(model_kwargs={}, max_length=99)
        response_body = {"outputs": [{"text": "This is a single response."}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"outputs": [{"text": " This"}]}'}},
            {"chunk": {"bytes": b'{"outputs": [{"text": " is"}]}'}},
            {"chunk": {"bytes": b'{"outputs": [{"text": " a"}]}'}},
            {"chunk": {"bytes": b'{"outputs": [{"text": " single"}]}'}},
            {"chunk": {"bytes": b'{"outputs": [{"text": " response."}]}'}},
        ]

        adapter = MistralAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(StreamingChunk(content=" This", meta={"outputs": [{"text": " This"}]})),
                call(StreamingChunk(content=" is", meta={"outputs": [{"text": " is"}]})),
                call(StreamingChunk(content=" a", meta={"outputs": [{"text": " a"}]})),
                call(StreamingChunk(content=" single", meta={"outputs": [{"text": " single"}]})),
                call(StreamingChunk(content=" response.", meta={"outputs": [{"text": " response."}]})),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        streaming_callback_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = MistralAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_not_called()


class TestCohereCommandAdapter:
    def test_prepare_body_with_default_params(self) -> None:
        layer = CohereCommandAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {"prompt": "Hello, how are you?", "max_tokens": 99}

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        layer = CohereCommandAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "stop_sequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "p": 0.8,
            "k": 5,
            "return_likelihoods": "GENERATION",
            "stream": True,
            "logit_bias": {"token_id": 10.0},
            "num_generations": 1,
            "truncate": "START",
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            p=0.8,
            k=5,
            max_tokens=50,
            stop_sequences=["CUSTOM_STOP"],
            return_likelihoods="GENERATION",
            stream=True,
            logit_bias={"token_id": 10.0},
            num_generations=1,
            truncate="START",
            unknown_arg="unknown_value",
        )

        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        layer = CohereCommandAdapter(
            model_kwargs={
                "temperature": 0.7,
                "p": 0.8,
                "k": 5,
                "max_tokens": 50,
                "stop_sequences": ["CUSTOM_STOP"],
                "return_likelihoods": "GENERATION",
                "stream": True,
                "logit_bias": {"token_id": 10.0},
                "num_generations": 1,
                "truncate": "START",
                "unknown_arg": "unknown_value",
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "stop_sequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "p": 0.8,
            "k": 5,
            "return_likelihoods": "GENERATION",
            "stream": True,
            "logit_bias": {"token_id": 10.0},
            "num_generations": 1,
            "truncate": "START",
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        layer = CohereCommandAdapter(
            model_kwargs={
                "temperature": 0.6,
                "p": 0.7,
                "k": 4,
                "max_tokens": 49,
                "stop_sequences": ["CUSTOM_STOP_MODEL_KWARGS"],
                "return_likelihoods": "ALL",
                "stream": False,
                "logit_bias": {"token_id": 9.0},
                "num_generations": 2,
                "truncate": "NONE",
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "stop_sequences": ["CUSTOM_STOP_MODEL_KWARGS"],
            "temperature": 0.7,
            "p": 0.8,
            "k": 5,
            "return_likelihoods": "GENERATION",
            "stream": True,
            "logit_bias": {"token_id": 10.0},
            "num_generations": 1,
            "truncate": "START",
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            p=0.8,
            k=5,
            max_tokens=50,
            return_likelihoods="GENERATION",
            stream=True,
            logit_bias={"token_id": 10.0},
            num_generations=1,
            truncate="START",
        )

        assert body == expected_body

    def test_get_responses(self) -> None:
        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        response_body = {"generations": [{"text": "This is a single response."}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        response_body = {"generations": [{"text": "\n\t This is a single response."}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_multiple_responses(self) -> None:
        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        response_body = {
            "generations": [
                {"text": "This is a single response."},
                {"text": "This is a second response."},
            ]
        }
        expected_responses = [
            "This is a single response.",
            "This is a second response.",
        ]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"text": " This"}'}},
            {"chunk": {"bytes": b'{"text": " is"}'}},
            {"chunk": {"bytes": b'{"text": " a"}'}},
            {"chunk": {"bytes": b'{"text": " single"}'}},
            {"chunk": {"bytes": b'{"text": " response."}'}},
            {"chunk": {"bytes": b'{"finish_reason": "MAX_TOKENS", "is_finished": true}'}},
        ]

        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(StreamingChunk(content=" This", meta={"text": " This"})),
                call(StreamingChunk(content=" is", meta={"text": " is"})),
                call(StreamingChunk(content=" a", meta={"text": " a"})),
                call(StreamingChunk(content=" single", meta={"text": " single"})),
                call(StreamingChunk(content=" response.", meta={"text": " response."})),
                call(StreamingChunk(content="", meta={"finish_reason": "MAX_TOKENS", "is_finished": True})),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_not_called()


class TestCohereCommandRAdapter:
    def test_prepare_body(self) -> None:
        adapter = CohereCommandRAdapter(
            model_kwargs={
                "chat_history": [
                    {"role": "CHATBOT", "content": "How can I help you today?"},
                ],
                "documents": [
                    {"title": "France", "snippet": "Paris is the capital of France."},
                    {"title": "Germany", "snippet": "Berlin is the capital of Germany."},
                ],
                "search_query_only": False,
                "preamble": "preamble",
                "temperature": 0,
                "p": 0.9,
                "k": 50,
                "prompt_truncation": "AUTO_PRESERVE_ORDER",
                "frequency_penalty": 0.3,
                "presence_penalty": 0.4,
                "seed": 42,
                "return_prompt": True,
                "tools": [
                    {
                        "name": "query_daily_sales_report",
                        "description": "Connects to a database to retrieve overall sales volumes and sales "
                        "information for a given day.",
                        "parameter_definitions": {
                            "day": {
                                "description": "Retrieves sales data for this day, formatted as YYYY-MM-DD.",
                                "type": "str",
                                "required": True,
                            }
                        },
                    }
                ],
                "tool_results": [
                    {
                        "call": {"name": "query_daily_sales_report", "parameters": {"day": "2023-09-29"}},
                        "outputs": [
                            {"date": "2023-09-29", "summary": "Total Sales Amount: 10000, Total Units Sold: 250"}
                        ],
                    }
                ],
                "stop_sequences": ["\n\n"],
                "raw_prompting": True,
                "stream": True,
                "unknown_arg": "unknown_arg",
            },
            max_length=100,
        )
        body = adapter.prepare_body(prompt="test")
        assert body == {
            "message": "test",
            "chat_history": [
                {"role": "CHATBOT", "content": "How can I help you today?"},
            ],
            "documents": [
                {"title": "France", "snippet": "Paris is the capital of France."},
                {"title": "Germany", "snippet": "Berlin is the capital of Germany."},
            ],
            "search_query_only": False,
            "preamble": "preamble",
            "max_tokens": 100,
            "temperature": 0,
            "p": 0.9,
            "k": 50,
            "prompt_truncation": "AUTO_PRESERVE_ORDER",
            "frequency_penalty": 0.3,
            "presence_penalty": 0.4,
            "seed": 42,
            "return_prompt": True,
            "tools": [
                {
                    "name": "query_daily_sales_report",
                    "description": "Connects to a database to retrieve overall sales volumes and sales "
                    "information for a given day.",
                    "parameter_definitions": {
                        "day": {
                            "description": "Retrieves sales data for this day, formatted as YYYY-MM-DD.",
                            "type": "str",
                            "required": True,
                        }
                    },
                }
            ],
            "tool_results": [
                {
                    "call": {"name": "query_daily_sales_report", "parameters": {"day": "2023-09-29"}},
                    "outputs": [{"date": "2023-09-29", "summary": "Total Sales Amount: 10000, Total Units Sold: 250"}],
                }
            ],
            "stop_sequences": ["\n\n"],
            "raw_prompting": True,
        }

    def test_extract_completions_from_response(self) -> None:
        adapter = CohereCommandRAdapter(model_kwargs={}, max_length=100)
        response_body = {"text": "response"}
        completions = adapter._extract_completions_from_response(response_body=response_body)
        assert completions == ["response"]

    def test_build_chunk(self) -> None:
        adapter = CohereCommandRAdapter(model_kwargs={}, max_length=100)
        chunk = {"text": "response_token"}
        streaming_chunk = adapter._build_streaming_chunk(chunk=chunk)
        assert streaming_chunk == StreamingChunk(content="response_token", meta=chunk)


class TestAI21LabsJurassic2Adapter:
    def test_prepare_body_with_default_params(self) -> None:
        layer = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {"prompt": "Hello, how are you?", "maxTokens": 99}

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        layer = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "maxTokens": 50,
            "stopSequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "topP": 0.8,
            "countPenalty": {"scale": 1.0},
            "presencePenalty": {"scale": 5.0},
            "frequencyPenalty": {"scale": 500.0},
            "numResults": 1,
        }

        body = layer.prepare_body(
            prompt,
            maxTokens=50,
            stopSequences=["CUSTOM_STOP"],
            temperature=0.7,
            topP=0.8,
            countPenalty={"scale": 1.0},
            presencePenalty={"scale": 5.0},
            frequencyPenalty={"scale": 500.0},
            numResults=1,
            unknown_arg="unknown_value",
        )

        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        layer = AI21LabsJurassic2Adapter(
            model_kwargs={
                "maxTokens": 50,
                "stopSequences": ["CUSTOM_STOP"],
                "temperature": 0.7,
                "topP": 0.8,
                "countPenalty": {"scale": 1.0},
                "presencePenalty": {"scale": 5.0},
                "frequencyPenalty": {"scale": 500.0},
                "numResults": 1,
                "unknown_arg": "unknown_value",
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "maxTokens": 50,
            "stopSequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "topP": 0.8,
            "countPenalty": {"scale": 1.0},
            "presencePenalty": {"scale": 5.0},
            "frequencyPenalty": {"scale": 500.0},
            "numResults": 1,
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        layer = AI21LabsJurassic2Adapter(
            model_kwargs={
                "maxTokens": 49,
                "stopSequences": ["CUSTOM_STOP_MODEL_KWARGS"],
                "temperature": 0.6,
                "topP": 0.7,
                "countPenalty": {"scale": 0.9},
                "presencePenalty": {"scale": 4.0},
                "frequencyPenalty": {"scale": 499.0},
                "numResults": 2,
                "unknown_arg": "unknown_value",
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "maxTokens": 50,
            "stopSequences": ["CUSTOM_STOP_MODEL_KWARGS"],
            "temperature": 0.7,
            "topP": 0.8,
            "countPenalty": {"scale": 1.0},
            "presencePenalty": {"scale": 5.0},
            "frequencyPenalty": {"scale": 500.0},
            "numResults": 1,
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            topP=0.8,
            maxTokens=50,
            countPenalty={"scale": 1.0},
            presencePenalty={"scale": 5.0},
            frequencyPenalty={"scale": 500.0},
            numResults=1,
        )

        assert body == expected_body

    def test_get_responses(self) -> None:
        adapter = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        response_body = {"completions": [{"data": {"text": "This is a single response."}}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        adapter = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        response_body = {"completions": [{"data": {"text": "\n\t This is a single response."}}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_multiple_responses(self) -> None:
        adapter = AI21LabsJurassic2Adapter(model_kwargs={}, max_length=99)
        response_body = {
            "completions": [
                {"data": {"text": "This is a single response."}},
                {"data": {"text": "This is a second response."}},
            ]
        }
        expected_responses = [
            "This is a single response.",
            "This is a second response.",
        ]
        assert adapter.get_responses(response_body) == expected_responses


class TestAmazonTitanAdapter:
    def test_prepare_body_with_default_params(self) -> None:
        layer = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "inputText": "Hello, how are you?",
            "textGenerationConfig": {"maxTokenCount": 99},
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        layer = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "inputText": "Hello, how are you?",
            "textGenerationConfig": {
                "maxTokenCount": 50,
                "stopSequences": ["CUSTOM_STOP"],
                "temperature": 0.7,
                "topP": 0.8,
            },
        }

        body = layer.prepare_body(
            prompt,
            maxTokenCount=50,
            stopSequences=["CUSTOM_STOP"],
            temperature=0.7,
            topP=0.8,
            unknown_arg="unknown_value",
        )

        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        layer = AmazonTitanAdapter(
            model_kwargs={
                "maxTokenCount": 50,
                "stopSequences": ["CUSTOM_STOP"],
                "temperature": 0.7,
                "topP": 0.8,
                "unknown_arg": "unknown_value",
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "inputText": "Hello, how are you?",
            "textGenerationConfig": {
                "maxTokenCount": 50,
                "stopSequences": ["CUSTOM_STOP"],
                "temperature": 0.7,
                "topP": 0.8,
            },
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        layer = AmazonTitanAdapter(
            model_kwargs={
                "maxTokenCount": 49,
                "stopSequences": ["CUSTOM_STOP_MODEL_KWARGS"],
                "temperature": 0.6,
                "topP": 0.7,
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "inputText": "Hello, how are you?",
            "textGenerationConfig": {
                "maxTokenCount": 50,
                "stopSequences": ["CUSTOM_STOP_MODEL_KWARGS"],
                "temperature": 0.7,
                "topP": 0.8,
            },
        }

        body = layer.prepare_body(prompt, temperature=0.7, topP=0.8, maxTokenCount=50)

        assert body == expected_body

    def test_get_responses(self) -> None:
        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        response_body = {"results": [{"outputText": "This is a single response."}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        response_body = {"results": [{"outputText": "\n\t This is a single response."}]}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_multiple_responses(self) -> None:
        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        response_body = {
            "results": [
                {"outputText": "This is a single response."},
                {"outputText": "This is a second response."},
            ]
        }
        expected_responses = [
            "This is a single response.",
            "This is a second response.",
        ]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"outputText": " This"}'}},
            {"chunk": {"bytes": b'{"outputText": " is"}'}},
            {"chunk": {"bytes": b'{"outputText": " a"}'}},
            {"chunk": {"bytes": b'{"outputText": " single"}'}},
            {"chunk": {"bytes": b'{"outputText": " response."}'}},
        ]

        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(StreamingChunk(content=" This", meta={"outputText": " This"})),
                call(StreamingChunk(content=" is", meta={"outputText": " is"})),
                call(StreamingChunk(content=" a", meta={"outputText": " a"})),
                call(StreamingChunk(content=" single", meta={"outputText": " single"})),
                call(StreamingChunk(content=" response.", meta={"outputText": " response."})),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_not_called()


class TestMetaLlamaAdapter:
    def test_prepare_body_with_default_params(self) -> None:
        layer = MetaLlamaAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {"prompt": "Hello, how are you?", "max_gen_len": 99}

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        layer = MetaLlamaAdapter(model_kwargs={}, max_length=99)
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "max_gen_len": 50,
            "temperature": 0.7,
            "top_p": 0.8,
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            top_p=0.8,
            max_gen_len=50,
            unknown_arg="unknown_value",
        )

        assert body == expected_body

    def test_prepare_body_with_model_kwargs(self) -> None:
        layer = MetaLlamaAdapter(
            model_kwargs={
                "temperature": 0.7,
                "top_p": 0.8,
                "max_gen_len": 50,
                "unknown_arg": "unknown_value",
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "max_gen_len": 50,
            "temperature": 0.7,
            "top_p": 0.8,
        }

        body = layer.prepare_body(prompt)

        assert body == expected_body

    def test_prepare_body_with_model_kwargs_and_custom_inference_params(self) -> None:
        layer = MetaLlamaAdapter(
            model_kwargs={
                "temperature": 0.6,
                "top_p": 0.7,
                "top_k": 4,
                "max_gen_len": 49,
            },
            max_length=99,
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "Hello, how are you?",
            "max_gen_len": 50,
            "temperature": 0.7,
            "top_p": 0.7,
        }

        body = layer.prepare_body(prompt, temperature=0.7, max_gen_len=50)

        assert body == expected_body

    def test_get_responses(self) -> None:
        adapter = MetaLlamaAdapter(model_kwargs={}, max_length=99)
        response_body = {"generation": "This is a single response."}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_responses_leading_whitespace(self) -> None:
        adapter = MetaLlamaAdapter(model_kwargs={}, max_length=99)
        response_body = {"generation": "\n\t This is a single response."}
        expected_responses = ["This is a single response."]
        assert adapter.get_responses(response_body) == expected_responses

    def test_get_stream_responses(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"generation": " This"}'}},
            {"chunk": {"bytes": b'{"generation": " is"}'}},
            {"chunk": {"bytes": b'{"generation": " a"}'}},
            {"chunk": {"bytes": b'{"generation": " single"}'}},
            {"chunk": {"bytes": b'{"generation": " response."}'}},
        ]

        adapter = MetaLlamaAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_has_calls(
            [
                call(StreamingChunk(content=" This", meta={"generation": " This"})),
                call(StreamingChunk(content=" is", meta={"generation": " is"})),
                call(StreamingChunk(content=" a", meta={"generation": " a"})),
                call(StreamingChunk(content=" single", meta={"generation": " single"})),
                call(StreamingChunk(content=" response.", meta={"generation": " response."})),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        streaming_callback_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        adapter = MetaLlamaAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, streaming_callback_mock) == expected_responses

        streaming_callback_mock.assert_not_called()

    def test_run_with_metadata(self, mock_boto3_session) -> None:
        generator = AmazonBedrockGenerator(model="anthropic.claude-v2")
        mock_client = mock_boto3_session.return_value.client.return_value

        # Create a proper mock for the response body
        mock_body = MagicMock()
        mock_body.read.return_value = b'{"content": [{"type": "text", "text": "test response"}]}'

        mock_response = {
            "body": mock_body,
            "ResponseMetadata": {
                "RequestId": "test-request-id",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {"x-amzn-requestid": "test-request-id", "content-type": "application/json"},
            },
        }
        mock_client.invoke_model.return_value = mock_response

        result = generator.run("Hello, how are you?")

        assert isinstance(result, dict)
        assert "meta" in result
        assert result["meta"] == mock_response["ResponseMetadata"]
        assert result["meta"]["RequestId"] == "test-request-id"
        assert result["meta"]["HTTPStatusCode"] == 200
        assert result["meta"]["HTTPHeaders"]["content-type"] == "application/json"
