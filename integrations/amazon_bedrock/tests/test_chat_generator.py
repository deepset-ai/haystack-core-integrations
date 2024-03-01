from typing import Optional, Type

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator
from haystack_integrations.components.generators.amazon_bedrock.chat.adapters import (
    AnthropicClaudeChatAdapter,
    BedrockModelChatAdapter,
    MetaLlama2ChatAdapter,
)

clazz = "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"


def test_to_dict(mock_boto3_session):
    """
    Test that the to_dict method returns the correct dictionary without aws credentials
    """
    generator = AmazonBedrockChatGenerator(
        model="anthropic.claude-v2",
        generation_kwargs={"temperature": 0.7},
        streaming_callback=print_streaming_chunk,
    )
    expected_dict = {
        "type": clazz,
        "init_parameters": {
            "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
            "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
            "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
            "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
            "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
            "model": "anthropic.claude-v2",
            "generation_kwargs": {"temperature": 0.7},
            "stop_words": [],
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
        },
    }

    assert generator.to_dict() == expected_dict


def test_from_dict(mock_boto3_session):
    """
    Test that the from_dict method returns the correct object
    """
    generator = AmazonBedrockChatGenerator.from_dict(
        {
            "type": clazz,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "anthropic.claude-v2",
                "generation_kwargs": {"temperature": 0.7},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            },
        }
    )
    assert generator.model == "anthropic.claude-v2"
    assert generator.model_adapter.generation_kwargs == {"temperature": 0.7}
    assert generator.streaming_callback == print_streaming_chunk


def test_default_constructor(mock_boto3_session, set_env_variables):
    """
    Test that the default constructor sets the correct values
    """

    layer = AmazonBedrockChatGenerator(
        model="anthropic.claude-v2",
    )

    assert layer.model == "anthropic.claude-v2"

    assert layer.model_adapter.prompt_handler is not None
    assert layer.model_adapter.prompt_handler.model_max_length == 100000

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

    layer = AmazonBedrockChatGenerator(model="anthropic.claude-v2", generation_kwargs=generation_kwargs)
    assert "temperature" in layer.model_adapter.generation_kwargs
    assert layer.model_adapter.generation_kwargs["temperature"] == 0.7


def test_constructor_with_empty_model():
    """
    Test that the constructor raises an error when the model is empty
    """
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        AmazonBedrockChatGenerator(model="")


def test_invoke_with_no_kwargs(mock_boto3_session):
    """
    Test invoke raises an error if no messages are provided
    """
    layer = AmazonBedrockChatGenerator(model="anthropic.claude-v2")
    with pytest.raises(ValueError, match="The model anthropic.claude-v2 requires"):
        layer.invoke()


@pytest.mark.parametrize(
    "model, expected_model_adapter",
    [
        ("anthropic.claude-v1", AnthropicClaudeChatAdapter),
        ("anthropic.claude-v2", AnthropicClaudeChatAdapter),
        ("anthropic.claude-instant-v1", AnthropicClaudeChatAdapter),
        ("anthropic.claude-super-v5", AnthropicClaudeChatAdapter),  # artificial
        ("meta.llama2-13b-chat-v1", MetaLlama2ChatAdapter),
        ("meta.llama2-70b-chat-v1", MetaLlama2ChatAdapter),
        ("meta.llama2-130b-v5", MetaLlama2ChatAdapter),  # artificial
        ("unknown_model", None),
    ],
)
def test_get_model_adapter(model: str, expected_model_adapter: Optional[Type[BedrockModelChatAdapter]]):
    """
    Test that the correct model adapter is returned for a given model
    """
    model_adapter = AmazonBedrockChatGenerator.get_model_adapter(model=model)
    assert model_adapter == expected_model_adapter


class TestAnthropicClaudeAdapter:
    def test_prepare_body_with_default_params(self) -> None:
        layer = AnthropicClaudeChatAdapter(generation_kwargs={})
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "\n\nHuman: Hello, how are you?\n\nAssistant: ",
            "max_tokens_to_sample": 512,
            "stop_sequences": ["\n\nHuman:"],
        }

        body = layer.prepare_body([ChatMessage.from_user(prompt)])

        assert body == expected_body

    def test_prepare_body_with_custom_inference_params(self) -> None:
        layer = AnthropicClaudeChatAdapter(generation_kwargs={"temperature": 0.7, "top_p": 0.8, "top_k": 4})
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "\n\nHuman: Hello, how are you?\n\nAssistant: ",
            "max_tokens_to_sample": 69,
            "stop_sequences": ["\n\nHuman:", "CUSTOM_STOP"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
        }

        body = layer.prepare_body(
            [ChatMessage.from_user(prompt)], top_p=0.8, top_k=5, max_tokens_to_sample=69, stop_sequences=["CUSTOM_STOP"]
        )

        assert body == expected_body

    @pytest.mark.integration
    def test_get_responses(self) -> None:
        adapter = AnthropicClaudeChatAdapter(generation_kwargs={})
        response_body = {"completion": "This is a single response."}
        expected_response = "This is a single response."
        response_message = adapter.get_responses(response_body)
        # assert that the type of each item in the list is a ChatMessage
        for message in response_message:
            assert isinstance(message, ChatMessage)

        assert response_message == [ChatMessage.from_assistant(expected_response)]


class TestMetaLlama2ChatAdapter:
    @pytest.mark.integration
    def test_prepare_body_with_default_params(self) -> None:
        # leave this test as integration because we really need only tokenizer from HF
        # that way we can ensure prompt chat message formatting
        layer = MetaLlama2ChatAdapter(generation_kwargs={})
        prompt = "Hello, how are you?"
        expected_body = {"prompt": "<s>[INST] Hello, how are you? [/INST]", "max_gen_len": 512}

        body = layer.prepare_body([ChatMessage.from_user(prompt)])

        assert body == expected_body

    @pytest.mark.integration
    def test_prepare_body_with_custom_inference_params(self) -> None:
        # leave this test as integration because we really need only tokenizer from HF
        # that way we can ensure prompt chat message formatting
        layer = MetaLlama2ChatAdapter(
            generation_kwargs={"temperature": 0.7, "top_p": 0.8, "top_k": 5, "stop_sequences": ["CUSTOM_STOP"]}
        )
        prompt = "Hello, how are you?"
        expected_body = {
            "prompt": "<s>[INST] Hello, how are you? [/INST]",
            "max_gen_len": 69,
            "stop_sequences": ["CUSTOM_STOP"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 5,
        }

        body = layer.prepare_body(
            [ChatMessage.from_user(prompt)],
            temperature=0.7,
            top_p=0.8,
            top_k=5,
            max_gen_len=69,
            stop_sequences=["CUSTOM_STOP"],
        )

        assert body == expected_body

    @pytest.mark.integration
    def test_get_responses(self) -> None:
        adapter = MetaLlama2ChatAdapter(generation_kwargs={})
        response_body = {"generation": "This is a single response."}
        expected_response = "This is a single response."
        response_message = adapter.get_responses(response_body)
        # assert that the type of each item in the list is a ChatMessage
        for message in response_message:
            assert isinstance(message, ChatMessage)

        assert response_message == [ChatMessage.from_assistant(expected_response)]
