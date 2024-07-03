from typing import Optional, Type
from unittest.mock import MagicMock, call, patch

import pytest

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


def test_to_dict(mock_boto3_session):
    """
    Test that the to_dict method returns the correct dictionary without aws credentials
    """
    generator = AmazonBedrockGenerator(model="anthropic.claude-v2", max_length=99, truncate=False, temperature=10)

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
            "truncate": False,
            "temperature": 10,
        },
    }

    assert generator.to_dict() == expected_dict


def test_from_dict(mock_boto3_session):
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
            },
        }
    )

    assert generator.max_length == 99
    assert generator.model == "anthropic.claude-v2"


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

    assert layer.prompt_handler is not None
    assert layer.prompt_handler.model_max_length == 4096

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


def test_constructor_prompt_handler_initialized(mock_boto3_session, mock_prompt_handler):
    """
    Test that the constructor sets the prompt_handler correctly, with the correct model_max_length for llama-2
    """
    layer = AmazonBedrockGenerator(model="anthropic.claude-v2", prompt_handler=mock_prompt_handler)
    assert layer.prompt_handler is not None
    assert layer.prompt_handler.model_max_length == 4096


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


def test_invoke_with_no_kwargs(mock_boto3_session):
    """
    Test invoke raises an error if no prompt is provided
    """
    layer = AmazonBedrockGenerator(model="anthropic.claude-v2")
    with pytest.raises(ValueError, match="The model anthropic.claude-v2 requires a valid prompt."):
        layer.invoke()


def test_short_prompt_is_not_truncated(mock_boto3_session):
    """
    Test that a short prompt is not truncated
    """
    # Define a short mock prompt and its tokenized version
    mock_prompt_text = "I am a tokenized prompt"
    mock_prompt_tokens = mock_prompt_text.split()

    # Mock the tokenizer so it returns our predefined tokens
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = mock_prompt_tokens

    # We set a small max_length for generated text (3 tokens) and a total model_max_length of 10 tokens
    # Since our mock prompt is 5 tokens long, it doesn't exceed the
    # total limit (5 prompt tokens + 3 generated tokens < 10 tokens)
    max_length_generated_text = 3
    total_model_max_length = 10

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        layer = AmazonBedrockGenerator(
            "anthropic.claude-v2",
            max_length=max_length_generated_text,
            model_max_length=total_model_max_length,
        )
        prompt_after_resize = layer._ensure_token_limit(mock_prompt_text)

    # The prompt doesn't exceed the limit, _ensure_token_limit doesn't truncate it
    assert prompt_after_resize == mock_prompt_text


def test_long_prompt_is_truncated(mock_boto3_session):
    """
    Test that a long prompt is truncated
    """
    # Define a long mock prompt and its tokenized version
    long_prompt_text = "I am a tokenized prompt of length eight"
    long_prompt_tokens = long_prompt_text.split()

    # _ensure_token_limit will truncate the prompt to make it fit into the model's max token limit
    truncated_prompt_text = "I am a tokenized prompt of length"

    # Mock the tokenizer to return our predefined tokens
    # convert tokens to our predefined truncated text
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = long_prompt_tokens
    mock_tokenizer.convert_tokens_to_string.return_value = truncated_prompt_text

    # We set a small max_length for generated text (3 tokens) and a total model_max_length of 10 tokens
    # Our mock prompt is 8 tokens long, so it exceeds the total limit (8 prompt tokens + 3 generated tokens > 10 tokens)
    max_length_generated_text = 3
    total_model_max_length = 10

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        layer = AmazonBedrockGenerator(
            "anthropic.claude-v2",
            max_length=max_length_generated_text,
            model_max_length=total_model_max_length,
        )
        prompt_after_resize = layer._ensure_token_limit(long_prompt_text)

    # The prompt exceeds the limit, _ensure_token_limit truncates it
    assert prompt_after_resize == truncated_prompt_text


def test_long_prompt_is_not_truncated_when_truncate_false(mock_boto3_session):
    """
    Test that a long prompt is not truncated and _ensure_token_limit is not called when truncate is set to False
    """
    long_prompt_text = "I am a tokenized prompt of length eight"

    # Our mock prompt is 8 tokens long, so it exceeds the total limit (8 prompt tokens + 3 generated tokens > 10 tokens)
    max_length_generated_text = 3
    total_model_max_length = 10

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()):
        generator = AmazonBedrockGenerator(
            model="anthropic.claude-v2",
            max_length=max_length_generated_text,
            model_max_length=total_model_max_length,
            truncate=False,
        )

        # Mock the _ensure_token_limit method to track if it is called
        with patch.object(
            generator, "_ensure_token_limit", wraps=generator._ensure_token_limit
        ) as mock_ensure_token_limit:
            # Mock the model adapter to avoid actual invocation
            generator.model_adapter.prepare_body = MagicMock(return_value={})
            generator.client = MagicMock()
            generator.client.invoke_model = MagicMock(
                return_value={"body": MagicMock(read=MagicMock(return_value=b'{"generated_text": "response"}'))}
            )
            generator.model_adapter.get_responses = MagicMock(return_value=["response"])

            # Invoke the generator
            generator.invoke(prompt=long_prompt_text)

        # Ensure _ensure_token_limit was not called
        mock_ensure_token_limit.assert_not_called(),

        # Check the prompt passed to prepare_body
        generator.model_adapter.prepare_body.assert_called_with(prompt=long_prompt_text)


@pytest.mark.parametrize(
    "model, expected_model_adapter",
    [
        ("anthropic.claude-v1", AnthropicClaudeAdapter),
        ("anthropic.claude-v2", AnthropicClaudeAdapter),
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
        ("amazon.titan-text-agile-v1", AmazonTitanAdapter),
        ("amazon.titan-text-lightning-v8", AmazonTitanAdapter),  # artificial
        ("meta.llama2-13b-chat-v1", MetaLlamaAdapter),
        ("meta.llama2-70b-chat-v1", MetaLlamaAdapter),
        ("meta.llama2-130b-v5", MetaLlamaAdapter),  # artificial
        ("meta.llama3-8b-instruct-v1:0", MetaLlamaAdapter),
        ("meta.llama3-70b-instruct-v1:0", MetaLlamaAdapter),
        ("meta.llama3-130b-instruct-v5:9", MetaLlamaAdapter),  # artificial
        ("mistral.mistral-7b-instruct-v0:2", MistralAdapter),
        ("mistral.mixtral-8x7b-instruct-v0:1", MistralAdapter),
        ("mistral.mistral-large-2402-v1:0", MistralAdapter),
        ("mistral.mistral-medium-v8:0", MistralAdapter),  # artificial
        ("unknown_model", None),
    ],
)
def test_get_model_adapter(model: str, expected_model_adapter: Optional[Type[BedrockModelAdapter]]):
    """
    Test that the correct model adapter is returned for a given model
    """
    model_adapter = AmazonBedrockGenerator.get_model_adapter(model=model)
    assert model_adapter == expected_model_adapter


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
        }

        body = layer.prepare_body(
            prompt,
            temperature=0.7,
            top_p=0.8,
            top_k=5,
            max_tokens=50,
            system="new system prompt",
            anthropic_version="new_custom_version",
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

    def test_get_stream_responses(self) -> None:
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"delta": {"text": " This"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " is"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " a"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " single"}}'}},
            {"chunk": {"bytes": b'{"delta": {"text": " response."}}'}},
        ]

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_has_calls(
            [
                call(" This", event_data={"delta": {"text": " This"}}),
                call(" is", event_data={"delta": {"text": " is"}}),
                call(" a", event_data={"delta": {"text": " a"}}),
                call(" single", event_data={"delta": {"text": " single"}}),
                call(" response.", event_data={"delta": {"text": " response."}}),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = AnthropicClaudeAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_not_called()


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
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"completion": " This"}'}},
            {"chunk": {"bytes": b'{"completion": " is"}'}},
            {"chunk": {"bytes": b'{"completion": " a"}'}},
            {"chunk": {"bytes": b'{"completion": " single"}'}},
            {"chunk": {"bytes": b'{"completion": " response."}'}},
        ]

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_has_calls(
            [
                call(" This", event_data={"completion": " This"}),
                call(" is", event_data={"completion": " is"}),
                call(" a", event_data={"completion": " a"}),
                call(" single", event_data={"completion": " single"}),
                call(" response.", event_data={"completion": " response."}),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = AnthropicClaudeAdapter(model_kwargs={"use_messages_api": False}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_not_called()


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
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"outputs": [{"text": " This"}]}'}},
            {"chunk": {"bytes": b'{"outputs": [{"text": " is"}]}'}},
            {"chunk": {"bytes": b'{"outputs": [{"text": " a"}]}'}},
            {"chunk": {"bytes": b'{"outputs": [{"text": " single"}]}'}},
            {"chunk": {"bytes": b'{"outputs": [{"text": " response."}]}'}},
        ]

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = MistralAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_has_calls(
            [
                call(" This", event_data={"outputs": [{"text": " This"}]}),
                call(" is", event_data={"outputs": [{"text": " is"}]}),
                call(" a", event_data={"outputs": [{"text": " a"}]}),
                call(" single", event_data={"outputs": [{"text": " single"}]}),
                call(" response.", event_data={"outputs": [{"text": " response."}]}),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = MistralAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_not_called()


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
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"text": " This"}'}},
            {"chunk": {"bytes": b'{"text": " is"}'}},
            {"chunk": {"bytes": b'{"text": " a"}'}},
            {"chunk": {"bytes": b'{"text": " single"}'}},
            {"chunk": {"bytes": b'{"text": " response."}'}},
            {"chunk": {"bytes": b'{"finish_reason": "MAX_TOKENS", "is_finished": true}'}},
        ]

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_has_calls(
            [
                call(" This", event_data={"text": " This"}),
                call(" is", event_data={"text": " is"}),
                call(" a", event_data={"text": " a"}),
                call(" single", event_data={"text": " single"}),
                call(" response.", event_data={"text": " response."}),
                call("", event_data={"finish_reason": "MAX_TOKENS", "is_finished": True}),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = CohereCommandAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_not_called()


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

    def test_extract_token_from_stream(self) -> None:
        adapter = CohereCommandRAdapter(model_kwargs={}, max_length=100)
        chunk = {"text": "response_token"}
        token = adapter._extract_token_from_stream(chunk=chunk)
        assert token == "response_token"


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
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"outputText": " This"}'}},
            {"chunk": {"bytes": b'{"outputText": " is"}'}},
            {"chunk": {"bytes": b'{"outputText": " a"}'}},
            {"chunk": {"bytes": b'{"outputText": " single"}'}},
            {"chunk": {"bytes": b'{"outputText": " response."}'}},
        ]

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_has_calls(
            [
                call(" This", event_data={"outputText": " This"}),
                call(" is", event_data={"outputText": " is"}),
                call(" a", event_data={"outputText": " a"}),
                call(" single", event_data={"outputText": " single"}),
                call(" response.", event_data={"outputText": " response."}),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = AmazonTitanAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_not_called()


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
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = [
            {"chunk": {"bytes": b'{"generation": " This"}'}},
            {"chunk": {"bytes": b'{"generation": " is"}'}},
            {"chunk": {"bytes": b'{"generation": " a"}'}},
            {"chunk": {"bytes": b'{"generation": " single"}'}},
            {"chunk": {"bytes": b'{"generation": " response."}'}},
        ]

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = MetaLlamaAdapter(model_kwargs={}, max_length=99)
        expected_responses = ["This is a single response."]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_has_calls(
            [
                call(" This", event_data={"generation": " This"}),
                call(" is", event_data={"generation": " is"}),
                call(" a", event_data={"generation": " a"}),
                call(" single", event_data={"generation": " single"}),
                call(" response.", event_data={"generation": " response."}),
            ]
        )

    def test_get_stream_responses_empty(self) -> None:
        stream_mock = MagicMock()
        stream_handler_mock = MagicMock()

        stream_mock.__iter__.return_value = []

        stream_handler_mock.side_effect = lambda token_received, **kwargs: token_received

        adapter = MetaLlamaAdapter(model_kwargs={}, max_length=99)
        expected_responses = [""]
        assert adapter.get_stream_responses(stream_mock, stream_handler_mock) == expected_responses

        stream_handler_mock.assert_not_called()
