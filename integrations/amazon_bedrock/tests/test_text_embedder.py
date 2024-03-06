import io
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.components.embedders.amazon_bedrock import AmazonBedrockTextEmbedder


class TestAmazonBedrockTextEmbedder:
    def test_init(self, mock_boto3_session, set_env_variables):
        embedder = AmazonBedrockTextEmbedder(
            model="cohere.embed-english-v3",
            input_type="fake_input_type",
        )

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"input_type": "fake_input_type"}

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

    def test_connection_error(self, mock_boto3_session):
        mock_boto3_session.side_effect = Exception("some connection error")

        with pytest.raises(AmazonBedrockConfigurationError):
            AmazonBedrockTextEmbedder(
                model="cohere.embed-english-v3",
                input_type="fake_input_type",
            )

    def test_to_dict(self, mock_boto3_session):

        embedder = AmazonBedrockTextEmbedder(
            model="cohere.embed-english-v3",
            input_type="search_query",
        )

        expected_dict = {
            "type": "haystack_integrations.components.embedders.amazon_bedrock.text_embedder.AmazonBedrockTextEmbedder",
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "cohere.embed-english-v3",
                "input_type": "search_query",
            },
        }

        assert embedder.to_dict() == expected_dict

    def test_from_dict(self, mock_boto3_session):

        data = {
            "type": "haystack_integrations.components.embedders.amazon_bedrock.text_embedder.AmazonBedrockTextEmbedder",
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "cohere.embed-english-v3",
                "input_type": "search_query",
            },
        }

        embedder = AmazonBedrockTextEmbedder.from_dict(data)

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"input_type": "search_query"}

    def test_init_invalid_model(self):
        with pytest.raises(ValueError):
            AmazonBedrockTextEmbedder(model="")

        with pytest.raises(ValueError):
            AmazonBedrockTextEmbedder(model="my-unsupported-model")

    def test_run_wrong_type(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        with pytest.raises(TypeError):
            embedder.run(text=123)

    def test_cohere_invocation(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")

        with patch.object(embedder._client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.return_value = {
                "body": io.StringIO('{"embeddings": [[0.1, 0.2, 0.3]]}'),
            }
            result = embedder.run(text="some text")

            mock_invoke_model.assert_called_once_with(
                body='{"texts": ["some text"], "input_type": "search_query"}',
                modelId="cohere.embed-english-v3",
                accept="*/*",
                contentType="application/json",
            )

            assert result == {"embedding": [0.1, 0.2, 0.3]}

    def test_titan_invocation(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="amazon.titan-embed-text-v1")

        with patch.object(embedder._client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.return_value = {
                "body": io.StringIO('{"embedding": [0.1, 0.2, 0.3]}'),
            }
            result = embedder.run(text="some text")

            mock_invoke_model.assert_called_once_with(
                body='{"inputText": "some text"}',
                modelId="amazon.titan-embed-text-v1",
                accept="*/*",
                contentType="application/json",
            )

            assert result == {"embedding": [0.1, 0.2, 0.3]}

    def test_run_invocation_error(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")

        with patch.object(embedder._client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.side_effect = ClientError(
                error_response={"Error": {"Code": "some_code", "Message": "some_message"}},
                operation_name="some_operation",
            )

            with pytest.raises(AmazonBedrockInferenceError):
                embedder.run(text="some text")
