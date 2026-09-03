import io
import os
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from haystack.utils import Secret

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.components.embedders.amazon_bedrock import (
    AmazonBedrockTextEmbedder,
)


class TestAmazonBedrockTextEmbedder:
    def test_init(self):
        embedder = AmazonBedrockTextEmbedder(
            model="cohere.embed-english-v3",
            input_type="fake_input_type",
        )

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"input_type": "fake_input_type"}

    def test_to_dict(self):
        embedder = AmazonBedrockTextEmbedder(
            model="cohere.embed-english-v3",
            input_type="search_query",
        )

        expected_dict = {
            "type": "haystack_integrations.components.embedders.amazon_bedrock.text_embedder.AmazonBedrockTextEmbedder",
            "init_parameters": {
                "aws_access_key_id": {
                    "type": "env_var",
                    "env_vars": ["AWS_ACCESS_KEY_ID"],
                    "strict": False,
                },
                "aws_secret_access_key": {
                    "type": "env_var",
                    "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                    "strict": False,
                },
                "aws_session_token": {
                    "type": "env_var",
                    "env_vars": ["AWS_SESSION_TOKEN"],
                    "strict": False,
                },
                "aws_region_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_DEFAULT_REGION"],
                    "strict": False,
                },
                "aws_profile_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_PROFILE"],
                    "strict": False,
                },
                "model": "cohere.embed-english-v3",
                "input_type": "search_query",
                "boto3_config": None,
            },
        }

        assert embedder.to_dict() == expected_dict

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.embedders.amazon_bedrock.text_embedder.AmazonBedrockTextEmbedder",
            "init_parameters": {
                "aws_access_key_id": {
                    "type": "env_var",
                    "env_vars": ["AWS_ACCESS_KEY_ID"],
                    "strict": False,
                },
                "aws_secret_access_key": {
                    "type": "env_var",
                    "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                    "strict": False,
                },
                "aws_session_token": {
                    "type": "env_var",
                    "env_vars": ["AWS_SESSION_TOKEN"],
                    "strict": False,
                },
                "aws_region_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_DEFAULT_REGION"],
                    "strict": False,
                },
                "aws_profile_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_PROFILE"],
                    "strict": False,
                },
                "model": "cohere.embed-english-v3",
                "input_type": "search_query",
                "boto3_config": {
                    "read_timeout": 1000,
                },
            },
        }

        embedder = AmazonBedrockTextEmbedder.from_dict(data)

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"input_type": "search_query"}
        assert embedder.boto3_config == {"read_timeout": 1000}

    def test_from_dict_aws_region_name(self):
        """
        Test that aws_region_name as str value is correctly parsed
        """
        embedder = AmazonBedrockTextEmbedder.from_dict(
            {
                "type": "haystack_integrations.components.embedders.amazon_bedrock.text_embedder.AmazonBedrockTextEmbedder",  # noqa: E501
                "init_parameters": {
                    "aws_region_name": "my-fake-region",
                    "model": "cohere.embed-english-v3",
                },
            }
        )
        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.aws_region_name == "my-fake-region"

        serialized = embedder.to_dict()
        assert serialized["init_parameters"]["aws_region_name"] == "my-fake-region"

    def test_init_invalid_model(self):
        with pytest.raises(ValueError):
            AmazonBedrockTextEmbedder(model="")

        with pytest.raises(ValueError):
            AmazonBedrockTextEmbedder(model="my-unsupported-model")

    def test_run_wrong_type(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        with pytest.raises(TypeError):
            embedder.run(text=123)

    @pytest.mark.parametrize(
        "response_body",
        [
            '{"embeddings": [[0.1, 0.2, 0.3]]}',  # embeddings as list of lists
            '{"embeddings": {"float": [[0.1, 0.2, 0.3]]}}',  # embeddings as dict with embedding type as key
        ],
    )
    def test_cohere_invocation(self, mock_boto3_session, response_body):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        client = mock_boto3_session.return_value.client.return_value

        with patch.object(client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.return_value = {
                "body": io.StringIO(response_body),
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
        client = mock_boto3_session.return_value.client.return_value

        with patch.object(client, "invoke_model") as mock_invoke_model:
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

    def test_titan_v2_invocation_with_dimensions_and_normalize(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(
            model="amazon.titan-embed-text-v2:0",
            dimensions=512,
            normalize=False,
        )
        client = mock_boto3_session.return_value.client.return_value

        with patch.object(client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.return_value = {
                "body": io.StringIO('{"embedding": [0.1, 0.2, 0.3]}'),
            }
            embedder.run(text="some text")

            mock_invoke_model.assert_called_once_with(
                body='{"inputText": "some text", "dimensions": 512, "normalize": false}',
                modelId="amazon.titan-embed-text-v2:0",
                accept="*/*",
                contentType="application/json",
            )

    def test_titan_v2_invocation_without_extra_params(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="amazon.titan-embed-text-v2:0")
        client = mock_boto3_session.return_value.client.return_value

        with patch.object(client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.return_value = {
                "body": io.StringIO('{"embedding": [0.1, 0.2, 0.3]}'),
            }
            embedder.run(text="some text")

            mock_invoke_model.assert_called_once_with(
                body='{"inputText": "some text"}',
                modelId="amazon.titan-embed-text-v2:0",
                accept="*/*",
                contentType="application/json",
            )

    def test_titan_v1_ignores_dimensions_and_normalize(self, mock_boto3_session):
        # Titan G1 (v1) does not support `dimensions`/`normalize`, so they must not be sent.
        embedder = AmazonBedrockTextEmbedder(
            model="amazon.titan-embed-text-v1",
            dimensions=512,
            normalize=False,
        )
        client = mock_boto3_session.return_value.client.return_value

        with patch.object(client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.return_value = {
                "body": io.StringIO('{"embedding": [0.1, 0.2, 0.3]}'),
            }
            embedder.run(text="some text")

            mock_invoke_model.assert_called_once_with(
                body='{"inputText": "some text"}',
                modelId="amazon.titan-embed-text-v1",
                accept="*/*",
                contentType="application/json",
            )

    def test_titan_non_text_v2_ignores_dimensions_and_normalize(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(
            model="amazon.titan-embed-image-v2:0",
            dimensions=512,
            normalize=False,
        )
        client = mock_boto3_session.return_value.client.return_value

        with patch.object(client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.return_value = {
                "body": io.StringIO('{"embedding": [0.1, 0.2, 0.3]}'),
            }
            embedder.run(text="some text")

            mock_invoke_model.assert_called_once_with(
                body='{"inputText": "some text"}',
                modelId="amazon.titan-embed-image-v2:0",
                accept="*/*",
                contentType="application/json",
            )

    def test_run_invocation_error(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        client = mock_boto3_session.return_value.client.return_value

        with patch.object(client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.side_effect = ClientError(
                error_response={"Error": {"Code": "some_code", "Message": "some_message"}},
                operation_name="some_operation",
            )

            with pytest.raises(AmazonBedrockInferenceError):
                embedder.run(text="some text")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("AWS_ACCESS_KEY_ID")
        or not os.getenv("AWS_SECRET_ACCESS_KEY")
        or not os.getenv("AWS_DEFAULT_REGION"),
        reason="AWS credentials are not set",
    )
    @pytest.mark.parametrize(
        "model",
        ["cohere.embed-v4:0", "cohere.embed-english-v3", "amazon.titan-embed-text-v1"],
    )
    def test_live_run(self, model):
        embedder = AmazonBedrockTextEmbedder(model=model)

        embedding = embedder.run(text="some text")["embedding"]

        assert isinstance(embedding, list)
        assert len(embedding) > 1000
        assert all(isinstance(embedding, float) for embedding in embedding)


class TestComponentLifecycle:
    def test_client_is_none_after_init(self):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        assert embedder._client is None

    def test_warm_up_uses_resolved_credentials(self, mock_boto3_session, set_env_variables):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        embedder.warm_up()
        mock_boto3_session.assert_called_once_with(
            aws_access_key_id="some_fake_id",
            aws_secret_access_key="some_fake_key",
            aws_session_token="some_fake_token",
            region_name="fake_region",
            profile_name="some_fake_profile",
        )

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_AWS_ACCESS_KEY", raising=False)
        embedder = AmazonBedrockTextEmbedder(
            model="cohere.embed-english-v3",
            aws_access_key_id=Secret.from_env_var("MISSING_AWS_ACCESS_KEY"),
        )
        with pytest.raises(AmazonBedrockConfigurationError):
            embedder.warm_up()

    def test_warm_up_connection_error(self, mock_boto3_session):
        mock_boto3_session.side_effect = Exception("some connection error")
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        with pytest.raises(AmazonBedrockConfigurationError):
            embedder.warm_up()

    def test_sync_lifecycle(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        client = mock_boto3_session.return_value.client.return_value
        embedder.warm_up()
        assert embedder._client is client
        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None
        embedder.warm_up()
        assert mock_boto3_session.call_count == 2

    def test_warm_up_is_idempotent(self, mock_boto3_session):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        embedder.warm_up()
        embedder.warm_up()
        mock_boto3_session.assert_called_once()

    def test_close_is_safe_without_warm_up(self):
        embedder = AmazonBedrockTextEmbedder(model="cohere.embed-english-v3")
        embedder.close()
        assert embedder._client is None
