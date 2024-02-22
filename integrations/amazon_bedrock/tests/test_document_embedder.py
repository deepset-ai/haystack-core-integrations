import io
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from haystack.dataclasses import Document

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.components.embedders.amazon_bedrock import AmazonBedrockDocumentEmbedder

TYPE = "haystack_integrations.components.embedders.amazon_bedrock.document_embedder.AmazonBedrockDocumentEmbedder"


class TestAmazonBedrockDocumentEmbedder:
    def test_init(self, mock_boto3_session, set_env_variables):
        embedder = AmazonBedrockDocumentEmbedder(
            model="cohere.embed-english-v3",
            input_type="fake_input_type",
        )

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"input_type": "fake_input_type"}
        assert embedder.batch_size == 32
        assert embedder.progress_bar
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

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

    def test_init_custom_parameters(self, mock_boto3_session, set_env_variables):
        embedder = AmazonBedrockDocumentEmbedder(
            model="cohere.embed-english-v3",
            input_type="fake_input_type",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["meta1", "meta2"],
            embedding_separator=" ",
        )

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"input_type": "fake_input_type"}
        assert embedder.batch_size == 64
        assert not embedder.progress_bar
        assert embedder.meta_fields_to_embed == ["meta1", "meta2"]
        assert embedder.embedding_separator == " "

    def test_connection_error(self, mock_boto3_session):
        mock_boto3_session.side_effect = Exception("some connection error")

        with pytest.raises(AmazonBedrockConfigurationError):
            AmazonBedrockDocumentEmbedder(
                model="cohere.embed-english-v3",
                input_type="fake_input_type",
            )

    def test_to_dict(self, mock_boto3_session):
        embedder = AmazonBedrockDocumentEmbedder(
            model="cohere.embed-english-v3",
            input_type="search_document",
        )

        expected_dict = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "cohere.embed-english-v3",
                "input_type": "search_document",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

        assert embedder.to_dict() == expected_dict

    def test_from_dict(self, mock_boto3_session):
        data = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "cohere.embed-english-v3",
                "input_type": "search_document",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

        embedder = AmazonBedrockDocumentEmbedder.from_dict(data)

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"input_type": "search_document"}
        assert embedder.batch_size == 32
        assert embedder.progress_bar
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_invalid_model(self):
        with pytest.raises(ValueError):
            AmazonBedrockDocumentEmbedder(model="")

        with pytest.raises(ValueError):
            AmazonBedrockDocumentEmbedder(model="my-unsupported-model")

    def test_run_wrong_type(self, mock_boto3_session):
        embedder = AmazonBedrockDocumentEmbedder(model="cohere.embed-english-v3")
        with pytest.raises(TypeError):
            embedder.run(documents="some string")

    def test_run_invocation_error(self, mock_boto3_session):
        embedder = AmazonBedrockDocumentEmbedder(model="cohere.embed-english-v3")

        with patch.object(embedder._client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.side_effect = ClientError(
                error_response={"Error": {"Code": "some_code", "Message": "some_message"}},
                operation_name="some_operation",
            )

            docs = [Document(content="some text"), Document(content="some other text")]

            with pytest.raises(AmazonBedrockInferenceError):
                embedder.run(documents=docs)

    def test_prepare_texts_to_embed_w_metadata(self, mock_boto3_session):
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = AmazonBedrockDocumentEmbedder(
            model="cohere.embed-english-v3", meta_fields_to_embed=["meta_field"], embedding_separator=" | "
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "meta_value 0 | document number 0: content",
            "meta_value 1 | document number 1: content",
            "meta_value 2 | document number 2: content",
            "meta_value 3 | document number 3: content",
            "meta_value 4 | document number 4: content",
        ]

    def test_embed_cohere(self, mock_boto3_session):
        embedder = AmazonBedrockDocumentEmbedder(model="cohere.embed-english-v3")

        with patch.object(embedder, "_client") as mock_client:
            mock_client.invoke_model.return_value = {
                "body": io.StringIO('{"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}'),
            }

            docs = [Document(content="some text"), Document(content="some other text")]

            result = embedder._embed_cohere(documents=docs)

        mock_client.invoke_model.assert_called_once_with(
            body='{"texts": ["some text", "some other text"], "input_type": "search_document"}',
            modelId="cohere.embed-english-v3",
            accept="*/*",
            contentType="application/json",
        )

        assert result[0].content == "some text"
        assert result[1].content == "some other text"
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[1].embedding == [0.4, 0.5, 0.6]

    def test_embed_cohere_batching(self, mock_boto3_session):
        embedder = AmazonBedrockDocumentEmbedder(model="cohere.embed-english-v3", batch_size=2)

        mock_response = {
            "body": io.StringIO('{"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}'),
        }

        def mock_invoke_model(*args, **kwargs):
            # since the response body is read in the method, we need to reset the StringIO object
            mock_response["body"].seek(0)
            return mock_response

        with patch.object(embedder, "_client") as mock_client:
            mock_client.invoke_model.side_effect = mock_invoke_model

            docs = [
                Document(content="some text"),
                Document(content="some other text"),
                Document(content="yet another text"),
                Document(content="and another text"),
            ]

            result = embedder._embed_cohere(documents=docs)

        assert mock_client.invoke_model.call_count == 2
        for i, doc in enumerate(result):
            assert doc.content == docs[i].content
            assert doc.embedding == [0.1, 0.2, 0.3] if i % 2 == 0 else [0.4, 0.5, 0.6]

    def test_embed_titan(self, mock_boto3_session):
        embedder = AmazonBedrockDocumentEmbedder(model="amazon.titan-embed-text-v1")

        mock_response = {
            "body": io.StringIO('{"embedding": [0.1, 0.2, 0.3]}'),
        }

        def mock_invoke_model(*args, **kwargs):
            # since the response body is read in the method, we need to reset the StringIO object
            mock_response["body"].seek(0)
            return mock_response

        with patch.object(embedder, "_client") as mock_client:
            mock_client.invoke_model.side_effect = mock_invoke_model

            docs = [Document(content="some text"), Document(content="some other text")]

            result = embedder._embed_titan(documents=docs)

        assert mock_client.invoke_model.call_count == 2
        assert mock_client.invoke_model.call_args_list[0][1]["modelId"] == "amazon.titan-embed-text-v1"
        assert mock_client.invoke_model.call_args_list[0][1]["body"] == '{"inputText": "some text"}'
        assert mock_client.invoke_model.call_args_list[1][1]["modelId"] == "amazon.titan-embed-text-v1"
        assert mock_client.invoke_model.call_args_list[1][1]["body"] == '{"inputText": "some other text"}'

        for i, doc in enumerate(result):
            assert doc.content == docs[i].content
            assert doc.embedding == [0.1, 0.2, 0.3]
