import glob
import io
import os
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from haystack.components.converters.image.image_utils import _encode_image_to_base64
from haystack.dataclasses import ByteStream, Document

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.components.embedders.amazon_bedrock import AmazonBedrockDocumentImageEmbedder

TYPE = (
    "haystack_integrations.components.embedders.amazon_bedrock."
    "document_image_embedder.AmazonBedrockDocumentImageEmbedder"
)


class TestAmazonBedrockDocumentImageEmbedder:
    def test_init(self, mock_boto3_session, set_env_variables):
        embedder = AmazonBedrockDocumentImageEmbedder(model="cohere.embed-english-v3", input_type="fake_input_type")

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"input_type": "fake_input_type"}
        assert embedder.progress_bar

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
        embedder = AmazonBedrockDocumentImageEmbedder(
            model="cohere.embed-english-v3",
            embedding_types=["float"],
            progress_bar=False,
        )

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"embedding_types": ["float"]}
        assert not embedder.progress_bar

    def test_connection_error(self, mock_boto3_session):
        mock_boto3_session.side_effect = Exception("some connection error")

        with pytest.raises(AmazonBedrockConfigurationError):
            AmazonBedrockDocumentImageEmbedder(
                model="cohere.embed-english-v3",
                embedding_types=["float"],
            )

    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 1000}])
    def test_to_dict(self, mock_boto3_session: Any, boto3_config: Optional[Dict[str, Any]]):
        embedder = AmazonBedrockDocumentImageEmbedder(
            model="cohere.embed-english-v3",
            embedding_types=["float"],
            boto3_config=boto3_config,
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
                "file_path_meta_field": "file_path",
                "embedding_types": ["float"],
                "progress_bar": True,
                "boto3_config": boto3_config,
                "root_path": "",
                "image_size": None,
            },
        }

        assert embedder.to_dict() == expected_dict

    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 1000}])
    def test_from_dict(self, mock_boto3_session: Any, boto3_config: Optional[Dict[str, Any]]):
        data = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "cohere.embed-english-v3",
                "embedding_types": ["float"],
                "root_path": None,
                "progress_bar": True,
                "boto3_config": boto3_config,
                "image_size": None,
            },
        }

        embedder = AmazonBedrockDocumentImageEmbedder.from_dict(data)

        assert embedder.model == "cohere.embed-english-v3"
        assert embedder.kwargs == {"embedding_types": ["float"]}
        assert embedder.progress_bar
        assert embedder.boto3_config == boto3_config

    def test_init_invalid_model(self):
        with pytest.raises(ValueError):
            AmazonBedrockDocumentImageEmbedder(model="")

        with pytest.raises(ValueError):
            AmazonBedrockDocumentImageEmbedder(model="my-unsupported-model")

    def test_run_wrong_type(self, mock_boto3_session):
        embedder = AmazonBedrockDocumentImageEmbedder(model="cohere.embed-english-v3")
        with pytest.raises(TypeError):
            embedder.run(documents="some string")

    def test_run_invocation_error(self, mock_boto3_session, test_files_path):
        embedder = AmazonBedrockDocumentImageEmbedder(model="cohere.embed-english-v3")
        image_paths = glob.glob(str(test_files_path / "*.jpg"))

        with patch.object(embedder._client, "invoke_model") as mock_invoke_model:
            mock_invoke_model.side_effect = ClientError(
                error_response={"Error": {"Code": "some_code", "Message": "some_message"}},
                operation_name="some_operation",
            )

            docs = [
                Document(content="some text", meta={"file_path": image_paths[0]}),
                Document(content="some other text", meta={"file_path": image_paths[0]}),
            ]

            with pytest.raises(AmazonBedrockInferenceError):
                embedder.run(documents=docs)

    def test_embed_cohere(self, mock_boto3_session, test_files_path):
        embedder = AmazonBedrockDocumentImageEmbedder(model="cohere.embed-english-v3", embedding_types=["float"])
        image_paths = glob.glob(str(test_files_path / "*.*"))

        with patch.object(embedder, "_client") as mock_client:
            mock_client.invoke_model.return_value = {
                "body": io.StringIO('{"embeddings": {"float": [[0.1, 0.2, 0.3]]}}'),
            }
            docs = [Document(content="some text", meta={"file_path": image_paths[0]})]
            base64_images = []
            for doc in docs:
                image_byte_stream = ByteStream.from_file_path(filepath=doc.meta["file_path"], mime_type="image/jpeg")
                mime_type, base64_image = _encode_image_to_base64(image_byte_stream)
                base64_images.append(f"data:{mime_type};base64,{base64_image}")

            result = embedder._embed_cohere(image_uris=base64_images)

        mock_client.invoke_model.assert_called_once_with(
            body=f'{{"images": ["{base64_images[0]}"], "input_type": "image", "embedding_types": ["float"]}}',
            modelId="cohere.embed-english-v3",
            accept="*/*",
            contentType="application/json",
        )

        assert result[0] == [0.1, 0.2, 0.3]

    def test_embed_titan(self, mock_boto3_session, test_files_path):
        embedder = AmazonBedrockDocumentImageEmbedder(model="amazon.titan-embed-image-v1")
        image_paths = glob.glob(str(test_files_path / "*.*"))

        mock_response = {
            "body": io.StringIO('{"embedding": [0.1, 0.2, 0.3]}'),
        }

        def mock_invoke_model(*args, **kwargs):
            # since the response body is read in the method, we need to reset the StringIO object
            mock_response["body"].seek(0)
            return mock_response

        with patch.object(embedder, "_client") as mock_client:
            mock_client.invoke_model.side_effect = mock_invoke_model

            docs = [
                Document(content="some text", meta={"file_path": image_paths[0]}),
                Document(content="some other text", meta={"file_path": image_paths[1]}),
            ]
            base64_images = []
            # Process images directly
            for doc in docs:
                image_byte_stream = ByteStream.from_file_path(filepath=doc.meta["file_path"], mime_type="image/jpeg")
                _, base64_image = _encode_image_to_base64(image_byte_stream)
                base64_images.append(base64_image)

            result = embedder._embed_titan(images=base64_images)

        assert mock_client.invoke_model.call_count == 2
        assert mock_client.invoke_model.call_args_list[0][1]["modelId"] == "amazon.titan-embed-image-v1"
        assert mock_client.invoke_model.call_args_list[0][1]["body"] == f'{{"inputImage": "{base64_images[0]}"}}'
        assert mock_client.invoke_model.call_args_list[1][1]["modelId"] == "amazon.titan-embed-image-v1"
        assert mock_client.invoke_model.call_args_list[1][1]["body"] == f'{{"inputImage": "{base64_images[1]}"}}'

        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.1, 0.2, 0.3]

    def test_embed_cohere_multiple_embedding_types(self, mock_boto3_session):
        with pytest.raises(ValueError):
            AmazonBedrockDocumentImageEmbedder(model="cohere.embed-english-v3", embedding_types=["float", "int8"])

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("AWS_ACCESS_KEY_ID")
        or not os.getenv("AWS_SECRET_ACCESS_KEY")
        or not os.getenv("AWS_DEFAULT_REGION"),
        reason="AWS credentials are not set",
    )
    def test_live_run_with_cohere(self, test_files_path):
        embedder = AmazonBedrockDocumentImageEmbedder(model="cohere.embed-english-v3", embedding_types=["int8"])

        image_paths = glob.glob(str(test_files_path / "*.*"))
        documents = []
        for i, path in enumerate(image_paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        result = embedder.run(documents=documents)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc, new_doc in zip(documents, result["documents"]):
            assert doc.embedding is None
            assert new_doc is not doc
            assert isinstance(new_doc, Document)
            assert isinstance(new_doc.embedding, list)
            assert isinstance(new_doc.embedding[0], int)
            assert "embedding_source" not in doc.meta
            assert "embedding_source" in new_doc.meta
            assert new_doc.meta["embedding_source"]["type"] == "image"
            assert "file_path_meta_field" in new_doc.meta["embedding_source"]

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("AWS_ACCESS_KEY_ID")
        or not os.getenv("AWS_SECRET_ACCESS_KEY")
        or not os.getenv("AWS_DEFAULT_REGION"),
        reason="AWS credentials are not set",
    )
    def test_live_run_with_titan(self, test_files_path):
        embedder = AmazonBedrockDocumentImageEmbedder(model="amazon.titan-embed-image-v1", image_size=(100, 100))

        image_paths = glob.glob(str(test_files_path / "*.*"))
        documents = []
        for i, path in enumerate(image_paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        result = embedder.run(documents=documents)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc, new_doc in zip(documents, result["documents"]):
            assert doc.embedding is None
            assert new_doc is not doc
            assert isinstance(new_doc, Document)
            assert isinstance(new_doc.embedding, list)
            assert isinstance(new_doc.embedding[0], float)
            assert "embedding_source" not in doc.meta
            assert "embedding_source" in new_doc.meta
            assert new_doc.meta["embedding_source"]["type"] == "image"
            assert "file_path_meta_field" in new_doc.meta["embedding_source"]
