# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from haystack import Document
from haystack.utils.auth import Secret
from PIL import Image

from haystack_integrations.components.embedders.jina.document_image_embedder import JinaDocumentImageEmbedder

MOCK_EMBEDDING_DIM = 512


class TestJinaDocumentImageEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        embedder = JinaDocumentImageEmbedder()
        assert embedder.model_name == "jina-clip-v2"
        assert embedder.file_path_meta_field == "file_path"
        assert embedder.root_path == ""
        assert embedder.embedding_dimension is None
        assert embedder.image_size is None
        assert embedder.batch_size == 5
        assert embedder.api_key == Secret.from_env_var("JINA_API_KEY")

    def test_init_with_parameters(self):
        embedder = JinaDocumentImageEmbedder(
            api_key=Secret.from_token("fake-api-token"),
            model="jina-embeddings-v4",
            file_path_meta_field="custom_file_path",
            root_path="/custom/root",
            embedding_dimension=256,
            image_size=(512, 512),
            batch_size=5,
        )
        assert embedder.model_name == "jina-embeddings-v4"
        assert embedder.file_path_meta_field == "custom_file_path"
        assert embedder.root_path == "/custom/root"
        assert embedder.embedding_dimension == 256
        assert embedder.image_size == (512, 512)
        assert embedder.batch_size == 5
        assert embedder.api_key == Secret.from_token("fake-api-token")

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        component = JinaDocumentImageEmbedder(
            api_key=Secret.from_env_var("JINA_API_KEY"),
            model="jina-clip-v2",
            file_path_meta_field="image_path",
            root_path="/images",
            embedding_dimension=512,
            image_size=(256, 256),
            batch_size=5,
        )
        data = component.to_dict()
        expected = {
            "type": "haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "jina-clip-v2",
                "file_path_meta_field": "image_path",
                "root_path": "/images",
                "embedding_dimension": 512,
                "image_size": (256, 256),
                "batch_size": 5,
            },
        }
        assert data == expected

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "jina-clip-v2",
                "file_path_meta_field": "image_path",
                "root_path": "/images",
                "embedding_dimension": 512,
                "image_size": (256, 256),
                "batch_size": 5,
            },
        }
        component = JinaDocumentImageEmbedder.from_dict(data)
        assert component.model_name == "jina-clip-v2"
        assert component.file_path_meta_field == "image_path"
        assert component.root_path == "/images"
        assert component.embedding_dimension == 512
        assert component.image_size == (256, 256)
        assert component.batch_size == 5
        assert component.api_key == Secret.from_env_var("JINA_API_KEY")

    def test_run_wrong_input_format(self):
        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"))
        list_integers_input = [1, 2, 3]
        with pytest.raises(TypeError, match="JinaDocumentImageEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

        string_input = "text"
        with pytest.raises(TypeError, match="JinaDocumentImageEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

    def test_run_on_empty_list(self):
        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"))
        empty_list_input = []
        result = embedder.run(documents=empty_list_input)
        assert result == {"documents": []}

    def test_run_with_successful_request(self):
        documents = [Document(content="Test image", meta={"file_path": "test.jpg"})]

        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"))

        # Mock the _extract_images_to_embed method to return base64 image data
        with patch.object(embedder, "_extract_images_to_embed", return_value=["data:image/jpeg;base64,fake_base64"]):
            with patch.object(embedder._session, "post") as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "data": [{"embedding": [1.0] * MOCK_EMBEDDING_DIM}],
                    "model": "jina-clip-v2",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
                mock_post.return_value = mock_response
                result = embedder.run(documents=documents)

                assert "documents" in result
                assert len(result["documents"]) == 1
                assert result["documents"][0].embedding == [1.0] * MOCK_EMBEDDING_DIM
                assert result["documents"][0].meta["embedding_source"]["type"] == "image"

    def test_run_with_api_error(self):
        documents = [Document(content="Test image", meta={"file_path": "test.jpg"})]

        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"))

        # Mock the _extract_images_to_embed method to return base64 image data
        with patch.object(embedder, "_extract_images_to_embed", return_value=["data:image/jpeg;base64,fake_base64"]):
            with patch.object(embedder._session, "post") as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {"detail": "API Error occurred"}
                mock_post.return_value = mock_response
                with pytest.raises(RuntimeError, match="Jina API error: API Error occurred"):
                    embedder.run(documents=documents)

    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._extract_image_sources_info")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._batch_convert_pdf_pages_to_images")
    def test_extract_images_to_embed_none_images(self, mock_batch_convert, mock_extract_info):
        documents = [Document(content="Test image", meta={"file_path": "test.jpg"})]
        mock_extract_info.return_value = [{"path": "test.jpg", "mime_type": "image/jpeg"}]
        mock_batch_convert.return_value = {}

        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"))

        # Mock ByteStream to simulate complete failure
        mock_path = "haystack_integrations.components.embedders.jina.document_image_embedder.ByteStream"
        with patch(mock_path) as mock_bytestream:
            mock_bytestream.from_file_path.side_effect = Exception("File not found")
            with pytest.raises(Exception, match="File not found"):
                embedder._extract_images_to_embed(documents)

    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._extract_image_sources_info")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder.ByteStream")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._encode_image_to_base64")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._batch_convert_pdf_pages_to_images")
    def test_extract_images_to_embed_conversion_failure(
        self, mock_batch_convert, mock_encode, mock_bytestream, mock_extract_info
    ):
        documents = [Document(content="Test image", meta={"file_path": "test.jpg"})]
        mock_extract_info.return_value = [{"path": "test.jpg", "mime_type": "image/jpeg"}]
        mock_bytestream.from_file_path.return_value = Mock()
        mock_encode.side_effect = Exception("Encoding failed")
        mock_batch_convert.return_value = {}

        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(Exception, match="Encoding failed"):
            embedder._extract_images_to_embed(documents)

    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._extract_image_sources_info")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder.ByteStream")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._encode_image_to_base64")
    def test_extract_images_to_embed_success(self, mock_encode, mock_bytestream, mock_extract_info):
        documents = [Document(content="Test image", meta={"file_path": "test.jpg"})]
        mock_extract_info.return_value = [{"path": "test.jpg", "mime_type": "image/jpeg"}]
        mock_bytestream.from_file_path.return_value = Mock()
        mock_encode.return_value = ("image/jpeg", "fake_base64_data")

        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"))
        result = embedder._extract_images_to_embed(documents)

        assert result == ["data:image/jpeg;base64,fake_base64_data"]

    def test_run_with_connection_error(self):
        documents = [Document(content="Test image", meta={"file_path": "test.jpg"})]
        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"))

        with patch.object(embedder, "_extract_images_to_embed", return_value=["data:image/jpeg;base64,fake_base64"]):
            with patch.object(embedder._session, "post", side_effect=Exception("Connection failed")):
                with pytest.raises(RuntimeError, match="Error calling Jina API: Connection failed"):
                    embedder.run(documents=documents)

    def test_run_with_batch_processing(self):
        # Test with more documents than batch size
        documents = [Document(content=f"Test image {i}", meta={"file_path": f"test{i}.jpg"}) for i in range(12)]
        embedder = JinaDocumentImageEmbedder(api_key=Secret.from_token("fake-api-key"), batch_size=5)

        fake_images = [f"data:image/jpeg;base64,fake_base64_{i}" for i in range(12)]

        with patch.object(embedder, "_extract_images_to_embed", return_value=fake_images):
            with patch.object(embedder._session, "post") as mock_post:
                # Mock response that adapts to batch size
                def mock_response_func(*_args, **kwargs):
                    batch_size = len(kwargs["json"]["input"])
                    mock_response = Mock()
                    mock_response.json.return_value = {
                        "data": [{"embedding": [1.0] * MOCK_EMBEDDING_DIM} for _ in range(batch_size)]
                    }
                    return mock_response

                mock_post.side_effect = mock_response_func

                result = embedder.run(documents=documents)

                # Should have made 3 API calls (12 images / 5 batch_size = 3 batches: 5, 5, 2)
                assert mock_post.call_count == 3
                assert len(result["documents"]) == 12

                # Check batch sizes: first two should be 5, last should be 2
                call_args_list = mock_post.call_args_list
                assert len(call_args_list[0][1]["json"]["input"]) == 5  # First batch: 5 images
                assert len(call_args_list[1][1]["json"]["input"]) == 5  # Second batch: 5 images
                assert len(call_args_list[2][1]["json"]["input"]) == 2  # Third batch: 2 images

    @pytest.mark.skipif(not os.environ.get("JINA_API_KEY", None), reason="JINA_API_KEY not set")
    @pytest.mark.integration
    def test_run_integration(self):
        """Integration test for JinaDocumentImageEmbedder."""
        embedder = JinaDocumentImageEmbedder(model="jina-clip-v2")

        # Create a simple test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            # Create a simple RGB image (100x100 red square)
            test_image = Image.new("RGB", (100, 100), color="red")
            test_image.save(tmp_file.name, "JPEG")

            # Create document with the test image
            documents = [Document(content="A red square", meta={"file_path": tmp_file.name})]

            # Run the embedder
            result = embedder.run(documents=documents)

            # Verify the results
            assert "documents" in result
            assert len(result["documents"]) == 1

            doc = result["documents"][0]
            assert doc.embedding is not None
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0  # Should have embedding dimensions
            assert all(isinstance(x, (int, float)) for x in doc.embedding)
            assert doc.meta["embedding_source"]["type"] == "image"

            # Clean up
            os.unlink(tmp_file.name)
