# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import Mock, patch

import pytest
from haystack import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.jina.document_image_embedder import JinaDocumentImageEmbedder

MOCK_EMBEDDING_DIM = 512


class TestJinaDocumentImageEmbedder:
    def test_init_default(self):
        embedder = JinaDocumentImageEmbedder()
        assert embedder.model_name == "jina-clip-v1"
        assert embedder.file_path_meta_field == "file_path"
        assert embedder.root_path == ""
        assert embedder.dimensions is None
        assert embedder.api_key == Secret.from_env_var("JINA_API_KEY")

    def test_init_with_parameters(self):
        embedder = JinaDocumentImageEmbedder(
            api_key=Secret.from_token("fake-api-token"),
            model="jina-embeddings-v4",
            file_path_meta_field="custom_file_path",
            root_path="/custom/root",
            dimensions=256,
        )
        assert embedder.model_name == "jina-embeddings-v4"
        assert embedder.file_path_meta_field == "custom_file_path"
        assert embedder.root_path == "/custom/root"
        assert embedder.dimensions == 256
        assert embedder.api_key == Secret.from_token("fake-api-token")

    def test_to_dict(self):
        component = JinaDocumentImageEmbedder(
            api_key=Secret.from_env_var("JINA_API_KEY"),
            model="jina-clip-v2",
            file_path_meta_field="image_path",
            root_path="/images",
            dimensions=512,
        )
        data = component.to_dict()
        expected = {
            "type": "haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "jina-clip-v2",
                "file_path_meta_field": "image_path",
                "root_path": "/images",
                "dimensions": 512,
            },
        }
        assert data == expected

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.embedders.jina.document_image_embedder.JinaDocumentImageEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["JINA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "jina-clip-v2",
                "file_path_meta_field": "image_path",
                "root_path": "/images",
                "dimensions": 512,
            },
        }
        component = JinaDocumentImageEmbedder.from_dict(data)
        assert component.model_name == "jina-clip-v2"
        assert component.file_path_meta_field == "image_path"
        assert component.root_path == "/images"
        assert component.dimensions == 512
        assert component.api_key == Secret.from_env_var("JINA_API_KEY")

    def test_run_wrong_input_format(self):
        embedder = JinaDocumentImageEmbedder()
        with patch("haystack_integrations.components.embedders.jina.document_image_embedder.pillow_import"):
            list_integers_input = [1, 2, 3]
            with pytest.raises(TypeError, match="JinaDocumentImageEmbedder expects a list of Documents as input"):
                embedder.run(documents=list_integers_input)

            string_input = "text"
            with pytest.raises(TypeError, match="JinaDocumentImageEmbedder expects a list of Documents as input"):
                embedder.run(documents=string_input)

    def test_run_on_empty_list(self):
        embedder = JinaDocumentImageEmbedder()
        with patch("haystack_integrations.components.embedders.jina.document_image_embedder.pillow_import"):
            empty_list_input = []
            result = embedder.run(documents=empty_list_input)
            assert result == {"documents": []}

    @patch("haystack_integrations.components.embedders.jina.document_image_embedder.pillow_import")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._extract_image_sources_info")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder.Image")
    def test_run_with_successful_request(self, mock_image, mock_extract, _mock_pillow):
        documents = [Document(content="Test image", meta={"file_path": "test.jpg"})]
        mock_extract.return_value = [{"path": "test.jpg", "mime_type": "image/jpeg"}]
        mock_pil_image = Mock()
        mock_image.open.return_value = mock_pil_image

        embedder = JinaDocumentImageEmbedder()
        with patch.object(embedder, "_image_to_base64", return_value="data:image/jpeg;base64,fake_base64"):
            with patch.object(embedder._session, "post") as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "data": [{"embedding": [1.0] * MOCK_EMBEDDING_DIM}],
                    "model": "jina-clip-v1",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
                mock_post.return_value = mock_response
                result = embedder.run(documents=documents)

                assert "documents" in result
                assert len(result["documents"]) == 1
                assert result["documents"][0].embedding == [1.0] * MOCK_EMBEDDING_DIM
                assert result["documents"][0].meta["embedding_source"]["type"] == "image"

    @patch("haystack_integrations.components.embedders.jina.document_image_embedder.pillow_import")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder._extract_image_sources_info")
    @patch("haystack_integrations.components.embedders.jina.document_image_embedder.Image")
    def test_run_with_api_error(self, mock_image, mock_extract, _mock_pillow):
        documents = [Document(content="Test image", meta={"file_path": "test.jpg"})]
        mock_extract.return_value = [{"path": "test.jpg", "mime_type": "image/jpeg"}]
        mock_pil_image = Mock()
        mock_image.open.return_value = mock_pil_image

        embedder = JinaDocumentImageEmbedder()
        with patch.object(embedder, "_image_to_base64", return_value="data:image/jpeg;base64,fake_base64"):
            with patch.object(embedder._session, "post") as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {"detail": "API Error occurred"}
                mock_post.return_value = mock_response
                with pytest.raises(RuntimeError, match="Jina API error: API Error occurred"):
                    embedder.run(documents=documents)

    def test_image_to_base64(self):
        embedder = JinaDocumentImageEmbedder()
        with patch("haystack_integrations.components.embedders.jina.document_image_embedder.pillow_import"):
            mock_pil_image = Mock()
            mock_pil_image.mode = "RGB"
            mock_pil_image.save = Mock()

            with patch("base64.b64encode") as mock_b64encode:
                mock_b64encode.return_value = b"fake_base64"
                result = embedder._image_to_base64(mock_pil_image)

                assert result == "data:image/jpeg;base64,fake_base64"
                mock_pil_image.save.assert_called_once()

    def test_get_telemetry_data(self):
        embedder = JinaDocumentImageEmbedder(model="jina-embeddings-v4")
        with patch("haystack_integrations.components.embedders.jina.document_image_embedder.pillow_import"):
            telemetry_data = embedder._get_telemetry_data()
            assert telemetry_data == {"model": "jina-embeddings-v4"}
