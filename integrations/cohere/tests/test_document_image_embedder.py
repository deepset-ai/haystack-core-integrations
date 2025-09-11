# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cohere.types import EmbedByTypeResponse, EmbedByTypeResponseEmbeddings
from haystack import Document

from haystack_integrations.components.embedders.cohere.document_image_embedder import (
    CohereDocumentImageEmbedder,
)
from haystack_integrations.components.embedders.cohere.embedding_types import EmbeddingTypes

IMPORT_PATH = "haystack_integrations.components.embedders.cohere.document_image_embedder"


class TestCohereDocumentImageEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder()

        assert embedder.file_path_meta_field == "file_path"
        assert embedder.root_path == ""
        assert embedder.model == "embed-v4.0"
        assert embedder.image_size is None
        assert embedder.progress_bar is True
        assert embedder.embedding_dimension is None
        assert embedder.embedding_type == EmbeddingTypes.FLOAT

        assert embedder._api_base_url == "https://api.cohere.com"
        assert embedder._timeout == 120
        assert embedder._api_key.resolve_value() == "test-api-key"
        assert embedder._client is not None
        assert embedder._async_client is not None

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(
            file_path_meta_field="custom_file_path",
            root_path="root_path",
            image_size=(100, 100),
            model="model",
            api_base_url="https://api.cohere.com/v1",
            timeout=300,
            progress_bar=False,
            embedding_dimension=256,
            embedding_type=EmbeddingTypes.INT8,
        )
        assert embedder.file_path_meta_field == "custom_file_path"
        assert embedder.root_path == "root_path"
        assert embedder.model == "model"
        assert embedder.image_size == (100, 100)
        assert embedder.progress_bar is False
        assert embedder.embedding_dimension == 256
        assert embedder.embedding_type == EmbeddingTypes.INT8
        assert embedder._api_base_url == "https://api.cohere.com/v1"
        assert embedder._timeout == 300
        assert embedder._api_key.resolve_value() == "test-api-key"
        assert embedder._client is not None
        assert embedder._async_client is not None

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component = CohereDocumentImageEmbedder(
            model="model",
            api_base_url="https://api.cohere.com/v1",
            timeout=300,
            progress_bar=False,
            embedding_dimension=256,
            embedding_type=EmbeddingTypes.INT8,
        )
        data = component.to_dict()
        assert data == {
            "type": f"{IMPORT_PATH}.CohereDocumentImageEmbedder",
            "init_parameters": {
                "file_path_meta_field": "file_path",
                "root_path": "",
                "model": "model",
                "image_size": None,
                "progress_bar": False,
                "embedding_dimension": 256,
                "embedding_type": "int8",
                "api_base_url": "https://api.cohere.com/v1",
                "timeout": 300,
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        init_parameters = {
            "file_path_meta_field": "custom_file_path",
            "root_path": "root_path",
            "model": "model",
            "image_size": (100, 100),
            "progress_bar": False,
            "embedding_dimension": 256,
            "embedding_type": "int8",
            "api_base_url": "https://api.cohere.com/v1",
            "timeout": 300,
            "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
        }
        component = CohereDocumentImageEmbedder.from_dict(
            {"type": f"{IMPORT_PATH}.CohereDocumentImageEmbedder", "init_parameters": init_parameters}
        )
        assert component.file_path_meta_field == "custom_file_path"
        assert component.root_path == "root_path"
        assert component.model == "model"
        assert component.image_size == (100, 100)
        assert component.progress_bar is False
        assert component.embedding_dimension == 256
        assert component.embedding_type == EmbeddingTypes.INT8
        assert component._api_base_url == "https://api.cohere.com/v1"
        assert component._timeout == 300
        assert component._api_key.resolve_value() == "test-api-key"
        assert component._client is not None
        assert component._async_client is not None

    def test_extract_images_to_embed_wrong_input_format(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(model="model")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="CohereDocumentImageEmbedder expects a list of Documents as input"):
            embedder._extract_images_to_embed(string_input)

        with pytest.raises(TypeError, match="CohereDocumentImageEmbedder expects a list of Documents as input"):
            embedder._extract_images_to_embed(list_integers_input)

    @patch(f"{IMPORT_PATH}._extract_image_sources_info")
    def test_extract_images_to_embed_unsupported_image_mime_type(self, mocked_extract_image_sources_info, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(model="model")

        mocked_extract_image_sources_info.return_value = [
            {"path": "unsupported.txt", "mime_type": "text/plain"},
        ]

        documents = [
            Document(content="Doc with unsupported mime type", meta={"file_path": "unsupported.txt"}),
        ]

        with pytest.raises(ValueError, match="Unsupported image MIME type"):
            embedder._extract_images_to_embed(documents)

    @patch(f"{IMPORT_PATH}._extract_image_sources_info")
    @patch(f"{IMPORT_PATH}._batch_convert_pdf_pages_to_images")
    @patch(f"{IMPORT_PATH}._encode_image_to_base64")
    @patch(f"{IMPORT_PATH}.ByteStream.from_file_path")
    def test_extract_images_to_embed_none_images(
        self,
        mocked_byte_stream_from_file_path,
        mocked_encode_image_to_base64,
        mocked_batch_convert_pdf_pages_to_images,
        mocked_extract_image_sources_info,
        monkeypatch,
    ):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(model="model")

        mocked_extract_image_sources_info.return_value = [
            {"path": "pdf_1.pdf", "mime_type": "application/pdf", "page_number": 999},  # Page 999 doesn't exist
            {"path": "image_1.jpg", "mime_type": "image/jpeg"},
        ]
        mocked_batch_convert_pdf_pages_to_images.return_value = {}  # Empty dict because page was skipped
        mocked_encode_image_to_base64.return_value = ("image/jpeg", "base64_image")
        mocked_byte_stream_from_file_path.return_value = MagicMock()

        documents = [
            Document(content="PDF 1", meta={"file_path": "pdf_1.pdf", "page_number": 999}),
            Document(content="Image 1", meta={"file_path": "image_1.jpg"}),
        ]

        with pytest.raises(RuntimeError, match=r"Conversion failed for some documents\."):
            embedder._extract_images_to_embed(documents)

    @patch(f"{IMPORT_PATH}._extract_image_sources_info")
    @patch(f"{IMPORT_PATH}._batch_convert_pdf_pages_to_images")
    @patch(f"{IMPORT_PATH}._encode_image_to_base64")
    @patch(f"{IMPORT_PATH}.ByteStream.from_file_path")
    def test_extract_images_to_embed_success(
        self,
        mocked_byte_stream_from_file_path,
        mocked_encode_image_to_base64,
        mocked_batch_convert_pdf_pages_to_images,
        mocked_extract_image_sources_info,
        monkeypatch,
    ):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(model="model")

        mocked_extract_image_sources_info.return_value = [
            {"path": "pdf_1.pdf", "mime_type": "application/pdf", "page_number": 1},
            {"path": "image_1.jpg", "mime_type": "image/jpeg"},
        ]
        mocked_batch_convert_pdf_pages_to_images.return_value = {0: "base64_pdf_image"}
        mocked_encode_image_to_base64.return_value = ("image/jpeg", "base64_image")
        mocked_byte_stream_from_file_path.return_value = MagicMock()

        documents = [
            Document(content="PDF 1", meta={"file_path": "pdf_1.pdf", "page_number": 1}),
            Document(content="Image 1", meta={"file_path": "image_1.jpg"}),
        ]

        result = embedder._extract_images_to_embed(documents)

        assert len(result) == 2
        assert result[0] == "data:image/jpeg;base64,base64_pdf_image"
        assert result[1] == "data:image/jpeg;base64,base64_image"

    def test_run(self, test_files_path, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(model="model")
        embedder._client = MagicMock()

        mock_response = EmbedByTypeResponse(
            id="test-id",
            embeddings=EmbedByTypeResponseEmbeddings(float_=[[random.random() for _ in range(1536)]]),  # noqa: S311
            meta=None,
        )

        embedder._client.embed.return_value = mock_response

        image_paths = glob.glob(str(test_files_path / "*.jpg")) + glob.glob(str(test_files_path / "*.pdf"))
        assert len(image_paths) == 2
        assert image_paths[0].endswith(".jpg")
        assert image_paths[1].endswith(".pdf")

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

    def test_run_client_errors(self, test_files_path, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(model="model")
        embedder._client = MagicMock()
        embedder._client.embed.side_effect = Exception("Error embedding image")

        image_paths = glob.glob(str(test_files_path / "*.jpg")) + glob.glob(str(test_files_path / "*.pdf"))
        assert len(image_paths) == 2
        assert image_paths[0].endswith(".jpg")
        assert image_paths[1].endswith(".pdf")

        documents = []
        for i, path in enumerate(image_paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        with pytest.raises(RuntimeError, match="Error embedding Document"):
            embedder.run(documents=documents)

    @pytest.mark.asyncio
    async def test_run_async(self, test_files_path, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(model="model")
        embedder._async_client = AsyncMock()

        mock_response = EmbedByTypeResponse(
            id="test-id",
            embeddings=EmbedByTypeResponseEmbeddings(float_=[[random.random() for _ in range(1536)]]),  # noqa: S311
            meta=None,
        )

        embedder._async_client.embed.return_value = mock_response

        image_paths = glob.glob(str(test_files_path / "*.jpg")) + glob.glob(str(test_files_path / "*.pdf"))
        assert len(image_paths) == 2
        assert image_paths[0].endswith(".jpg")
        assert image_paths[1].endswith(".pdf")

        documents = []
        for i, path in enumerate(image_paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        result = await embedder.run_async(documents=documents)

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

    @pytest.mark.asyncio
    async def test_run_async_client_errors(self, test_files_path, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentImageEmbedder(model="model")
        embedder._async_client = AsyncMock()
        embedder._async_client.embed.side_effect = Exception("Error embedding image")

        image_paths = glob.glob(str(test_files_path / "*.jpg")) + glob.glob(str(test_files_path / "*.pdf"))
        assert len(image_paths) == 2
        assert image_paths[0].endswith(".jpg")
        assert image_paths[1].endswith(".pdf")

        documents = []
        for i, path in enumerate(image_paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        with pytest.raises(RuntimeError, match="Error embedding Document"):
            await embedder.run_async(documents=documents)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    def test_live_run(self, test_files_path):
        embedder = CohereDocumentImageEmbedder(model="embed-v4.0", image_size=(100, 100))

        documents = [
            Document(
                content="PDF document",
                meta={"file_path": str(test_files_path / "sample_pdf_1.pdf"), "page_number": 1},
            ),
            Document(content="Image document", meta={"file_path": str(test_files_path / "apple.jpg")}),
        ]

        result = embedder.run(documents=documents)
        assert len(result["documents"]) == len(documents)
        for doc, new_doc in zip(documents, result["documents"]):
            assert doc.embedding is None
            assert new_doc is not doc
            assert isinstance(new_doc, Document)
            assert isinstance(new_doc.embedding, list)
            assert len(new_doc.embedding) == 1536
            assert all(isinstance(x, float) for x in new_doc.embedding)
            assert "embedding_source" not in doc.meta
            assert "embedding_source" in new_doc.meta
            assert new_doc.meta["embedding_source"]["type"] == "image"
            assert "file_path_meta_field" in new_doc.meta["embedding_source"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    async def test_live_run_async(self, test_files_path):
        embedder = CohereDocumentImageEmbedder(model="embed-v4.0", image_size=(100, 100))

        documents = [
            Document(
                content="PDF document",
                meta={"file_path": str(test_files_path / "sample_pdf_1.pdf"), "page_number": 1},
            ),
            Document(content="Image document", meta={"file_path": str(test_files_path / "apple.jpg")}),
        ]

        result = await embedder.run_async(documents=documents)
        assert len(result["documents"]) == len(documents)
        for doc, new_doc in zip(documents, result["documents"]):
            assert doc.embedding is None
            assert new_doc is not doc
            assert isinstance(new_doc, Document)
            assert isinstance(new_doc.embedding, list)
            assert len(new_doc.embedding) == 1536
            assert all(isinstance(x, float) for x in new_doc.embedding)
            assert "embedding_source" not in doc.meta
            assert "embedding_source" in new_doc.meta
            assert new_doc.meta["embedding_source"]["type"] == "image"
            assert "file_path_meta_field" in new_doc.meta["embedding_source"]
