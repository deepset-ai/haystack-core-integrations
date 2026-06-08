# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import glob
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.google_genai import GoogleGenAIMultimodalDocumentEmbedder
from haystack_integrations.components.embedders.google_genai.multimodal_document_embedder import _extract_sources_info


class TestExtractSourcesInfo:
    def test_extract_source_info(self, test_files_path):
        paths = (
            glob.glob(str(test_files_path / "*.jpg"))
            + glob.glob(str(test_files_path / "*.png"))
            + glob.glob(str(test_files_path / "*.pdf"))
        )

        documents = []
        for i, path in enumerate(paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        sources_info = _extract_sources_info(documents=documents, file_path_meta_field="file_path", root_path="")
        assert len(sources_info) == len(documents)

        for source_info in sources_info:
            assert str(source_info["path"]) in paths
            assert source_info["mime_type"] in ["image/jpeg", "image/png", "application/pdf"]
            if source_info["mime_type"] == "application/pdf":
                assert source_info.get("page_number") == 1
            else:
                assert "page_number" not in source_info

    def test_extract_source_info_pdf_wo_page_number(self, test_files_path):
        document = Document(content="test", meta={"file_path": str(test_files_path / "sample_pdf_3.pdf")})
        sources_info = _extract_sources_info(documents=[document], file_path_meta_field="file_path", root_path="")
        assert len(sources_info) == 1
        assert sources_info[0]["mime_type"] == "application/pdf"
        assert sources_info[0]["send_raw"] is True
        assert sources_info[0].get("page_number") is None

    def test_extract_sources_info_errors(self, test_files_path):
        document = Document(content="test")
        with pytest.raises(ValueError, match="missing the 'file_path' key"):
            _extract_sources_info(documents=[document], file_path_meta_field="file_path", root_path="")

        document = Document(content="test", meta={"file_path": "invalid_path"})
        with pytest.raises(ValueError, match="has an invalid file path"):
            _extract_sources_info(documents=[document], file_path_meta_field="file_path", root_path="")

        document = Document(content="test", meta={"file_path": str(test_files_path / "sample_docx.docx")})
        with pytest.raises(ValueError, match="has an unsupported MIME type"):
            _extract_sources_info(documents=[document], file_path_meta_field="file_path", root_path="")


class TestGoogleGenAIMultimodalDocumentEmbedder:
    def test_init_with_parameters(self):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key-2"),
            model="model",
            batch_size=64,
            progress_bar=False,
            file_path_meta_field="file_path",
            root_path="root_path",
            image_size=(1024, 1024),
            config={"task_type": "CLASSIFICATION"},
        )
        assert embedder._api_key.resolve_value() == "fake-api-key-2"
        assert embedder._model == "model"
        assert embedder._file_path_meta_field == "file_path"
        assert embedder._root_path == "root_path"
        assert embedder._image_size == (1024, 1024)
        assert embedder._batch_size == 64
        assert embedder._progress_bar is False
        assert embedder._config == {"task_type": "CLASSIFICATION"}

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="you must export the GOOGLE_API_KEY or GEMINI_API_KEY"):
            GoogleGenAIMultimodalDocumentEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        component = GoogleGenAIMultimodalDocumentEmbedder()

        data = component_to_dict(component, "embedder")

        assert data == {
            "type": (
                "haystack_integrations.components.embedders.google_genai.multimodal_document_embedder.GoogleGenAIMultimodalDocumentEmbedder"
            ),
            "init_parameters": {
                "model": "gemini-embedding-2",
                "file_path_meta_field": "file_path",
                "root_path": None,
                "image_size": None,
                "batch_size": 6,
                "progress_bar": True,
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "strict": False},
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
                "config": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        data = {
            "type": (
                "haystack_integrations.components.embedders.google_genai.multimodal_document_embedder.GoogleGenAIMultimodalDocumentEmbedder"
            ),
            "init_parameters": {
                "model": "gemini-embedding-2",
                "file_path_meta_field": "file_path",
                "root_path": "some_root_path",
                "image_size": (1024, 1024),
                "batch_size": 12,
                "progress_bar": True,
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "strict": False},
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
                "config": {"task_type": "CLASSIFICATION"},
            },
        }
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")

        embedder = component_from_dict(GoogleGenAIMultimodalDocumentEmbedder, data, "embedder")
        assert embedder._api_key.resolve_value() == "fake-api-key"
        assert embedder._model == "gemini-embedding-2"
        assert embedder._file_path_meta_field == "file_path"
        assert embedder._root_path == "some_root_path"
        assert embedder._image_size == (1024, 1024)
        assert embedder._batch_size == 12
        assert embedder._progress_bar is True
        assert embedder._config == {"task_type": "CLASSIFICATION"}
        assert embedder._api == "gemini"
        assert embedder._vertex_ai_project is None
        assert embedder._vertex_ai_location is None

    def test_extract_parts_to_embed_images(self, test_files_path):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))
        docs = [
            Document(content=None, meta={"file_path": str(test_files_path / "apple.jpg")}),
            Document(content=None, meta={"file_path": str(test_files_path / "banana.png")}),
        ]
        parts = embedder._extract_parts_to_embed(docs)
        assert len(parts) == 2
        for part in parts:
            assert isinstance(part, types.Part)

    def test_extract_parts_to_embed_pdf_raw(self, test_files_path):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))
        docs = [Document(content=None, meta={"file_path": str(test_files_path / "sample_pdf_3.pdf")})]
        parts = embedder._extract_parts_to_embed(docs)
        assert len(parts) == 1
        assert isinstance(parts[0], types.Part)

    @patch(
        "haystack_integrations.components.embedders.google_genai.multimodal_document_embedder._batch_convert_pdf_pages_to_images"
    )
    def test_extract_parts_to_embed_pdf_with_page(self, mock_convert, test_files_path):
        fake_b64 = base64.b64encode(b"fake-image-data").decode()
        mock_convert.return_value = {0: fake_b64}

        embedder = GoogleGenAIMultimodalDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))
        docs = [Document(content=None, meta={"file_path": str(test_files_path / "sample_pdf_3.pdf"), "page_number": 1})]
        parts = embedder._extract_parts_to_embed(docs)
        assert len(parts) == 1
        assert isinstance(parts[0], types.Part)
        mock_convert.assert_called_once()

    def test_embed_batch_no_embeddings_response(self, caplog):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(api_key=Secret.from_token("fake-api-key"), progress_bar=False)

        mock_response = MagicMock()
        mock_response.embeddings = None
        embedder._client.models.embed_content = MagicMock(return_value=mock_response)

        parts = [MagicMock(spec=types.Part)]
        with caplog.at_level("WARNING"):
            embeddings, _ = embedder._embed_batch(parts, batch_size=10)
        assert embeddings == [None]
        assert "No embeddings returned by the API" in caplog.text

    def test_embed_batch_embedding_with_no_values(self, caplog):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(api_key=Secret.from_token("fake-api-key"), progress_bar=False)

        mock_embedding = MagicMock()
        mock_embedding.values = None
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        embedder._client.models.embed_content = MagicMock(return_value=mock_response)

        parts = [MagicMock(spec=types.Part)]
        with caplog.at_level("WARNING"):
            embeddings, _ = embedder._embed_batch(parts, batch_size=10)
        assert embeddings == [None]
        assert "has no values" in caplog.text

    def test_run_wrong_input(self):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))
        with pytest.raises(TypeError, match="expects a list of Documents"):
            embedder.run(documents="not a list")
        with pytest.raises(TypeError, match="expects a list of Documents"):
            embedder.run(documents=[1, 2, 3])

    def test_run_with_mocked_client(self, test_files_path):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(api_key=Secret.from_token("fake-api-key"), progress_bar=False)
        docs = [
            Document(content=None, meta={"file_path": str(test_files_path / "apple.jpg")}),
            Document(content=None, meta={"file_path": str(test_files_path / "banana.png")}),
        ]

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding, mock_embedding]
        embedder._client.models.embed_content = MagicMock(return_value=mock_response)

        result = embedder.run(documents=docs)
        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert doc.embedding == [0.1, 0.2, 0.3]
        assert result["meta"]["model"] == "gemini-embedding-2"

    @pytest.mark.asyncio
    async def test_run_async_with_mocked_client(self, test_files_path):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(api_key=Secret.from_token("fake-api-key"), progress_bar=False)
        docs = [
            Document(content=None, meta={"file_path": str(test_files_path / "apple.jpg")}),
        ]

        mock_embedding = MagicMock()
        mock_embedding.values = [0.4, 0.5, 0.6]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        embedder._client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        result = await embedder.run_async(documents=docs)
        assert len(result["documents"]) == 1
        assert result["documents"][0].embedding == [0.4, 0.5, 0.6]
        assert result["meta"]["model"] == "gemini-embedding-2"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    def test_live_run(self, test_files_path):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(progress_bar=False)
        docs = [
            Document(content=None, meta={"file_path": str(test_files_path / "apple.jpg")}),
            Document(content=None, meta={"file_path": str(test_files_path / "sample_pdf_3.pdf")}),
        ]

        result = embedder.run(documents=docs)
        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert len(doc.embedding) == 3072
        assert result["meta"]["model"] == "gemini-embedding-2"

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    async def test_live_run_async(self, test_files_path):
        embedder = GoogleGenAIMultimodalDocumentEmbedder(progress_bar=False)
        docs = [
            Document(content=None, meta={"file_path": str(test_files_path / "apple.jpg")}),
            Document(content=None, meta={"file_path": str(test_files_path / "sample_pdf_3.pdf")}),
        ]

        result = await embedder.run_async(documents=docs)
        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert len(doc.embedding) == 3072
        assert result["meta"]["model"] == "gemini-embedding-2"
