from unittest.mock import MagicMock, NonCallableMagicMock

import numpy as np
import pytest
from gradientai.openapi.client.models.generate_embedding_success import GenerateEmbeddingSuccess
from haystack import Document

from gradient_haystack.embedders.gradient_document_embedder import GradientDocumentEmbedder

access_token = "access_token"
workspace_id = "workspace_id"
model = "bge-large"


class TestGradientDocumentEmbedder:
    @pytest.mark.unit
    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("GRADIENT_ACCESS_TOKEN", access_token)
        monkeypatch.setenv("GRADIENT_WORKSPACE_ID", workspace_id)

        embedder = GradientDocumentEmbedder()
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_init_without_access_token(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_ACCESS_TOKEN", raising=False)

        with pytest.raises(ValueError):
            GradientDocumentEmbedder(workspace_id=workspace_id)

    @pytest.mark.unit
    def test_init_without_workspace(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_WORKSPACE_ID", raising=False)

        with pytest.raises(ValueError):
            GradientDocumentEmbedder(access_token=access_token)

    @pytest.mark.unit
    def test_init_from_params(self):
        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_init_from_params_precedence(self, monkeypatch):
        monkeypatch.setenv("GRADIENT_ACCESS_TOKEN", "env_access_token")
        monkeypatch.setenv("GRADIENT_WORKSPACE_ID", "env_workspace_id")

        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_to_dict(self):
        component = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)
        data = component.to_dict()
        assert data == {
            "type": "gradient_haystack.embedders.gradient_document_embedder.GradientDocumentEmbedder",
            "init_parameters": {"workspace_id": workspace_id, "model_name": "bge-large"},
        }

    @pytest.mark.unit
    def test_warmup(self):
        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._gradient.get_embeddings_model = MagicMock()
        embedder.warm_up()
        embedder._gradient.get_embeddings_model.assert_called_once_with(slug="bge-large")

    @pytest.mark.unit
    def test_warmup_doesnt_reload(self):
        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._gradient.get_embeddings_model = MagicMock(default_return_value="fake model")
        embedder.warm_up()
        embedder.warm_up()
        embedder._gradient.get_embeddings_model.assert_called_once_with(slug="bge-large")

    @pytest.mark.unit
    def test_run_fail_if_not_warmed_up(self):
        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)

        with pytest.raises(RuntimeError, match="warm_up()"):
            embedder.run(documents=[Document(content=f"document number {i}") for i in range(5)])

    @pytest.mark.unit
    def test_run(self):
        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._embedding_model = NonCallableMagicMock()
        embedder._embedding_model.embed.return_value = GenerateEmbeddingSuccess(
            embeddings=[{"embedding": np.random.rand(1024).tolist(), "index": i} for i in range(5)]
        )

        documents = [Document(content=f"document number {i}") for i in range(5)]

        result = embedder.run(documents=documents)

        assert embedder._embedding_model.embed.call_count == 1
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)

    @pytest.mark.unit
    def test_run_batch(self):
        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._embedding_model = NonCallableMagicMock()

        embedder._embedding_model.embed.return_value = GenerateEmbeddingSuccess(
            embeddings=[{"embedding": np.random.rand(1024).tolist(), "index": i} for i in range(110)]
        )

        documents = [Document(content=f"document number {i}") for i in range(110)]

        result = embedder.run(documents=documents)

        assert embedder._embedding_model.embed.call_count == 1
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)

    @pytest.mark.unit
    def test_run_custom_batch(self):
        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id, batch_size=20)
        embedder._embedding_model = NonCallableMagicMock()

        document_count = 101
        embedder._embedding_model.embed.return_value = GenerateEmbeddingSuccess(
            embeddings=[{"embedding": np.random.rand(1024).tolist(), "index": i} for i in range(document_count)]
        )

        documents = [Document(content=f"document number {i}") for i in range(document_count)]

        result = embedder.run(documents=documents)

        assert embedder._embedding_model.embed.call_count == 6
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)

    @pytest.mark.unit
    def test_run_empty(self):
        embedder = GradientDocumentEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._embedding_model = NonCallableMagicMock()

        result = embedder.run(documents=[])

        assert result["documents"] == []
