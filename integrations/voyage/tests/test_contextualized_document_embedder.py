# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.voyage import VoyageContextualizedDocumentEmbedder


def _result(embeddings_per_group, total_tokens):
    results = [
        SimpleNamespace(index=i, embeddings=embs, chunk_texts=None) for i, embs in enumerate(embeddings_per_group)
    ]
    return SimpleNamespace(results=results, total_tokens=total_tokens)


class TestVoyageContextualizedDocumentEmbedder:
    def test_supported_models(self):
        models = VoyageContextualizedDocumentEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert "voyage-context-4" in models

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageContextualizedDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var("VOYAGE_API_KEY")
        assert embedder.model == "voyage-context-4"
        assert embedder.input_type == "document"
        assert embedder.batch_size == 32
        assert embedder.group_by == "source_id"

    def test_init_with_parameters(self):
        embedder = VoyageContextualizedDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="voyage-context-3",
            output_dimension=2048,
            output_dtype="int8",
            group_by="file_id",
        )
        assert embedder.model == "voyage-context-3"
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "int8"
        assert embedder.group_by == "file_id"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        component_dict = VoyageContextualizedDocumentEmbedder().to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.voyage."
            "contextualized_document_embedder.VoyageContextualizedDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-context-4",
                "prefix": "",
                "suffix": "",
                "input_type": "document",
                "output_dimension": None,
                "output_dtype": None,
                "timeout": None,
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "group_by": "source_id",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        data = VoyageContextualizedDocumentEmbedder().to_dict()
        embedder = VoyageContextualizedDocumentEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var("VOYAGE_API_KEY")
        assert embedder.model == "voyage-context-4"
        assert embedder.group_by == "source_id"

    def test_run_wrong_input_format(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageContextualizedDocumentEmbedder()
        with pytest.raises(TypeError):
            embedder.run(documents="not a list")

    def test_run_on_empty_list(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageContextualizedDocumentEmbedder()
        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}

    def test_group_documents_by_meta_field(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageContextualizedDocumentEmbedder(group_by="source_id")
        docs = [
            Document(content="a", meta={"source_id": "1"}),
            Document(content="b", meta={"source_id": "2"}),
            Document(content="c", meta={"source_id": "1"}),
        ]
        assert embedder._group_documents(docs) == [[0, 2], [1]]

    def test_group_documents_single_group(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageContextualizedDocumentEmbedder(group_by=None)
        docs = [Document(content="a"), Document(content="b")]
        assert embedder._group_documents(docs) == [[0, 1]]

    def test_run(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageContextualizedDocumentEmbedder(group_by="source_id")
        embedder._client.contextualized_embed = MagicMock(
            return_value=_result([[[0.1], [0.3]], [[0.2]]], total_tokens=6)
        )
        docs = [
            Document(content="chunk a1", meta={"source_id": "doc1"}),
            Document(content="chunk b1", meta={"source_id": "doc2"}),
            Document(content="chunk a2", meta={"source_id": "doc1"}),
        ]

        result = embedder.run(documents=docs)

        embedder._client.contextualized_embed.assert_called_once_with(
            inputs=[["chunk a1", "chunk a2"], ["chunk b1"]],
            model="voyage-context-4",
            input_type="document",
            output_dimension=None,
            output_dtype=None,
        )
        embeddings = [doc.embedding for doc in result["documents"]]
        assert embeddings == [[0.1], [0.2], [0.3]]
        assert result["meta"] == {"model": "voyage-context-4", "total_tokens": 6}

    def test_run_does_not_modify_original_documents(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageContextualizedDocumentEmbedder(group_by=None)
        embedder._client.contextualized_embed = MagicMock(return_value=_result([[[0.1, 0.2]]], total_tokens=2))
        docs = [Document(content="a")]
        embedder.run(documents=docs)
        assert docs[0].embedding is None

    @pytest.mark.asyncio
    async def test_run_async(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageContextualizedDocumentEmbedder(group_by=None)
        embedder._async_client.contextualized_embed = AsyncMock(return_value=_result([[[0.1], [0.2]]], total_tokens=4))
        docs = [Document(content="a"), Document(content="b")]

        result = await embedder.run_async(documents=docs)

        embeddings = [doc.embedding for doc in result["documents"]]
        assert embeddings == [[0.1], [0.2]]
        assert result["meta"] == {"model": "voyage-context-4", "total_tokens": 4}

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY not set")
    def test_live_run(self):
        docs = [
            Document(content="The Eiffel Tower is in Paris.", meta={"source_id": "doc1"}),
            Document(content="It was completed in 1889.", meta={"source_id": "doc1"}),
        ]
        embedder = VoyageContextualizedDocumentEmbedder()
        result = embedder.run(documents=docs)
        documents = result["documents"]
        assert len(documents) == 2
        assert all(len(doc.embedding) > 0 for doc in documents)
        assert result["meta"]["total_tokens"] > 0
