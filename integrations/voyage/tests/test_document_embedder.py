# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.voyage import VoyageDocumentEmbedder


class TestVoyageDocumentEmbedder:
    def test_supported_models(self):
        models = VoyageDocumentEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var("VOYAGE_API_KEY")
        assert embedder.model == "voyage-3.5"
        assert embedder.input_type == "document"
        assert embedder.truncation is True
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = VoyageDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="voyage-3-large",
            prefix="pre ",
            suffix=" post",
            input_type="query",
            truncation=False,
            output_dimension=1024,
            output_dtype="float",
            timeout=60.0,
            batch_size=8,
            progress_bar=False,
            meta_fields_to_embed=["title"],
            embedding_separator=" | ",
        )
        assert embedder.model == "voyage-3-large"
        assert embedder.input_type == "query"
        assert embedder.truncation is False
        assert embedder.output_dimension == 1024
        assert embedder.batch_size == 8
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["title"]
        assert embedder.embedding_separator == " | "

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        component_dict = VoyageDocumentEmbedder().to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.voyage.document_embedder.VoyageDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-3.5",
                "prefix": "",
                "suffix": "",
                "input_type": "document",
                "truncation": True,
                "output_dimension": None,
                "output_dtype": None,
                "timeout": None,
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage.document_embedder.VoyageDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-3.5",
                "prefix": "",
                "suffix": "",
                "input_type": "document",
                "truncation": True,
                "output_dimension": None,
                "output_dtype": None,
                "timeout": None,
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }
        embedder = VoyageDocumentEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var("VOYAGE_API_KEY")
        assert embedder.model == "voyage-3.5"

    def test_run_wrong_input_format(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageDocumentEmbedder()
        with pytest.raises(TypeError):
            embedder.run(documents="I'm a string, not a list of Documents")
        with pytest.raises(TypeError):
            embedder.run(documents=[1, 2, 3])

    def test_run_on_empty_list(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageDocumentEmbedder()
        result = embedder.run(documents=[])
        assert result == {"documents": [], "meta": {}}

    def test_prepare_texts_to_embed(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageDocumentEmbedder(meta_fields_to_embed=["title"], embedding_separator=" | ")
        docs = [Document(content="my content", meta={"title": "my title"})]
        assert embedder._prepare_texts_to_embed(docs) == ["my title | my content"]

    def test_run(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageDocumentEmbedder(batch_size=2)
        embedder._client.embed = MagicMock(
            side_effect=[
                SimpleNamespace(embeddings=[[0.1, 0.1], [0.2, 0.2]], total_tokens=4),
                SimpleNamespace(embeddings=[[0.3, 0.3]], total_tokens=2),
            ]
        )
        docs = [Document(content="a"), Document(content="b"), Document(content="c")]

        result = embedder.run(documents=docs)

        assert embedder._client.embed.call_count == 2
        embeddings = [doc.embedding for doc in result["documents"]]
        assert embeddings == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        assert result["meta"] == {"model": "voyage-3.5", "total_tokens": 6}

    def test_run_does_not_modify_original_documents(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageDocumentEmbedder()
        embedder._client.embed = MagicMock(return_value=SimpleNamespace(embeddings=[[0.1, 0.2]], total_tokens=2))
        docs = [Document(content="a")]
        embedder.run(documents=docs)
        assert docs[0].embedding is None

    @pytest.mark.asyncio
    async def test_run_async(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
        embedder = VoyageDocumentEmbedder()
        embedder._async_client.embed = AsyncMock(
            return_value=SimpleNamespace(embeddings=[[0.1, 0.1], [0.2, 0.2]], total_tokens=4)
        )
        docs = [Document(content="a"), Document(content="b")]

        result = await embedder.run_async(documents=docs)

        embeddings = [doc.embedding for doc in result["documents"]]
        assert embeddings == [[0.1, 0.1], [0.2, 0.2]]
        assert result["meta"] == {"model": "voyage-3.5", "total_tokens": 4}

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY not set")
    def test_live_run(self):
        docs = [Document(content="I love pizza"), Document(content="I love pasta")]
        embedder = VoyageDocumentEmbedder()
        result = embedder.run(documents=docs)
        documents = result["documents"]
        assert len(documents) == 2
        assert all(len(doc.embedding) > 0 for doc in documents)
        assert result["meta"]["total_tokens"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY not set")
    async def test_live_run_async(self):
        docs = [Document(content="I love pizza"), Document(content="I love pasta")]
        embedder = VoyageDocumentEmbedder()
        result = await embedder.run_async(documents=docs)
        assert all(len(doc.embedding) > 0 for doc in result["documents"])
