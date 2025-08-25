# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.cohere import CohereDocumentEmbedder
from haystack_integrations.components.embedders.cohere.embedding_types import EmbeddingTypes

COHERE_API_URL = "https://api.cohere.com"


class TestCohereDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert embedder.model == "embed-english-v2.0"
        assert embedder.input_type == "search_document"
        assert embedder.api_base_url == COHERE_API_URL
        assert embedder.truncate == "END"
        assert embedder.timeout == 120
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.embedding_type == EmbeddingTypes.FLOAT

    def test_init_with_parameters(self):
        embedder = CohereDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="embed-multilingual-v2.0",
            input_type="search_query",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            timeout=60,
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "embed-multilingual-v2.0"
        assert embedder.input_type == "search_query"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.truncate == "START"
        assert embedder.timeout == 60
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"
        assert embedder.embedding_type == EmbeddingTypes.FLOAT

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder_component = CohereDocumentEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "model": "embed-english-v2.0",
                "input_type": "search_document",
                "api_base_url": COHERE_API_URL,
                "truncate": "END",
                "timeout": 120,
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "embedding_type": "float",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder_component = CohereDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="embed-multilingual-v2.0",
            input_type="search_query",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            timeout=60,
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["text_field"],
            embedding_separator="-",
            embedding_type=EmbeddingTypes.INT8,
        )
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "embed-multilingual-v2.0",
                "input_type": "search_query",
                "api_base_url": "https://custom-api-base-url.com",
                "truncate": "START",
                "timeout": 60,
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["text_field"],
                "embedding_separator": "-",
                "embedding_type": "int8",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component_dict = {
            "type": "haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "model": "embed-english-v2.0",
                "input_type": "search_document",
                "api_base_url": COHERE_API_URL,
                "truncate": "END",
                "timeout": 120,
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "embedding_type": "float",
                "use_async_client": False,  # legacy parameter
            },
        }
        embedder = CohereDocumentEmbedder.from_dict(component_dict)
        assert embedder.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert embedder.model == "embed-english-v2.0"
        assert embedder.input_type == "search_document"
        assert embedder.api_base_url == COHERE_API_URL
        assert embedder.truncate == "END"
        assert embedder.timeout == 120
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.embedding_type == EmbeddingTypes.FLOAT
        assert not hasattr(embedder, "use_async_client")

    def test_run_wrong_input_format(self):
        embedder = CohereDocumentEmbedder(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TypeError):
            embedder.run(documents="text")
        with pytest.raises(TypeError):
            embedder.run(documents=[1, 2, 3])

        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}

    @patch("haystack_integrations.components.embedders.cohere.document_embedder.get_response")
    def test_run(self, mock_get_response):
        embedder = CohereDocumentEmbedder(api_key=Secret.from_token("test-api-key"))

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_get_response.return_value = (embeddings, {"api_version": "1.0"})

        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = embedder.run(docs)

        assert result["meta"] == {"api_version": "1.0"}

        for doc, doc_with_embedding, embedding in zip(docs, result["documents"], embeddings):
            assert doc_with_embedding.content == doc.content
            assert doc_with_embedding.meta == doc.meta
            assert doc_with_embedding.embedding == embedding

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.cohere.document_embedder.get_async_response")
    async def test_run_async(self, mock_get_response):
        embedder = CohereDocumentEmbedder(api_key=Secret.from_token("test-api-key"))

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_get_response.return_value = (embeddings, {"api_version": "1.0"})

        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = await embedder.run_async(docs)

        assert result["meta"] == {"api_version": "1.0"}

        for doc, doc_with_embedding, embedding in zip(docs, result["documents"], embeddings):
            assert doc_with_embedding.content == doc.content
            assert doc_with_embedding.meta == doc.meta
            assert doc_with_embedding.embedding == embedding

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        embedder = CohereDocumentEmbedder(model="embed-english-v2.0", embedding_type=EmbeddingTypes.FLOAT)

        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = embedder.run(docs)
        docs_with_embeddings = result["documents"]

        assert isinstance(docs_with_embeddings, list)
        assert len(docs_with_embeddings) == len(docs)
        for doc in docs_with_embeddings:
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    async def test_live_run_async(self):
        embedder = CohereDocumentEmbedder(model="embed-english-v2.0", embedding_type=EmbeddingTypes.FLOAT)

        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = await embedder.run_async(documents=docs)
        docs_with_embeddings = result["documents"]

        assert isinstance(docs_with_embeddings, list)
        assert len(docs_with_embeddings) == len(docs)
        for doc in docs_with_embeddings:
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)
