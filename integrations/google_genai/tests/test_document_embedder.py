# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from haystack import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder, GoogleGenAITextEmbedder


def mock_google_response(contents: list[str], model: str = "gemini-embedding-001", **kwargs) -> dict:
    secure_random = random.SystemRandom()
    dict_response = {
        "embedding": [[secure_random.random() for _ in range(3072)] for _ in contents],
        "meta": {"model": model},
    }

    return dict_response


class TestGoogleGenAIDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleGenAIDocumentEmbedder()
        assert embedder._api_key.resolve_value() == "fake-api-key"
        assert embedder._api == "gemini"
        assert embedder._vertex_ai_project is None
        assert embedder._vertex_ai_location is None
        assert embedder._model == "gemini-embedding-001"
        assert embedder._prefix == ""
        assert embedder._suffix == ""
        assert embedder._batch_size == 32
        assert embedder._progress_bar is True
        assert embedder._meta_fields_to_embed == []
        assert embedder._embedding_separator == "\n"
        assert embedder._config is None

    def test_init_with_parameters(self, monkeypatch):
        embedder = GoogleGenAIDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key-2"),
            model="model",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
            config={"task_type": "CLASSIFICATION"},
        )
        assert embedder._api_key.resolve_value() == "fake-api-key-2"
        assert embedder._model == "model"
        assert embedder._prefix == "prefix"
        assert embedder._suffix == "suffix"
        assert embedder._batch_size == 64
        assert embedder._progress_bar is False
        assert embedder._meta_fields_to_embed == ["test_field"]
        assert embedder._embedding_separator == " | "
        assert embedder._config == {"task_type": "CLASSIFICATION"}

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="you must export the GOOGLE_API_KEY or GEMINI_API_KEY"):
            GoogleGenAIDocumentEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        component = GoogleGenAIDocumentEmbedder()
        data = component.to_dict()
        assert data == {
            "type": (
                "haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder"
            ),
            "init_parameters": {
                "model": "gemini-embedding-001",
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "strict": False},
                "config": None,
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = GoogleGenAIDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="model",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
            config={"task_type": "CLASSIFICATION"},
        )
        data = component.to_dict()
        assert data == {
            "type": (
                "haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder"
            ),
            "init_parameters": {
                "model": "model",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
                "api_key": {"type": "env_var", "env_vars": ["ENV_VAR"], "strict": False},
                "config": {"task_type": "CLASSIFICATION"},
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        data = {
            "type": (
                "haystack_integrations.components.embedders.google_genai.document_embedder.GoogleGenAIDocumentEmbedder"
            ),
            "init_parameters": {
                "model": "gemini-embedding-001",
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "api_key": {"type": "env_var", "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "strict": False},
                "config": {"task_type": "SEMANTIC_SIMILARITY"},
                "api": "gemini",
                "vertex_ai_project": None,
                "vertex_ai_location": None,
            },
        }
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleGenAIDocumentEmbedder.from_dict(data)
        assert embedder._api_key.resolve_value() == "fake-api-key"
        assert embedder._model == "gemini-embedding-001"
        assert embedder._prefix == ""
        assert embedder._suffix == ""
        assert embedder._batch_size == 32
        assert embedder._progress_bar is True
        assert embedder._meta_fields_to_embed == []
        assert embedder._embedding_separator == "\n"
        assert embedder._config == {"task_type": "SEMANTIC_SIMILARITY"}
        assert embedder._api == "gemini"
        assert embedder._vertex_ai_project is None
        assert embedder._vertex_ai_location is None

    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(id=f"{i}", content=f"document number {i}:\ncontent", meta={"meta_field": f"meta_value {i}"})
            for i in range(5)
        ]

        embedder = GoogleGenAIDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), meta_fields_to_embed=["meta_field"], embedding_separator=" | "
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)
        assert prepared_texts == [
            "meta_value 0 | document number 0:\ncontent",
            "meta_value 1 | document number 1:\ncontent",
            "meta_value 2 | document number 2:\ncontent",
            "meta_value 3 | document number 3:\ncontent",
            "meta_value 4 | document number 4:\ncontent",
        ]

    def test_run_wrong_input_format(self):
        embedder = GoogleGenAIDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="GoogleGenAIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="GoogleGenAIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_on_empty_list(self):
        embedder = GoogleGenAIDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    def test_run_does_not_modify_original_documents(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleGenAIDocumentEmbedder()

        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        # Mock the _embed_batch method to return fake embeddings
        def mock_embed_batch(texts_to_embed, batch_size):
            embeddings = [[0.1, 0.2, 0.3] for _ in texts_to_embed]
            meta = {"model": "gemini-embedding-001"}
            return embeddings, meta

        embedder._embed_batch = mock_embed_batch

        result = embedder.run(documents=docs)

        # Check that the original documents are not modified
        for doc in docs:
            assert doc.embedding is None

        # Check that the returned documents have embeddings
        for doc_with_embedding in result["documents"]:
            assert doc_with_embedding.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_run_async_does_not_modify_original_documents(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleGenAIDocumentEmbedder()

        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        # Mock the _embed_batch_async method to return fake embeddings
        async def mock_embed_batch_async(texts_to_embed, batch_size):
            embeddings = [[0.1, 0.2, 0.3] for _ in texts_to_embed]
            meta = {"model": "gemini-embedding-001"}
            return embeddings, meta

        embedder._embed_batch_async = mock_embed_batch_async

        result = await embedder.run_async(documents=docs)

        # Check that the original documents are not modified
        for doc in docs:
            assert doc.embedding is None

        # Check that the returned documents have embeddings
        for doc_with_embedding in result["documents"]:
            assert doc_with_embedding.embedding == [0.1, 0.2, 0.3]

    def test_embed_batch_passes_full_texts(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleGenAIDocumentEmbedder(batch_size=2)

        texts = ["first document text", "second document text", "third document text"]

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        embedder._client = MagicMock()
        embedder._client.models.embed_content.return_value = mock_response

        embedder._embed_batch(texts, batch_size=2)

        calls = embedder._client.models.embed_content.call_args_list
        assert len(calls) == 2
        assert [c.parts[0].text for c in calls[0].kwargs["contents"]] == [
            "first document text",
            "second document text",
        ]
        assert [c.parts[0].text for c in calls[1].kwargs["contents"]] == ["third document text"]

    @pytest.mark.asyncio
    async def test_embed_batch_async_passes_full_texts(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key")
        embedder = GoogleGenAIDocumentEmbedder(batch_size=2)

        texts = ["first document text", "second document text", "third document text"]

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        embedder._client = MagicMock()
        embedder._client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        await embedder._embed_batch_async(texts, batch_size=2)

        calls = embedder._client.aio.models.embed_content.call_args_list
        assert len(calls) == 2
        assert [c.parts[0].text for c in calls[0].kwargs["contents"]] == [
            "first document text",
            "second document text",
        ]
        assert [c.parts[0].text for c in calls[1].kwargs["contents"]] == ["third document text"]

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model,doc_config,query_config",
        [
            ("gemini-embedding-001", {"task_type": "RETRIEVAL_DOCUMENT"}, {"task_type": "RETRIEVAL_QUERY"}),
            ("gemini-embedding-2", None, None),
        ],
    )
    def test_run(self, model, doc_config, query_config):
        docs = [
            Document(content="The capybara is the largest rodent in the world and lives near rivers in South America."),
            Document(content="Dogs are domesticated mammals known for their loyalty and bond with humans."),
            Document(content="The tiger is the largest big cat, recognized by its orange coat with black stripes."),
        ]

        embedder = GoogleGenAIDocumentEmbedder(model=model, config=doc_config)

        result = embedder.run(documents=docs)
        documents_with_embeddings = result["documents"]
        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3072
            assert all(isinstance(x, float) for x in doc.embedding)

        assert result["meta"]["model"] == model

        text_embedder = GoogleGenAITextEmbedder(model=model, config=query_config)
        query_embedding = text_embedder.run("capybara")["embedding"]
        query_vec = np.array(query_embedding)

        similarities = []
        for doc in documents_with_embeddings:
            doc_vec = np.array(doc.embedding)
            cosine_sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append(cosine_sim)

        assert similarities[0] == max(similarities)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.parametrize("model", ["gemini-embedding-001", "gemini-embedding-2"])
    async def test_run_async(self, model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = GoogleGenAIDocumentEmbedder(model=model, meta_fields_to_embed=["topic"], embedding_separator=" | ")

        result = await embedder.run_async(documents=docs)
        documents_with_embeddings = result["documents"]
        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3072
            assert all(isinstance(x, float) for x in doc.embedding)

        assert result["meta"]["model"] == model
        assert result["documents"][0].meta == {"topic": "Cuisine"}
        assert result["documents"][1].meta == {"topic": "ML"}
