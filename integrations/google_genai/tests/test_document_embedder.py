# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
from typing import List

import pytest
from haystack import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder


def mock_google_response(contents: List[str], model: str = "text-embedding-004", **kwargs) -> dict:
    secure_random = random.SystemRandom()
    dict_response = {
        "embedding": [[secure_random.random() for _ in range(768)] for _ in contents],
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
        assert embedder._model == "text-embedding-004"
        assert embedder._prefix == ""
        assert embedder._suffix == ""
        assert embedder._batch_size == 32
        assert embedder._progress_bar is True
        assert embedder._meta_fields_to_embed == []
        assert embedder._embedding_separator == "\n"
        assert embedder._config == {"task_type": "SEMANTIC_SIMILARITY"}

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
                "model": "text-embedding-004",
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
                "model": "text-embedding-004",
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
        assert embedder._model == "text-embedding-004"
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

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "text-embedding-004"

        embedder = GoogleGenAIDocumentEmbedder(model=model, meta_fields_to_embed=["topic"], embedding_separator=" | ")

        result = embedder.run(documents=docs)
        documents_with_embeddings = result["documents"]
        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 768
            assert all(isinstance(x, float) for x in doc.embedding)

        assert "text" in result["meta"]["model"] and "004" in result["meta"]["model"], (
            "The model name does not contain 'text' and '004'"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    async def test_run_async(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "text-embedding-004"

        embedder = GoogleGenAIDocumentEmbedder(model=model, meta_fields_to_embed=["topic"], embedding_separator=" | ")

        result = await embedder.run_async(documents=docs)
        documents_with_embeddings = result["documents"]
        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 768
            assert all(isinstance(x, float) for x in doc.embedding)

        assert "text" in result["meta"]["model"] and "004" in result["meta"]["model"], (
            "The model name does not contain 'text' and '004'"
        )
        assert result["documents"][0].meta == {"topic": "Cuisine"}
        assert result["documents"][1].meta == {"topic": "ML"}
        assert result["meta"] == {"model": model}
