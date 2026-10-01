# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from haystack import Document
from haystack.utils import Secret
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from haystack_integrations.components.embedders.otari import OtariDocumentEmbedder

from .conftest import OTARI_API_BASE_URL, requires_api_key

DEFAULT_MODEL = "openai:text-embedding-3-small"
DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"
COMPONENT_TYPE = "haystack_integrations.components.embedders.otari.document_embedder.OtariDocumentEmbedder"


class TestOtariDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var("OTARI_API_KEY")
        assert embedder.model == DEFAULT_MODEL
        assert embedder.dimensions is None
        assert embedder.api_base_url == DEFAULT_API_BASE_URL
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.raise_on_failure is False

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        assert OtariDocumentEmbedder().to_dict() == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "api_key": {"env_vars": ["OTARI_API_KEY"], "strict": True, "type": "env_var"},
                "model": DEFAULT_MODEL,
                "dimensions": None,
                "api_base_url": DEFAULT_API_BASE_URL,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
                "raise_on_failure": False,
            },
        }

    def test_to_dict_from_dict_round_trip_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        embedder = OtariDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="mistral:mistral-embed",
            dimensions=256,
            api_base_url="https://otari.example.com/api/v1",
            prefix="passage: ",
            suffix=" end",
            batch_size=8,
            progress_bar=False,
            meta_fields_to_embed=["title"],
            embedding_separator=" | ",
            timeout=10,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
            raise_on_failure=True,
        )
        data = embedder.to_dict()
        assert "organization" not in data["init_parameters"]

        restored = OtariDocumentEmbedder.from_dict(data)
        assert restored.api_key == Secret.from_env_var("ENV_VAR")
        assert restored.model == "mistral:mistral-embed"
        assert restored.dimensions == 256
        assert restored.api_base_url == "https://otari.example.com/api/v1"
        assert restored.prefix == "passage: "
        assert restored.suffix == " end"
        assert restored.batch_size == 8
        assert restored.progress_bar is False
        assert restored.meta_fields_to_embed == ["title"]
        assert restored.embedding_separator == " | "
        assert restored.timeout == 10
        assert restored.max_retries == 2
        assert restored.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert restored.raise_on_failure is True

    def test_run(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        embedder = OtariDocumentEmbedder(meta_fields_to_embed=["title"], progress_bar=False)
        docs = [
            Document(content="I love pizza!", meta={"title": "Food"}),
            Document(content="Haystack builds pipelines."),
        ]
        with patch("openai.resources.embeddings.Embeddings.create") as mock_create:
            mock_create.return_value = CreateEmbeddingResponse(
                data=[
                    Embedding(embedding=[0.1, 0.2], index=0, object="embedding"),
                    Embedding(embedding=[0.3, 0.4], index=1, object="embedding"),
                ],
                model="text-embedding-3-small",
                object="list",
                usage=Usage(prompt_tokens=10, total_tokens=10),
            )
            result = embedder.run(docs)

        _, kwargs = mock_create.call_args
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["input"] == ["Food\nI love pizza!", "Haystack builds pipelines."]
        assert [doc.embedding for doc in result["documents"]] == [[0.1, 0.2], [0.3, 0.4]]
        assert result["meta"]["usage"] == {"prompt_tokens": 10, "total_tokens": 10}

    @requires_api_key
    @pytest.mark.integration
    def test_live_run(self):
        docs = [
            Document(content="I love pizza!"),
            Document(content="Haystack is an open-source framework for building RAG applications."),
        ]
        embedder = OtariDocumentEmbedder(api_base_url=OTARI_API_BASE_URL, dimensions=256, progress_bar=False)
        result = embedder.run(docs)

        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert len(doc.embedding) == 256
            assert all(isinstance(x, float) for x in doc.embedding)
        assert "text-embedding-3-small" in result["meta"]["model"]
