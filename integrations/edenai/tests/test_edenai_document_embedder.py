# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.edenai.document_embedder import EdenAIDocumentEmbedder

API_BASE_URL = "https://api.edenai.run/v3"
DEFAULT_MODEL = "openai/text-embedding-3-small"


class TestEdenAIDocumentEmbedder:
    def test_supported_models(self):
        """SUPPORTED_MODELS is a non-empty list of strings."""
        models = EdenAIDocumentEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "test-api-key")

        embedder = EdenAIDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var(["EDENAI_API_KEY"])
        assert embedder.api_base_url == API_BASE_URL
        assert embedder.model == DEFAULT_MODEL
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("EDENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            EdenAIDocumentEmbedder()

    def test_init_with_parameters(self):
        embedder = EdenAIDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="mistral/mistral-embed",
            prefix="START",
            suffix="END",
            batch_size=8,
            progress_bar=False,
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == API_BASE_URL
        assert embedder.model == "mistral/mistral-embed"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.batch_size == 8
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["topic"]
        assert embedder.embedding_separator == " | "

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "test-api-key")

        embedder_component = EdenAIDocumentEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.edenai.document_embedder.EdenAIDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["EDENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": DEFAULT_MODEL,
                "api_base_url": API_BASE_URL,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("EDENAI_API_KEY", "test-secret-key")
        data = {
            "type": "haystack_integrations.components.embedders.edenai.document_embedder.EdenAIDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["EDENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": DEFAULT_MODEL,
                "api_base_url": API_BASE_URL,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }
        embedder = EdenAIDocumentEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var(["EDENAI_API_KEY"])
        assert embedder.api_base_url == API_BASE_URL
        assert embedder.model == DEFAULT_MODEL
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.embedding_separator == "\n"
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.http_client_kwargs is None

    @pytest.mark.skipif(
        not os.environ.get("EDENAI_API_KEY", None),
        reason="Export an env var called EDENAI_API_KEY containing the Eden AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        docs = [
            Document(content="I love pizza!"),
            Document(content="Haystack is an open-source framework for building RAG applications."),
        ]
        embedder = EdenAIDocumentEmbedder()
        result = embedder.run(docs)
        documents = result["documents"]
        assert len(documents) == 2
        for doc in documents:
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0
            assert all(isinstance(x, float) for x in doc.embedding)
