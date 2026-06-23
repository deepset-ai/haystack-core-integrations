# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.telnyx.document_embedder import TelnyxDocumentEmbedder


class TestTelnyxDocumentEmbedder:
    def test_supported_models(self):
        models = TelnyxDocumentEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "test-api-key")

        embedder = TelnyxDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var(["TELNYX_API_KEY"])
        assert embedder.model == "thenlper/gte-large"
        assert embedder.api_base_url == "https://api.telnyx.com/v2/ai/openai"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.dimensions is None

    def test_init_with_parameters(self):
        embedder = TelnyxDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="custom-dimensions-model",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
            dimensions=256,
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "custom-dimensions-model"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"
        assert embedder.dimensions == 256

    def test_init_rejects_unsupported_dimensions(self):
        with pytest.raises(ValueError, match="does not support custom dimensions"):
            TelnyxDocumentEmbedder(api_key=Secret.from_token("test-api-key"), dimensions=256)

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "test-api-key")

        component_dict = TelnyxDocumentEmbedder().to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.telnyx.document_embedder.TelnyxDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["TELNYX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "thenlper/gte-large",
                "api_base_url": "https://api.telnyx.com/v2/ai/openai",
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "dimensions": None,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-secret-key")
        embedder = TelnyxDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="custom-dimensions-model",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
            dimensions=256,
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com"},
        )
        assert embedder.to_dict() == {
            "type": "haystack_integrations.components.embedders.telnyx.document_embedder.TelnyxDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "custom-dimensions-model",
                "api_base_url": "https://custom-api-base-url.com",
                "prefix": "START",
                "suffix": "END",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": "-",
                "dimensions": 256,
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "https://proxy.example.com"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("TELNYX_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.telnyx.document_embedder.TelnyxDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["TELNYX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "thenlper/gte-large",
                "api_base_url": "https://api.telnyx.com/v2/ai/openai",
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "dimensions": None,
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }
        embedder = TelnyxDocumentEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var(["TELNYX_API_KEY"])
        assert embedder.model == "thenlper/gte-large"
        assert embedder.api_base_url == "https://api.telnyx.com/v2/ai/openai"
        assert embedder.batch_size == 32
        assert embedder.dimensions is None

    def test_run_wrong_input_format(self):
        embedder = TelnyxDocumentEmbedder(api_key=Secret.from_token("test-api-key"))
        match_error_msg = "OpenAIDocumentEmbedder expects a list of Documents as input"

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents="text")
        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents=[1, 2, 3])

        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}

    @pytest.mark.skipif(
        not os.environ.get("TELNYX_API_KEY", None),
        reason="Export TELNYX_API_KEY to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = TelnyxDocumentEmbedder().run(docs)
        docs_with_embeddings = result["documents"]

        assert len(docs_with_embeddings) == len(docs)
        for doc in docs_with_embeddings:
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)
