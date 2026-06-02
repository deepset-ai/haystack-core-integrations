# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.forge.document_embedder import ForgeDocumentEmbedder


class TestForgeDocumentEmbedder:
    def test_supported_models(self) -> None:
        """SUPPORTED_MODELS is a non-empty list of strings that includes the default model."""
        models = ForgeDocumentEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)
        assert "forge-pro" in models

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("FORGE_API_KEY", "test-api-key")

        embedder = ForgeDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var(["FORGE_API_KEY"])
        assert embedder.model == "forge-pro"
        assert embedder.api_base_url == "https://api.voxell.ai/v1"
        assert embedder.dimensions is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = ForgeDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="forge-ultra-4k",
            api_base_url="https://custom-api-base-url.com",
            dimensions=512,
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "forge-ultra-4k"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.dimensions == 512
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("FORGE_API_KEY", "test-api-key")

        embedder_component = ForgeDocumentEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.forge.document_embedder.ForgeDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["FORGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "forge-pro",
                "api_base_url": "https://api.voxell.ai/v1",
                "dimensions": None,
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

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-secret-key")
        embedder = ForgeDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="forge-ultra-4k",
            api_base_url="https://custom-api-base-url.com",
            dimensions=512,
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        component_dict = embedder.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.forge.document_embedder.ForgeDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "forge-ultra-4k",
                "api_base_url": "https://custom-api-base-url.com",
                "dimensions": 512,
                "prefix": "START",
                "suffix": "END",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": "-",
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("FORGE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.forge.document_embedder.ForgeDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["FORGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "forge-pro",
                "api_base_url": "https://api.voxell.ai/v1",
                "dimensions": None,
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
        component = ForgeDocumentEmbedder.from_dict(data)
        assert component.api_key == Secret.from_env_var(["FORGE_API_KEY"])
        assert component.model == "forge-pro"
        assert component.api_base_url == "https://api.voxell.ai/v1"
        assert component.dimensions is None
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert component.timeout is None
        assert component.max_retries is None
        assert component.http_client_kwargs is None

    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("FORGE_API_KEY", "test-api-key")
        embedder = ForgeDocumentEmbedder(model="forge-turbo", dimensions=256, batch_size=8)
        round_tripped = ForgeDocumentEmbedder.from_dict(embedder.to_dict())
        assert round_tripped.to_dict() == embedder.to_dict()

    @pytest.mark.skipif(
        not os.environ.get("FORGE_API_KEY", None),
        reason="Export an env var called FORGE_API_KEY containing the Forge API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = ForgeDocumentEmbedder()

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

    def test_run_wrong_input_format(self):
        embedder = ForgeDocumentEmbedder(api_key=Secret.from_token("test-api-key"))

        match_error_msg = (
            "OpenAIDocumentEmbedder expects a list of Documents as input.In case you want to embed a string, "
            "please use the OpenAITextEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents="text")
        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents=[1, 2, 3])

        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}
