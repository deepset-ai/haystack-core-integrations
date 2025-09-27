# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.mistral.document_embedder import MistralDocumentEmbedder


class TestMistralDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        embedder = MistralDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var(["MISTRAL_API_KEY"])
        assert embedder.model == "mistral-embed"
        assert embedder.api_base_url == "https://api.mistral.ai/v1"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = MistralDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-embed-v2",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "mistral-embed-v2"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        embedder_component = MistralDocumentEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-embed",
                "api_base_url": "https://api.mistral.ai/v1",
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
        embedder = MistralDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="mistral-embed-v2",
            api_base_url="https://custom-api-base-url.com",
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
            "type": "haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "mistral-embed-v2",
                "api_base_url": "https://custom-api-base-url.com",
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
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.mistral.document_embedder.MistralDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-embed",
                "api_base_url": "https://api.mistral.ai/v1",
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
        component = MistralDocumentEmbedder.from_dict(data)
        assert component.api_key == Secret.from_env_var(["MISTRAL_API_KEY"])
        assert component.model == "mistral-embed"
        assert component.api_base_url == "https://api.mistral.ai/v1"
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert component.timeout is None
        assert component.max_retries is None
        assert component.http_client_kwargs is None

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = MistralDocumentEmbedder()

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
        embedder = MistralDocumentEmbedder(api_key=Secret.from_token("test-api-key"))

        match_error_msg = (
            "OpenAIDocumentEmbedder expects a list of Documents as input.In case you want to embed a string, "
            "please use the OpenAITextEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents="text")
        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents=[1, 2, 3])

        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}
