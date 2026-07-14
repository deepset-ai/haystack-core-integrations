# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from unittest.mock import Mock, patch

import pytest
from haystack import Document
from haystack.utils import Secret
from openai import BadRequestError

from haystack_integrations.components.embedders.stackit.document_embedder import STACKITDocumentEmbedder


class TestSTACKITDocumentEmbedder:
    def test_supported_models(self):
        """SUPPORTED_MODELS is a non-empty list of strings."""
        models = STACKITDocumentEmbedder.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")

        embedder = STACKITDocumentEmbedder(model="intfloat/e5-mistral-7b-instruct")
        assert embedder.api_key == Secret.from_env_var(["STACKIT_API_KEY"])
        assert embedder.model == "intfloat/e5-mistral-7b-instruct"
        assert embedder.dimensions is None
        assert embedder.api_base_url == "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = STACKITDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="Qwen/Qwen3-VL-Embedding-8B",
            dimensions=1024,
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "Qwen/Qwen3-VL-Embedding-8B"
        assert embedder.dimensions == 1024
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")

        embedder_component = STACKITDocumentEmbedder(model="intfloat/e5-mistral-7b-instruct")
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
                "model": "intfloat/e5-mistral-7b-instruct",
                "dimensions": None,
                "api_base_url": "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
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
        embedder = STACKITDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="Qwen/Qwen3-VL-Embedding-8B",
            dimensions=1024,
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com"},
        )
        component_dict = embedder.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "Qwen/Qwen3-VL-Embedding-8B",
                "dimensions": 1024,
                "api_base_url": "https://custom-api-base-url.com",
                "prefix": "START",
                "suffix": "END",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": "-",
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "https://proxy.example.com"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.stackit.document_embedder.STACKITDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
                "model": "intfloat/e5-mistral-7b-instruct",
                "dimensions": None,
                "api_base_url": "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
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
        embedder = STACKITDocumentEmbedder.from_dict(data)
        assert embedder.api_key == Secret.from_env_var(["STACKIT_API_KEY"])
        assert embedder.model == "intfloat/e5-mistral-7b-instruct"
        assert embedder.dimensions is None
        assert embedder.api_base_url == "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.http_client_kwargs is None

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = STACKITDocumentEmbedder(model="intfloat/e5-mistral-7b-instruct")

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

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_with_dimensions(self):
        embedder = STACKITDocumentEmbedder(model="Qwen/Qwen3-VL-Embedding-8B", dimensions=256)

        docs = [Document(content="I love cheese")]
        result = embedder.run(docs)
        docs_with_embeddings = result["documents"]

        assert len(docs_with_embeddings) == 1
        assert len(docs_with_embeddings[0].embedding) == 256

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_with_dimensions_unsupported_by_model(self, caplog):
        # OpenAIDocumentEmbedder defaults to raise_on_failure=False, so a per-batch API error
        # is logged and the affected documents come back without an embedding, instead of raising.
        embedder = STACKITDocumentEmbedder(model="intfloat/e5-mistral-7b-instruct", dimensions=256)

        with caplog.at_level(logging.ERROR):
            result = embedder.run([Document(content="I love cheese")])

        assert result["documents"][0].embedding is None
        logged_errors = [r.exc for r in caplog.records if isinstance(getattr(r, "exc", None), BadRequestError)]
        assert len(logged_errors) == 1
        assert logged_errors[0].status_code == 400

    def test_run_forwards_dimensions_to_client(self):
        embedder = STACKITDocumentEmbedder(
            model="Qwen/Qwen3-VL-Embedding-8B",
            dimensions=1024,
            api_key=Secret.from_token("test-api-key"),
        )

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.model = "Qwen/Qwen3-VL-Embedding-8B"
        mock_response.usage = {"prompt_tokens": 4, "total_tokens": 4}

        with patch.object(embedder.client.embeddings, "create", return_value=mock_response) as mock_create:
            embedder.run([Document(content="I love cheese")])

        assert mock_create.call_args.kwargs["dimensions"] == 1024

    def test_run_wrong_input_format(self):
        embedder = STACKITDocumentEmbedder(
            model="intfloat/e5-mistral-7b-instruct", api_key=Secret.from_token("test-api-key")
        )

        match_error_msg = (
            "OpenAIDocumentEmbedder expects a list of Documents as input.In case you want to embed a string, "
            "please use the OpenAITextEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents="text")
        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents=[1, 2, 3])

        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}
