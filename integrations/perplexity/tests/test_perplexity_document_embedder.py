# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import httpx
import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.perplexity import (
    document_embedder as document_embedder_module,
)
from haystack_integrations.components.embedders.perplexity.document_embedder import (
    PerplexityDocumentEmbedder,
)


def _make_transport(captured: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                    {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
                ],
                "model": "pplx-embed-v1-0.6b",
                "usage": {"prompt_tokens": 6, "total_tokens": 6},
            },
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(handler)


class TestPerplexityDocumentEmbedder:
    def test_attribution_header_falls_back_when_package_is_not_installed(self, monkeypatch):
        def raise_package_not_found(_package_name: str) -> str:
            raise document_embedder_module.importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(
            document_embedder_module.importlib.metadata,
            "version",
            raise_package_not_found,
        )

        assert document_embedder_module._attribution_header() == "haystack/unknown"

    def test_http_client_kwargs_with_attribution_keeps_existing_headers(self):
        kwargs = document_embedder_module._http_client_kwargs_with_attribution(
            {"headers": {"test-header": "test-value"}}
        )

        assert kwargs["headers"]["test-header"] == "test-value"
        assert kwargs["headers"]["X-Pplx-Integration"].startswith("haystack/")

    def test_supported_models(self) -> None:
        models = PerplexityDocumentEmbedder.SUPPORTED_MODELS
        assert models == ["pplx-embed-v1-0.6b", "pplx-embed-v1-4b"]

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-api-key")

        embedder = PerplexityDocumentEmbedder()

        assert embedder.api_key == Secret.from_env_var(["PERPLEXITY_API_KEY"])
        assert embedder.model == "pplx-embed-v1-0.6b"
        assert embedder.api_base_url == "https://api.perplexity.ai/v1"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = PerplexityDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="pplx-embed-v1-4b",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
        )

        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "pplx-embed-v1-4b"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-api-key")

        embedder = PerplexityDocumentEmbedder()
        component_dict = embedder.to_dict()

        assert component_dict["init_parameters"]["api_key"] == Secret.from_env_var("PERPLEXITY_API_KEY").to_dict()
        assert component_dict == {
            "type": (
                "haystack_integrations.components.embedders.perplexity.document_embedder.PerplexityDocumentEmbedder"
            ),
            "init_parameters": {
                "api_key": Secret.from_env_var("PERPLEXITY_API_KEY").to_dict(),
                "model": "pplx-embed-v1-0.6b",
                "api_base_url": "https://api.perplexity.ai/v1",
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
        embedder = PerplexityDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="pplx-embed-v1-4b",
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
            "type": (
                "haystack_integrations.components.embedders.perplexity.document_embedder.PerplexityDocumentEmbedder"
            ),
            "init_parameters": {
                "api_key": Secret.from_env_var("ENV_VAR", strict=False).to_dict(),
                "model": "pplx-embed-v1-4b",
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
        monkeypatch.setenv("PERPLEXITY_API_KEY", "fake-api-key")
        data = {
            "type": (
                "haystack_integrations.components.embedders.perplexity.document_embedder.PerplexityDocumentEmbedder"
            ),
            "init_parameters": {
                "api_key": Secret.from_env_var("PERPLEXITY_API_KEY").to_dict(),
                "model": "pplx-embed-v1-0.6b",
                "api_base_url": "https://api.perplexity.ai/v1",
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

        component = PerplexityDocumentEmbedder.from_dict(data)

        assert component.api_key == Secret.from_env_var(["PERPLEXITY_API_KEY"])
        assert component.model == "pplx-embed-v1-0.6b"
        assert component.api_base_url == "https://api.perplexity.ai/v1"
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert component.timeout is None
        assert component.max_retries is None
        assert component.http_client_kwargs is None

    def test_run_sends_attribution_header(self):
        captured: list[httpx.Request] = []
        embedder = PerplexityDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            progress_bar=False,
            http_client_kwargs={"transport": _make_transport(captured)},
        )
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(
                content="A transformer is a deep learning architecture",
                meta={"topic": "ML"},
            ),
        ]

        result = embedder.run(docs)

        docs_with_embeddings = result["documents"]
        assert docs_with_embeddings[0].embedding == [0.1, 0.2, 0.3]
        assert docs_with_embeddings[1].embedding == [0.4, 0.5, 0.6]
        assert len(captured) == 1
        request = captured[0]
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert request.headers["X-Pplx-Integration"].startswith("haystack/")
        body = json.loads(request.content)
        assert body["model"] == "pplx-embed-v1-0.6b"
        assert body["input"] == [
            "I love cheese",
            "A transformer is a deep learning architecture",
        ]
        assert body["encoding_format"] == "float"

    def test_run_wrong_input_format(self):
        embedder = PerplexityDocumentEmbedder(api_key=Secret.from_token("test-api-key"))
        match_error_msg = (
            "OpenAIDocumentEmbedder expects a list of Documents as input.In case you want to embed a string, "
            "please use the OpenAITextEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents="text")
        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(documents=[1, 2, 3])

        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}


@pytest.mark.skipif(
    not os.environ.get("PERPLEXITY_API_KEY"),
    reason="Export PERPLEXITY_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestPerplexityDocumentEmbedderInference:
    def test_live_run(self):
        docs = [
            Document(content="The capital of France is Paris."),
            Document(content="The capital of Germany is Berlin."),
        ]
        embedder = PerplexityDocumentEmbedder()
        result = embedder.run(documents=docs)

        embedded_docs = result["documents"]
        assert len(embedded_docs) == len(docs)
        for doc in embedded_docs:
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0
            assert all(isinstance(x, float) for x in doc.embedding)

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        docs = [
            Document(content="The capital of France is Paris."),
            Document(content="The capital of Germany is Berlin."),
        ]
        embedder = PerplexityDocumentEmbedder()
        result = await embedder.run_async(documents=docs)

        embedded_docs = result["documents"]
        assert len(embedded_docs) == len(docs)
        for doc in embedded_docs:
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0
