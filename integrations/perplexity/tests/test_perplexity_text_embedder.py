# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import httpx
import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.perplexity import (
    text_embedder as text_embedder_module,
)
from haystack_integrations.components.embedders.perplexity.text_embedder import (
    PerplexityTextEmbedder,
)


def _make_transport(captured: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
                "model": "pplx-embed-v1-0.6b",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(handler)


class TestPerplexityTextEmbedder:
    def test_attribution_header_falls_back_when_package_is_not_installed(self, monkeypatch):
        def raise_package_not_found(_package_name: str) -> str:
            raise text_embedder_module.importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(
            text_embedder_module.importlib.metadata,
            "version",
            raise_package_not_found,
        )

        assert text_embedder_module._attribution_header() == "haystack/unknown"

    def test_http_client_kwargs_with_attribution_keeps_existing_headers(self):
        kwargs = text_embedder_module._http_client_kwargs_with_attribution({"headers": {"test-header": "test-value"}})

        assert kwargs["headers"]["test-header"] == "test-value"
        assert kwargs["headers"]["X-Pplx-Integration"].startswith("haystack/")

    def test_supported_models(self) -> None:
        models = PerplexityTextEmbedder.SUPPORTED_MODELS
        assert models == ["pplx-embed-v1-0.6b", "pplx-embed-v1-4b"]

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-api-key")

        embedder = PerplexityTextEmbedder()

        assert embedder.api_key == Secret.from_env_var(["PERPLEXITY_API_KEY"])
        assert embedder.api_base_url == "https://api.perplexity.ai/v1"
        assert embedder.model == "pplx-embed-v1-0.6b"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = PerplexityTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="pplx-embed-v1-4b",
            prefix="START",
            suffix="END",
        )

        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.api_base_url == "https://api.perplexity.ai/v1"
        assert embedder.model == "pplx-embed-v1-4b"
        assert embedder.prefix == "START"
        assert embedder.suffix == "END"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-api-key")

        embedder = PerplexityTextEmbedder()
        component_dict = embedder.to_dict()

        assert component_dict["init_parameters"]["api_key"] == Secret.from_env_var("PERPLEXITY_API_KEY").to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.perplexity.text_embedder.PerplexityTextEmbedder",
            "init_parameters": {
                "api_key": Secret.from_env_var("PERPLEXITY_API_KEY").to_dict(),
                "model": "pplx-embed-v1-0.6b",
                "api_base_url": "https://api.perplexity.ai/v1",
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-secret-key")
        embedder = PerplexityTextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="pplx-embed-v1-4b",
            api_base_url="https://custom-api-base-url.com",
            prefix="START",
            suffix="END",
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )

        component_dict = embedder.to_dict()

        assert component_dict == {
            "type": "haystack_integrations.components.embedders.perplexity.text_embedder.PerplexityTextEmbedder",
            "init_parameters": {
                "api_key": Secret.from_env_var("ENV_VAR", strict=False).to_dict(),
                "model": "pplx-embed-v1-4b",
                "api_base_url": "https://custom-api-base-url.com",
                "prefix": "START",
                "suffix": "END",
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.perplexity.text_embedder.PerplexityTextEmbedder",
            "init_parameters": {
                "api_key": Secret.from_env_var("PERPLEXITY_API_KEY").to_dict(),
                "model": "pplx-embed-v1-0.6b",
                "api_base_url": "https://api.perplexity.ai/v1",
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
                "http_client_kwargs": None,
            },
        }

        component = PerplexityTextEmbedder.from_dict(data)

        assert component.api_key == Secret.from_env_var(["PERPLEXITY_API_KEY"])
        assert component.api_base_url == "https://api.perplexity.ai/v1"
        assert component.model == "pplx-embed-v1-0.6b"
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.timeout is None
        assert component.max_retries is None
        assert component.http_client_kwargs is None

    def test_run_sends_attribution_header(self):
        captured: list[httpx.Request] = []
        embedder = PerplexityTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            http_client_kwargs={"transport": _make_transport(captured)},
        )

        result = embedder.run(text="The food was delicious")

        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert len(captured) == 1
        request = captured[0]
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert request.headers["X-Pplx-Integration"].startswith("haystack/")
        body = json.loads(request.content)
        assert body["model"] == "pplx-embed-v1-0.6b"
        assert body["input"] == "The food was delicious"
        assert body["encoding_format"] == "float"

    def test_run_wrong_input_format(self):
        embedder = PerplexityTextEmbedder(api_key=Secret.from_token("test-api-key"))
        match_error_msg = (
            "OpenAITextEmbedder expects a string as an input.In case you want to embed a list of Documents,"
            " please use the OpenAIDocumentEmbedder."
        )

        with pytest.raises(TypeError, match=match_error_msg):
            embedder.run(text=["text_snippet_1", "text_snippet_2"])


@pytest.mark.skipif(
    not os.environ.get("PERPLEXITY_API_KEY"),
    reason="Export PERPLEXITY_API_KEY to run integration tests.",
)
@pytest.mark.integration
class TestPerplexityTextEmbedderInference:
    def test_live_run(self):
        text = "The capital of France is Paris."
        embedder = PerplexityTextEmbedder()
        result = embedder.run(text=text)

        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.asyncio
    async def test_live_run_async(self):
        text = "The capital of France is Paris."
        embedder = PerplexityTextEmbedder()
        result = await embedder.run_async(text=text)

        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0
