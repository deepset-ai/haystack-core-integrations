# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret

from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPISparseTextEmbedder

API_BASE_URL = "http://localhost:8080"


def sparse_response(data):
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = data
    return response


class TestHuggingFaceAPISparseTextEmbedder:
    @pytest.mark.parametrize("api_base_url", ["not-a-url", "ftp://localhost/path", "localhost:8080"])
    def test_init_rejects_invalid_api_base_url(self, api_base_url):
        with pytest.raises(ValueError, match="api_base_url must be a valid HTTP URL"):
            HuggingFaceAPISparseTextEmbedder(api_base_url=api_base_url)

    def test_init_defaults(self):
        embedder = HuggingFaceAPISparseTextEmbedder()

        assert embedder.api_base_url == "http://localhost:8080"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.timeout == 30.0
        assert embedder.headers == {}

    def test_to_dict_and_from_dict_preserve_env_secret(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_HF_TOKEN", "resolved-token")
        embedder = HuggingFaceAPISparseTextEmbedder(
            api_base_url="https://tei.example.test/base/",
            token=Secret.from_env_var("CUSTOM_HF_TOKEN"),
            prefix="query: ",
            suffix="!",
            timeout=None,
            headers={"X-Tenant": "test"},
        )

        data = embedder.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.huggingface_api.sparse_text_embedder."
            "HuggingFaceAPISparseTextEmbedder",
            "init_parameters": {
                "api_base_url": "https://tei.example.test/base/",
                "token": {"type": "env_var", "env_vars": ["CUSTOM_HF_TOKEN"], "strict": True},
                "prefix": "query: ",
                "suffix": "!",
                "timeout": None,
                "headers": {"X-Tenant": "test"},
            },
        }
        restored = HuggingFaceAPISparseTextEmbedder.from_dict(data)
        assert restored.api_base_url == embedder.api_base_url
        assert restored.prefix == "query: "
        assert restored.suffix == "!"
        assert restored.timeout is None
        assert restored.headers == {"X-Tenant": "test"}
        assert restored.token is not None
        assert restored.token.resolve_value() == "resolved-token"
        assert "_client" not in data["init_parameters"]
        assert "_async_client" not in data["init_parameters"]

    def test_token_secret_cannot_be_serialized(self):
        embedder = HuggingFaceAPISparseTextEmbedder(token=Secret.from_token("do-not-serialize"))

        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            embedder.to_dict()

    @pytest.mark.parametrize("invalid_text", [None, 42, ["text"]])
    def test_run_rejects_non_string_input(self, invalid_text):
        embedder = HuggingFaceAPISparseTextEmbedder()

        with pytest.raises(TypeError, match="expects a string"):
            embedder.run(invalid_text)

    def test_run_posts_tei_request_and_converts_response(self):
        response = sparse_response([[{"index": 12, "value": 1}, {"index": 99, "value": 0.25}]])
        embedder = HuggingFaceAPISparseTextEmbedder(
            api_base_url="https://tei.example.test/api/",
            token=Secret.from_token("secret"),
            prefix="query: ",
            suffix=" </s>",
            timeout=4.5,
            headers={"X-Tenant": "one", "Authorization": "custom"},
        )

        client = MagicMock(spec=httpx.Client)
        client.post.return_value = response
        embedder._client = client

        result = embedder.run("cheese")

        client.post.assert_called_once_with("embed_sparse", json={"inputs": "query: cheese </s>"})
        response.raise_for_status.assert_called_once_with()
        assert result == {"sparse_embedding": SparseEmbedding(indices=[12, 99], values=[1.0, 0.25])}

    @pytest.mark.asyncio
    async def test_run_async_uses_async_client_and_request_options(self):
        response = sparse_response([[{"index": 7, "value": 2.5}]])
        client = MagicMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=response)
        embedder = HuggingFaceAPISparseTextEmbedder(
            api_base_url="http://tei:8080/", token=Secret.from_token("token"), timeout=9, headers={"X-Test": "yes"}
        )
        embedder._async_client = client

        result = await embedder.run_async("input")

        client.post.assert_awaited_once_with("embed_sparse", json={"inputs": "input"})
        response.raise_for_status.assert_called_once_with()
        assert result["sparse_embedding"] == SparseEmbedding(indices=[7], values=[2.5])

    @pytest.mark.parametrize("payload", [{"index": 1, "value": 0.5}, [], [[{"index": 1}]]])
    def test_run_rejects_malformed_tei_responses(self, payload):
        embedder = HuggingFaceAPISparseTextEmbedder()
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = sparse_response(payload)

        with pytest.raises(ValueError):
            embedder.run("text")

    def test_run_propagates_http_error(self):
        request = httpx.Request("POST", "http://localhost:8080/embed_sparse")
        error_response = httpx.Response(503, request=request)

        embedder = HuggingFaceAPISparseTextEmbedder()
        embedder._client = MagicMock(spec=httpx.Client)
        embedder._client.post.return_value = error_response

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            embedder.run("text")

        assert exc_info.value.response.status_code == 503

    @pytest.mark.integration
    def test_live_run_tei(self):
        result = HuggingFaceAPISparseTextEmbedder(api_base_url=API_BASE_URL).run("sparse retrieval")

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices
