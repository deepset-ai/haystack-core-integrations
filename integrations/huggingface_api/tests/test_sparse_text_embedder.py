# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret

from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPISparseTextEmbedder

API_BASE_URL = "http://localhost:8080"
MODULE = "haystack_integrations.components.embedders.huggingface_api.sparse_text_embedder"


def sparse_response(data: Any) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = data
    return response


@contextmanager
def patched_client(*, is_async: bool = False) -> Iterator[tuple[MagicMock, MagicMock]]:
    """
    Patch the `httpx` client constructor and yield the (client, constructor) mocks.

    The component builds a client per call, so tests reach the client through the constructor rather than through
    an attribute. `__enter__`/`__aenter__` yield the same mock, so `client.post` assertions read naturally.
    """
    name = "AsyncClient" if is_async else "Client"
    client = MagicMock(spec=getattr(httpx, name))
    if is_async:
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False
    else:
        client.__enter__.return_value = client
        client.__exit__.return_value = False
    with patch(f"{MODULE}.httpx.{name}", return_value=client) as constructor:
        yield client, constructor


class TestHuggingFaceAPISparseTextEmbedder:
    @pytest.mark.parametrize("api_base_url", ["not-a-url", "ftp://localhost/path", "localhost:8080"])
    def test_init_rejects_invalid_api_base_url(self, api_base_url: str) -> None:
        with pytest.raises(ValueError, match="api_base_url must be a valid HTTP URL"):
            HuggingFaceAPISparseTextEmbedder(api_base_url=api_base_url)

    def test_init_defaults(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder()

        assert embedder.api_base_url == "http://localhost:8080"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.timeout == 30.0
        assert embedder.headers == {}

    def test_to_dict_and_from_dict_preserve_env_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_token_secret_cannot_be_serialized(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(token=Secret.from_token("do-not-serialize"))

        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            embedder.to_dict()

    @pytest.mark.parametrize("invalid_text", [None, 42, ["text"]])
    def test_run_rejects_non_string_input(self, invalid_text: Any) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder()

        with pytest.raises(TypeError, match="expects a string"):
            embedder.run(invalid_text)

    def test_run_posts_tei_request_and_converts_response(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(
            api_base_url="https://tei.example.test/api/",
            token=Secret.from_token("secret"),
            prefix="query: ",
            suffix=" </s>",
            timeout=4.5,
            headers={"X-Tenant": "one"},
        )

        with patched_client() as (client, constructor):
            response = sparse_response([[{"index": 12, "value": 1}, {"index": 99, "value": 0.25}]])
            client.post.return_value = response
            result = embedder.run("cheese")

        constructor.assert_called_once_with(
            base_url="https://tei.example.test/api/",
            timeout=4.5,
            headers={"Authorization": "Bearer secret", "X-Tenant": "one"},
        )
        client.post.assert_called_once_with("embed_sparse", json={"inputs": "query: cheese </s>"})
        response.raise_for_status.assert_called_once_with()
        assert result == {"sparse_embedding": SparseEmbedding(indices=[12, 99], values=[1.0, 0.25])}

    def test_token_is_sent_as_bearer_authorization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "env-token")
        embedder = HuggingFaceAPISparseTextEmbedder()

        with patched_client() as (client, constructor):
            client.post.return_value = sparse_response([[{"index": 1, "value": 1}]])
            embedder.run("text")

        assert constructor.call_args.kwargs["headers"] == {"Authorization": "Bearer env-token"}

    def test_explicit_authorization_header_wins_over_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit header must not be replaced by a token that only happens to be set in the environment."""
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "env-token")
        embedder = HuggingFaceAPISparseTextEmbedder(headers={"Authorization": "Basic test-key"})

        with patched_client() as (client, constructor):
            client.post.return_value = sparse_response([[{"index": 1, "value": 1}]])
            embedder.run("text")

        assert constructor.call_args.kwargs["headers"] == {"Authorization": "Basic test-key"}

    def test_token_is_resolved_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The token is read at request time, so a rotated environment variable is picked up."""
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "first-token")
        embedder = HuggingFaceAPISparseTextEmbedder()

        with patched_client() as (client, constructor):
            client.post.return_value = sparse_response([[{"index": 1, "value": 1}]])
            embedder.run("text")
            monkeypatch.setenv("HF_TOKEN", "second-token")
            embedder.run("text")

        assert [each.kwargs["headers"]["Authorization"] for each in constructor.call_args_list] == [
            "Bearer first-token",
            "Bearer second-token",
        ]

    @pytest.mark.asyncio
    async def test_run_async_uses_async_client_and_request_options(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(
            api_base_url="http://tei:8080/", token=Secret.from_token("token"), timeout=9, headers={"X-Test": "yes"}
        )

        with patched_client(is_async=True) as (client, constructor):
            response = sparse_response([[{"index": 7, "value": 2.5}]])
            client.post.return_value = response
            result = await embedder.run_async("input")

        constructor.assert_called_once_with(
            base_url="http://tei:8080/", timeout=9, headers={"Authorization": "Bearer token", "X-Test": "yes"}
        )
        client.post.assert_awaited_once_with("embed_sparse", json={"inputs": "input"})
        response.raise_for_status.assert_called_once_with()
        assert result["sparse_embedding"] == SparseEmbedding(indices=[7], values=[2.5])

    @pytest.mark.asyncio
    async def test_run_async_builds_and_closes_a_client_per_call(self) -> None:
        """
        The component must not hold on to an `httpx.AsyncClient`.

        A cached client binds its keep-alive connection pool to the first event loop that used it, so a component
        reused under a second `asyncio.run` would fail with `RuntimeError: Event loop is closed`.
        """
        embedder = HuggingFaceAPISparseTextEmbedder()

        with patched_client(is_async=True) as (client, constructor):
            client.post.return_value = sparse_response([[{"index": 1, "value": 1}]])
            await embedder.run_async("one")
            await embedder.run_async("two")

        assert constructor.call_count == 2
        assert client.__aexit__.await_count == 2
        assert not hasattr(embedder, "_async_client")

    def test_run_builds_and_closes_a_client_per_call(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder()

        with patched_client() as (client, constructor):
            client.post.return_value = sparse_response([[{"index": 1, "value": 1}]])
            embedder.run("one")
            embedder.run("two")

        assert constructor.call_count == 2
        assert client.__exit__.call_count == 2
        assert not hasattr(embedder, "_client")

    @pytest.mark.parametrize(
        "payload",
        [
            {"index": 1, "value": 0.5},
            [],
            ["not-a-list"],
            [[{"index": 1}]],
            [[{"index": "1", "value": 0.5}]],
            [[{"index": 1, "value": True}]],
        ],
    )
    def test_run_rejects_malformed_tei_responses(self, payload: Any) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder()

        with patched_client() as (client, _), pytest.raises(ValueError):
            client.post.return_value = sparse_response(payload)
            embedder.run("text")

    def test_run_propagates_http_error(self) -> None:
        request = httpx.Request("POST", "http://localhost:8080/embed_sparse")
        embedder = HuggingFaceAPISparseTextEmbedder()

        with patched_client() as (client, _), pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.post.return_value = httpx.Response(503, request=request)
            embedder.run("text")

        assert exc_info.value.response.status_code == 503

    @pytest.mark.integration
    def test_live_run_tei(self) -> None:
        result = HuggingFaceAPISparseTextEmbedder(api_base_url=API_BASE_URL).run("sparse retrieval")

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async_tei(self) -> None:
        result = await HuggingFaceAPISparseTextEmbedder(api_base_url=API_BASE_URL).run_async("sparse retrieval")

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices
