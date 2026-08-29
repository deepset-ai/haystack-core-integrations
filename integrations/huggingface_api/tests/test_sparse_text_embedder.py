# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret

from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPISparseTextEmbedder
from haystack_integrations.components.embedders.huggingface_api._grpc import tei_pb2

API_BASE_URL = "http://localhost:8080"
MODULE = "haystack_integrations.components.embedders.huggingface_api.sparse_text_embedder"


def sparse_response(data: Any) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = data
    return response


@contextmanager
def patched_grpc() -> Iterator[tuple[MagicMock, MagicMock, MagicMock, MagicMock]]:
    sync_channel = MagicMock()
    async_channel = MagicMock()
    sync_stub = MagicMock()
    async_stub = MagicMock()
    async_stub.EmbedSparse = AsyncMock()
    with (
        patch(f"{MODULE}.grpc.insecure_channel", return_value=sync_channel),
        patch(f"{MODULE}.grpc.aio.insecure_channel", return_value=async_channel),
        patch(f"{MODULE}.tei_pb2_grpc.EmbedStub", side_effect=[sync_stub, async_stub]),
    ):
        yield sync_channel, async_channel, sync_stub, async_stub


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
        assert not embedder.use_grpc

    def test_init_grpc_uses_api_base_url_as_target(self) -> None:
        sync_channel = MagicMock()
        async_channel = MagicMock()
        sync_stub = MagicMock()
        async_stub = MagicMock()
        with (
            patch(f"{MODULE}.grpc.insecure_channel", return_value=sync_channel) as sync_channel_constructor,
            patch(f"{MODULE}.grpc.aio.insecure_channel", return_value=async_channel) as async_channel_constructor,
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub", side_effect=[sync_stub, async_stub]) as stub_constructor,
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)

        sync_channel_constructor.assert_called_once_with("localhost:8082")
        async_channel_constructor.assert_called_once_with("localhost:8082")
        assert stub_constructor.call_args_list == [call(sync_channel), call(async_channel)]
        assert embedder.api_base_url == "localhost:8082"
        assert embedder.use_grpc
        assert embedder._stub is sync_stub
        assert embedder._async_stub is async_stub

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
                "use_grpc": False,
            },
        }
        restored = HuggingFaceAPISparseTextEmbedder.from_dict(data)
        assert restored.api_base_url == embedder.api_base_url
        assert restored.prefix == "query: "
        assert restored.suffix == "!"
        assert restored.timeout is None
        assert restored.headers == {"X-Tenant": "test"}
        assert not restored.use_grpc
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

    def test_run_grpc_calls_embed_sparse_and_converts_response(self) -> None:
        with patched_grpc() as (_, _, sync_stub, _):
            sync_stub.EmbedSparse.return_value = SimpleNamespace(
                sparse_embeddings=[SimpleNamespace(index=12, value=1.0), SimpleNamespace(index=99, value=0.25)]
            )
            embedder = HuggingFaceAPISparseTextEmbedder(
                api_base_url="localhost:8082", prefix="query: ", suffix=" </s>", use_grpc=True
            )
            result = embedder.run("cheese")

        sync_stub.EmbedSparse.assert_called_once_with(tei_pb2.EmbedSparseRequest(inputs="query: cheese </s>"))
        assert result == {"sparse_embedding": SparseEmbedding(indices=[12, 99], values=[1.0, 0.25])}
        assert all(isinstance(index, int) for index in result["sparse_embedding"].indices)
        assert all(isinstance(value, float) for value in result["sparse_embedding"].values)

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
    async def test_run_async_grpc_awaits_embed_sparse_and_converts_response(self) -> None:
        with patched_grpc() as (_, _, _, async_stub):
            async_stub.EmbedSparse.return_value = SimpleNamespace(
                sparse_embeddings=[SimpleNamespace(index=7, value=2.5)]
            )
            embedder = HuggingFaceAPISparseTextEmbedder(
                api_base_url="localhost:8082", prefix="query: ", suffix="!", use_grpc=True
            )
            result = await embedder.run_async("input")

        async_stub.EmbedSparse.assert_awaited_once_with(tei_pb2.EmbedSparseRequest(inputs="query: input!"))
        assert result == {"sparse_embedding": SparseEmbedding(indices=[7], values=[2.5])}
        assert isinstance(result["sparse_embedding"].indices[0], int)
        assert isinstance(result["sparse_embedding"].values[0], float)

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
    # `use_grpc` eagerly initializes an aio channel, so this test must run in an active event loop.
    @pytest.mark.asyncio
    async def test_live_run_tei_grpc(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)
        try:
            result = embedder.run("sparse retrieval")
        finally:
            await embedder._async_channel.close()

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async_tei_grpc(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)
        try:
            result = await embedder.run_async("sparse retrieval")
        finally:
            await embedder._async_channel.close()

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices

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
