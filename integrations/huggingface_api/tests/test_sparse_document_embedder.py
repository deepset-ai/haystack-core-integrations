# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret

from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPISparseDocumentEmbedder

API_BASE_URL = "http://localhost:8080"
MODULE = "haystack_integrations.components.embedders.huggingface_api.sparse_document_embedder"


def sparse_response(data: Any) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = data
    return response


def grpc_sparse_response(index: int, value: float) -> dict[str, list[dict[str, float | int]]]:
    return {"sparse_embeddings": [{"index": index, "value": value}]}


@contextmanager
def patched_grpc() -> Iterator[tuple[MagicMock, MagicMock, MagicMock, MagicMock]]:
    sync_client = MagicMock(channel=MagicMock())
    async_client = MagicMock(channel=MagicMock(close=AsyncMock()))
    with (
        patch(f"{MODULE}.Client", return_value=sync_client) as sync_constructor,
        patch(f"{MODULE}.AsyncClient.create", new=AsyncMock(return_value=async_client)) as async_constructor,
    ):
        sync_client._constructor = sync_constructor
        async_client._constructor = async_constructor
        yield sync_client, async_client, sync_client, async_client


@contextmanager
def patched_client(*, is_async: bool = False) -> Iterator[tuple[MagicMock, MagicMock]]:
    """Patch an `httpx` client constructor and yield the client and constructor mocks."""
    name = "AsyncClient" if is_async else "Client"
    client = MagicMock(spec=getattr(httpx, name))
    with patch(f"{MODULE}.httpx.{name}", return_value=client) as constructor:
        yield client, constructor


class TestHuggingFaceAPISparseDocumentEmbedder:
    @pytest.mark.parametrize("api_base_url", ["not-a-url", "file:///path", "localhost:8080"])
    def test_init_rejects_invalid_api_base_url(self, api_base_url: str) -> None:
        with pytest.raises(ValueError, match="api_base_url must be a valid HTTP URL"):
            HuggingFaceAPISparseDocumentEmbedder(api_base_url=api_base_url)

    @pytest.mark.parametrize(("parameter", "value"), [("batch_size", 0), ("batch_size", -2), ("concurrency_limit", 0)])
    def test_init_rejects_non_positive_numeric_options(self, parameter: str, value: int) -> None:
        with pytest.raises(ValueError, match=f"{parameter} must be > 0"):
            HuggingFaceAPISparseDocumentEmbedder(**{parameter: value})

    def test_init_defaults(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder()

        assert embedder.api_base_url == "http://localhost:8080"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.timeout == 30.0
        assert embedder.headers == {}
        assert embedder.concurrency_limit == 4
        assert embedder.use_grpc is False
        assert embedder._client is None
        assert embedder._async_client is None
        assert embedder._grpc_client is None
        assert embedder._async_grpc_client is None

    def test_init_grpc_does_not_create_clients(self) -> None:
        with (
            patch(f"{MODULE}.Client") as sync_constructor,
            patch(f"{MODULE}.AsyncClient.create") as async_constructor,
        ):
            embedder = HuggingFaceAPISparseDocumentEmbedder(api_base_url="localhost:8082", use_grpc=True)

        sync_constructor.assert_not_called()
        async_constructor.assert_not_called()
        assert embedder.use_grpc is True
        assert embedder._client is None
        assert embedder._async_client is None
        assert embedder._grpc_client is None
        assert embedder._async_grpc_client is None

    def test_to_dict_and_from_dict_preserve_configuration_and_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUSTOM_HF_TOKEN", "secret")
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="https://tei.example.test",
            token=Secret.from_env_var("CUSTOM_HF_TOKEN"),
            prefix="passage: ",
            suffix="!",
            batch_size=2,
            progress_bar=False,
            meta_fields_to_embed=["title"],
            embedding_separator=" | ",
            timeout=None,
            headers={"X-Test": "yes"},
            concurrency_limit=3,
        )

        data = embedder.to_dict()

        assert data["init_parameters"]["token"] == {
            "type": "env_var",
            "env_vars": ["CUSTOM_HF_TOKEN"],
            "strict": True,
        }
        assert data["init_parameters"] == {
            "api_base_url": "https://tei.example.test",
            "token": {"type": "env_var", "env_vars": ["CUSTOM_HF_TOKEN"], "strict": True},
            "prefix": "passage: ",
            "suffix": "!",
            "batch_size": 2,
            "progress_bar": False,
            "meta_fields_to_embed": ["title"],
            "embedding_separator": " | ",
            "timeout": None,
            "headers": {"X-Test": "yes"},
            "concurrency_limit": 3,
            "use_grpc": False,
        }
        restored = HuggingFaceAPISparseDocumentEmbedder.from_dict(data)

        assert restored.api_base_url == embedder.api_base_url
        assert restored.prefix == "passage: "
        assert restored.suffix == "!"
        assert restored.batch_size == 2
        assert restored.progress_bar is False
        assert restored.meta_fields_to_embed == ["title"]
        assert restored.embedding_separator == " | "
        assert restored.timeout is None
        assert restored.headers == {"X-Test": "yes"}
        assert restored.concurrency_limit == 3
        assert restored.use_grpc is False
        assert restored.token is not None and restored.token.resolve_value() == "secret"

    def test_token_secret_cannot_be_serialized(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(token=Secret.from_token("do-not-serialize"))

        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            embedder.to_dict()


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_HF_TOKEN", raising=False)
        embedder = HuggingFaceAPISparseDocumentEmbedder(token=Secret.from_env_var("MISSING_HF_TOKEN"))

        with patch(f"{MODULE}.httpx.Client") as constructor, pytest.raises(ValueError, match="MISSING_HF_TOKEN"):
            embedder.warm_up()

        constructor.assert_not_called()
        assert embedder._client is None

    def test_sync_lifecycle(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(token=None)

        with patched_client() as (client, constructor):
            embedder.warm_up()
            embedder.warm_up()

            constructor.assert_called_once_with(base_url=f"{API_BASE_URL}/", timeout=30.0, headers={})
            assert embedder._client is client
            assert embedder._async_client is None

            embedder.close()
            client.close.assert_called_once_with()
            assert embedder._client is None

            embedder.warm_up()
            assert constructor.call_count == 2
            embedder.close()

    @pytest.mark.asyncio
    async def test_async_lifecycle(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(token=None)

        with patched_client(is_async=True) as (client, constructor):
            await embedder.warm_up_async()
            await embedder.warm_up_async()

            constructor.assert_called_once_with(base_url=f"{API_BASE_URL}/", timeout=30.0, headers={})
            assert embedder._async_client is client
            assert embedder._client is None

            await embedder.close_async()
            client.aclose.assert_awaited_once_with()
            assert embedder._async_client is None

            await embedder.warm_up_async()
            assert constructor.call_count == 2
            await embedder.close_async()

    def test_grpc_sync_lifecycle(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(api_base_url="localhost:8082", use_grpc=True)
        with patched_grpc() as (client, async_client, _, _):
            embedder.warm_up()
            embedder.warm_up()
            client._constructor.assert_called_once_with("localhost:8082")
            async_client._constructor.assert_not_called()
            assert embedder._grpc_client is client
            assert embedder._async_grpc_client is None

            embedder.close()
            client.channel.close.assert_called_once_with()
            assert embedder._grpc_client is None
            embedder.warm_up()
            assert client._constructor.call_count == 2

    @pytest.mark.asyncio
    async def test_grpc_async_lifecycle(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(api_base_url="localhost:8082", use_grpc=True)
        with patched_grpc() as (sync_client, client, _, _):
            await embedder.warm_up_async()
            await embedder.warm_up_async()
            client._constructor.assert_awaited_once_with("localhost:8082")
            sync_client._constructor.assert_not_called()
            assert embedder._async_grpc_client is client
            assert embedder._grpc_client is None

            await embedder.close_async()
            client.channel.close.assert_awaited_once_with()
            assert embedder._async_grpc_client is None
            await embedder.warm_up_async()
            assert client._constructor.await_count == 2

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder()

        embedder.close()
        await embedder.close_async()

        assert embedder._client is None
        assert embedder._async_client is None
        assert embedder._grpc_client is None
        assert embedder._async_grpc_client is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("api_base_url", "use_grpc"), [(API_BASE_URL, False), ("localhost:8082", True)], ids=["http", "grpc"]
    )
    async def test_close_and_close_async_are_independent(self, api_base_url: str, use_grpc: bool) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(api_base_url=api_base_url, use_grpc=use_grpc, token=None)
        with (
            patched_client() as (sync_http_client, _),
            patched_client(is_async=True) as (async_http_client, _),
            patched_grpc() as (sync_grpc_client, async_grpc_client, _, _),
        ):
            embedder.warm_up()
            await embedder.warm_up_async()
            sync_client = sync_grpc_client if use_grpc else sync_http_client
            async_client = async_grpc_client if use_grpc else async_http_client
            sync_attr = "_grpc_client" if use_grpc else "_client"
            async_attr = "_async_grpc_client" if use_grpc else "_async_client"
            embedder.close()
            assert getattr(embedder, sync_attr) is None
            assert getattr(embedder, async_attr) is async_client
            async_close = async_client.channel.close if use_grpc else async_client.aclose
            async_close.assert_not_awaited()

            await embedder.close_async()
            assert getattr(embedder, async_attr) is None
            (sync_client.channel.close if use_grpc else sync_client.close).assert_called_once_with()


class TestRun:
    def test_prepare_texts_handles_metadata_prefix_suffix_and_missing_content(self) -> None:
        documents = [
            Document(content="body", meta={"title": "Title", "priority": 3, "ignored": "no"}),
            Document(content=None, meta={"title": None, "priority": 0}),
            Document(content="only body", meta={}),
        ]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            prefix="<p>", suffix="</p>", meta_fields_to_embed=["title", "priority"], embedding_separator=" | "
        )

        assert embedder._prepare_texts_to_embed(documents) == [
            "<p>Title | 3 | body</p>",
            "<p>0 | </p>",
            "<p>only body</p>",
        ]

    @pytest.mark.parametrize("documents", [None, "document", [1, 2], [Document(content="valid"), "invalid"]])
    def test_run_rejects_invalid_and_mixed_inputs(self, documents: Any) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)
        with patched_client(), pytest.raises(TypeError, match="expects a list of Documents"):
            embedder.run(documents)
        embedder.close()

    @pytest.mark.asyncio
    async def test_run_async_rejects_mixed_inputs(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)
        with patched_client(is_async=True), pytest.raises(TypeError, match="expects a list of Documents"):
            await embedder.run_async([Document(content="ok"), None])
        await embedder.close_async()

    def test_empty_list_returns_without_http_request(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)

        with patched_client() as (client, _):
            result = embedder.run([])

        assert result == {"documents": []}
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_list_async_returns_without_http_request(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)

        with patched_client(is_async=True) as (client, _):
            result = await embedder.run_async([])

        assert result == {"documents": []}
        client.post.assert_not_called()

    def test_run_batches_requests_preserves_order_and_copies_documents(self) -> None:
        documents = [Document(content=f"doc {number}", meta={"number": number}) for number in range(5)]
        responses = [
            sparse_response([[{"index": 10, "value": 1}], [{"index": 11, "value": 2}]]),
            sparse_response([[{"index": 12, "value": 3}], [{"index": 13, "value": 4}]]),
            sparse_response([[{"index": 14, "value": 5}]]),
        ]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="http://tei:80/root/",
            token=Secret.from_token("token"),
            prefix="passage: ",
            batch_size=2,
            progress_bar=False,
            timeout=6,
            headers={"X-Tenant": "tenant"},
        )

        with patched_client() as (client, constructor):
            client.post.side_effect = responses
            result = embedder.run(documents)

        constructor.assert_called_once_with(
            base_url="http://tei:80/root/",
            timeout=6,
            headers={"Authorization": "Bearer token", "X-Tenant": "tenant"},
        )
        assert client.post.call_args_list == [
            call("embed_sparse", json={"inputs": ["passage: doc 0", "passage: doc 1"]}),
            call("embed_sparse", json={"inputs": ["passage: doc 2", "passage: doc 3"]}),
            call("embed_sparse", json={"inputs": ["passage: doc 4"]}),
        ]
        output = result["documents"]
        assert [document.sparse_embedding.indices for document in output] == [[10], [11], [12], [13], [14]]
        assert [document.sparse_embedding.values for document in output] == [[1.0], [2.0], [3.0], [4.0], [5.0]]
        assert all(new is not original for original, new in zip(documents, output, strict=True))
        assert all(original.sparse_embedding is None for original in documents)
        assert [document.meta for document in output] == [document.meta for document in documents]

    def test_run_grpc_uses_one_stream_converts_embeddings_and_reuses_client(self) -> None:
        documents = [Document(content="doc 1"), Document(content="doc 2")]
        requests: list[dict[str, Any]] = []

        def embed_sparse_stream(service: str, method: str, stream: Any) -> list[dict[str, Any]]:
            assert (service, method) == ("tei.v1.Embed", "EmbedSparseStream")
            stream_requests = list(stream)
            requests.extend(stream_requests)
            return [grpc_sparse_response(position, float(position)) for position, _ in enumerate(stream_requests, 1)]

        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="localhost:8082", use_grpc=True, batch_size=1, progress_bar=False
        )
        with patched_grpc() as (client, _, _, _):
            client.stream_stream.side_effect = embed_sparse_stream
            result = embedder.run(documents)
            embedder.run(documents)

        assert client.stream_stream.call_count == 2
        assert client._constructor.call_count == 1
        assert [request["inputs"] for request in requests] == ["doc 1", "doc 2", "doc 1", "doc 2"]
        assert [document.sparse_embedding.indices for document in result["documents"]] == [[1], [2]]
        assert [document.sparse_embedding.values for document in result["documents"]] == [[1.0], [2.0]]
        assert embedder._grpc_client is client
        client.channel.close.assert_not_called()
        embedder.close()
        client.channel.close.assert_called_once_with()

    def test_run_grpc_resource_can_be_closed_after_stream_response_count_error(self) -> None:
        documents = [Document(content="doc 1"), Document(content="doc 2")]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="localhost:8082", use_grpc=True, progress_bar=False
        )
        with patched_grpc() as (client, _, _, _):
            client.stream_stream.return_value = [grpc_sparse_response(1, 1.0)]
            with pytest.raises(ValueError, match="Expected 2 sparse embeddings, got 1"):
                embedder.run(documents)

        assert embedder._grpc_client is client
        client.channel.close.assert_not_called()
        embedder.close()
        client.channel.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_run_async_grpc_uses_balanced_concurrent_streams_and_preserves_order(self) -> None:
        streams: list[list[str]] = []
        all_streams_started = asyncio.Event()

        async def embed_sparse_stream(service: str, method: str, stream: Any) -> Any:
            assert (service, method) == ("tei.v1.Embed", "EmbedSparseStream")
            stream_requests: list[str] = []
            streams.append(stream_requests)
            if len(streams) == 3:
                all_streams_started.set()

            async def responses() -> Any:
                await all_streams_started.wait()
                async for request in stream:
                    text = request["inputs"]
                    stream_requests.append(text)
                    number = int(text.removeprefix("doc "))
                    yield grpc_sparse_response(number, float(number))

            return responses()

        documents = [Document(content=f"doc {number}") for number in range(1, 8)]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="localhost:8082", use_grpc=True, concurrency_limit=3, batch_size=1, progress_bar=False
        )
        with patched_grpc() as (_, client, _, _):
            client.stream_stream = AsyncMock(side_effect=embed_sparse_stream)
            result = await embedder.run_async(documents)
            await embedder.run_async(documents)

        assert client._constructor.await_count == 1
        assert embedder._async_grpc_client is client
        client.channel.close.assert_not_awaited()
        await embedder.close_async()
        client.channel.close.assert_awaited_once_with()
        assert streams == [
            ["doc 1", "doc 2"],
            ["doc 3", "doc 4"],
            ["doc 5", "doc 6", "doc 7"],
            ["doc 1", "doc 2"],
            ["doc 3", "doc 4"],
            ["doc 5", "doc 6", "doc 7"],
        ]
        assert [document.sparse_embedding.indices for document in result["documents"]] == [
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
        ]

    @pytest.mark.asyncio
    async def test_run_async_grpc_resource_can_be_closed_after_stream_error(self) -> None:
        client_error = RuntimeError("stream failed")
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="localhost:8082", use_grpc=True, progress_bar=False
        )
        with patched_grpc() as (_, client, _, _):
            client.stream_stream = AsyncMock(side_effect=client_error)
            with pytest.raises(RuntimeError, match="stream failed"):
                await embedder.run_async([Document(content="doc")])

        assert embedder._async_grpc_client is client
        client.channel.close.assert_not_awaited()
        await embedder.close_async()
        client.channel.close.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_run_async_batches_with_one_client_and_preserves_batch_order(self) -> None:
        async def post(_url: str, *, json: dict[str, list[str]]) -> MagicMock:
            inputs = json["inputs"]
            # Let the second batch finish first; gather must still preserve input order.
            await asyncio.sleep(0.01 if inputs[0] == "doc 0" else 0)
            offset = int(inputs[0].split()[-1])
            return sparse_response([[{"index": offset + position, "value": 1}] for position, _ in enumerate(inputs)])

        documents = [Document(content=f"doc {number}") for number in range(4)]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="https://tei.test/", batch_size=2, progress_bar=False, timeout=None, headers={"X-Test": "yes"}
        )

        with patched_client(is_async=True) as (client, constructor):
            client.post.side_effect = post
            result = await embedder.run_async(documents)

        # Both concurrent batches share the component's persistent async client.
        constructor.assert_called_once_with(base_url="https://tei.test/", timeout=None, headers={"X-Test": "yes"})
        assert client.post.await_args_list == [
            call("embed_sparse", json={"inputs": ["doc 0", "doc 1"]}),
            call("embed_sparse", json={"inputs": ["doc 2", "doc 3"]}),
        ]
        assert [document.sparse_embedding.indices for document in result["documents"]] == [[0], [1], [2], [3]]

    def test_explicit_authorization_header_wins_over_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit header must not be replaced by a token that only happens to be set in the environment."""
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "env-token")
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False, headers={"Authorization": "Basic test-key"})

        with patched_client() as (client, constructor):
            client.post.return_value = sparse_response([[{"index": 1, "value": 1}]])
            embedder.run([Document(content="one")])

        assert constructor.call_args.kwargs["headers"] == {"Authorization": "Basic test-key"}

    @pytest.mark.asyncio
    async def test_async_concurrency_limit_is_respected(self) -> None:
        active = 0
        maximum_active = 0

        async def embed_batch(**kwargs: Any) -> list[SparseEmbedding]:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [SparseEmbedding(indices=[int(text)], values=[1.0]) for text in kwargs["inputs"]]

        embedder = HuggingFaceAPISparseDocumentEmbedder(batch_size=1, concurrency_limit=2, progress_bar=False)
        client = MagicMock(spec=httpx.AsyncClient)
        embedder._async_client = client
        with patch(f"{MODULE}._embed_sparse_async", side_effect=embed_batch):
            embeddings = await embedder._embed_batches_async(["0", "1", "2", "3"])

        assert maximum_active == 2
        assert [embedding.indices for embedding in embeddings] == [[0], [1], [2], [3]]

    def test_run_rejects_response_with_wrong_embedding_count(self) -> None:
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)

        with patched_client() as (client, _):
            client.post.return_value = sparse_response([[{"index": 1, "value": 1}]])
            with pytest.raises(ValueError, match="Expected one sparse embedding per input"):
                embedder.run([Document(content="one"), Document(content="two")])

            client.close.assert_not_called()
            embedder.close()
            client.close.assert_called_once_with()

    def test_run_propagates_http_error(self) -> None:
        request = httpx.Request("POST", "http://localhost:8080/embed_sparse")
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)

        with patched_client() as (client, _):
            client.post.return_value = httpx.Response(500, request=request)
            with pytest.raises(httpx.HTTPStatusError):
                embedder.run([Document(content="text")])

            client.close.assert_not_called()
            embedder.close()
            client.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_run_async_resource_can_be_closed_after_http_error(self) -> None:
        request = httpx.Request("POST", "http://localhost:8080/embed_sparse")
        embedder = HuggingFaceAPISparseDocumentEmbedder(progress_bar=False)

        with patched_client(is_async=True) as (client, _):
            client.post.return_value = httpx.Response(500, request=request)
            with pytest.raises(httpx.HTTPStatusError):
                await embedder.run_async([Document(content="text")])

            client.aclose.assert_not_awaited()
            await embedder.close_async()
            client.aclose.assert_awaited_once_with()

    @pytest.mark.integration
    def test_live_run_tei_grpc(self) -> None:
        documents = [Document(content="sparse retrieval"), Document(content="dense retrieval")]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="localhost:8082", use_grpc=True, progress_bar=False
        )
        try:
            result = embedder.run(documents)
        finally:
            embedder.close()

        documents_with_embeddings = result["documents"]
        assert len(documents_with_embeddings) == len(documents)
        for document in documents_with_embeddings:
            assert isinstance(document.sparse_embedding, SparseEmbedding)
            assert document.sparse_embedding.indices

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async_tei_grpc(self) -> None:
        documents = [Document(content="sparse retrieval"), Document(content="dense retrieval")]
        embedder = HuggingFaceAPISparseDocumentEmbedder(
            api_base_url="localhost:8082", use_grpc=True, progress_bar=False
        )
        try:
            result = await embedder.run_async(documents)
        finally:
            await embedder.close_async()

        documents_with_embeddings = result["documents"]
        assert len(documents_with_embeddings) == len(documents)
        for document in documents_with_embeddings:
            assert isinstance(document.sparse_embedding, SparseEmbedding)
            assert document.sparse_embedding.indices

    @pytest.mark.integration
    def test_live_run_tei(self) -> None:
        documents = [Document(content="sparse retrieval"), Document(content="dense retrieval")]
        embedder = HuggingFaceAPISparseDocumentEmbedder(api_base_url=API_BASE_URL, progress_bar=False)
        try:
            result = embedder.run(documents)
        finally:
            embedder.close()

        documents_with_embeddings = result["documents"]
        assert len(documents_with_embeddings) == len(documents)
        for document in documents_with_embeddings:
            assert isinstance(document.sparse_embedding, SparseEmbedding)
            assert document.sparse_embedding.indices

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async_tei(self) -> None:
        documents = [Document(content="sparse retrieval"), Document(content="dense retrieval")]
        embedder = HuggingFaceAPISparseDocumentEmbedder(api_base_url=API_BASE_URL, progress_bar=False)
        try:
            result = await embedder.run_async(documents)
        finally:
            await embedder.close_async()

        documents_with_embeddings = result["documents"]
        assert len(documents_with_embeddings) == len(documents)
        for document in documents_with_embeddings:
            assert isinstance(document.sparse_embedding, SparseEmbedding)
            assert document.sparse_embedding.indices
