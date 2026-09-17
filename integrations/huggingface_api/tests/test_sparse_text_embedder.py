# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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


def sync_http_client(payload: Any = None) -> MagicMock:
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = sparse_response([[{"index": 1, "value": 1}]] if payload is None else payload)
    return client


def async_http_client(payload: Any = None) -> MagicMock:
    client = MagicMock(spec=httpx.AsyncClient)
    client.post.return_value = sparse_response([[{"index": 1, "value": 1}]] if payload is None else payload)
    return client


class TestHuggingFaceAPISparseTextEmbedder:
    @pytest.mark.parametrize("api_base_url", ["not-a-url", "ftp://localhost/path", "localhost:8080"])
    def test_init_rejects_invalid_api_base_url(self, api_base_url: str) -> None:
        with pytest.raises(ValueError, match="api_base_url must be a valid HTTP URL"):
            HuggingFaceAPISparseTextEmbedder(api_base_url=api_base_url)

    def test_init_defaults_and_resources_are_none(self) -> None:
        with (
            patch(f"{MODULE}.httpx.Client") as sync_client_constructor,
            patch(f"{MODULE}.httpx.AsyncClient") as async_client_constructor,
        ):
            embedder = HuggingFaceAPISparseTextEmbedder()

        assert embedder.api_base_url == "http://localhost:8080"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.timeout == 30.0
        assert embedder.headers == {}
        assert not embedder.use_grpc
        assert embedder._client is None
        assert embedder._async_client is None
        assert embedder._channel is None
        assert embedder._async_channel is None
        assert embedder._stub is None
        assert embedder._async_stub is None
        sync_client_constructor.assert_not_called()
        async_client_constructor.assert_not_called()

    def test_init_grpc_does_not_create_channels_or_stubs(self) -> None:
        with (
            patch(f"{MODULE}.grpc.insecure_channel") as sync_channel_constructor,
            patch(f"{MODULE}.grpc.aio.insecure_channel") as async_channel_constructor,
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub") as stub_constructor,
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)

        sync_channel_constructor.assert_not_called()
        async_channel_constructor.assert_not_called()
        stub_constructor.assert_not_called()
        assert embedder.api_base_url == "localhost:8082"
        assert embedder.use_grpc
        assert embedder._channel is None
        assert embedder._async_channel is None
        assert embedder._stub is None
        assert embedder._async_stub is None

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


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_HF_TOKEN", raising=False)
        embedder = HuggingFaceAPISparseTextEmbedder(token=Secret.from_env_var("MISSING_HF_TOKEN"))

        with pytest.raises(ValueError, match="MISSING_HF_TOKEN"):
            embedder.warm_up()

    def test_sync_lifecycle(self) -> None:
        first_client = sync_http_client()
        second_client = sync_http_client()
        with patch(f"{MODULE}.httpx.Client", side_effect=[first_client, second_client]) as constructor:
            embedder = HuggingFaceAPISparseTextEmbedder(token=None)
            constructor.assert_not_called()

            embedder.warm_up()
            embedder.warm_up()
            assert embedder._client is first_client
            assert embedder._async_client is None
            constructor.assert_called_once_with(base_url="http://localhost:8080/", timeout=30.0, headers={})

            embedder.close()
            first_client.close.assert_called_once_with()
            assert embedder._client is None

            embedder.warm_up()
            assert embedder._client is second_client
            embedder.close()

    @pytest.mark.asyncio
    async def test_async_lifecycle(self) -> None:
        first_client = async_http_client()
        second_client = async_http_client()
        with patch(f"{MODULE}.httpx.AsyncClient", side_effect=[first_client, second_client]) as constructor:
            embedder = HuggingFaceAPISparseTextEmbedder(token=None)
            constructor.assert_not_called()

            await embedder.warm_up_async()
            await embedder.warm_up_async()
            assert embedder._async_client is first_client
            assert embedder._client is None
            constructor.assert_called_once_with(base_url="http://localhost:8080/", timeout=30.0, headers={})

            await embedder.close_async()
            first_client.aclose.assert_awaited_once_with()
            assert embedder._async_client is None

            await embedder.warm_up_async()
            assert embedder._async_client is second_client
            await embedder.close_async()

    def test_grpc_sync_lifecycle(self) -> None:
        first_channel = MagicMock()
        second_channel = MagicMock()
        with (
            patch(f"{MODULE}.grpc.insecure_channel", side_effect=[first_channel, second_channel]) as constructor,
            patch(f"{MODULE}.grpc.aio.insecure_channel") as async_constructor,
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub") as stub_constructor,
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)
            constructor.assert_not_called()

            embedder.warm_up()
            embedder.warm_up()
            constructor.assert_called_once_with("localhost:8082")
            async_constructor.assert_not_called()
            stub_constructor.assert_called_once_with(first_channel)
            assert embedder._channel is first_channel
            assert embedder._async_channel is None

            embedder.close()
            first_channel.close.assert_called_once_with()
            assert embedder._channel is None
            assert embedder._stub is None

            embedder.warm_up()
            assert embedder._channel is second_channel
            embedder.close()

    @pytest.mark.asyncio
    async def test_grpc_async_lifecycle(self) -> None:
        first_channel = MagicMock(close=AsyncMock())
        second_channel = MagicMock(close=AsyncMock())
        with (
            patch(f"{MODULE}.grpc.insecure_channel") as sync_constructor,
            patch(f"{MODULE}.grpc.aio.insecure_channel", side_effect=[first_channel, second_channel]) as constructor,
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub") as stub_constructor,
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)
            constructor.assert_not_called()

            await embedder.warm_up_async()
            await embedder.warm_up_async()
            constructor.assert_called_once_with("localhost:8082")
            sync_constructor.assert_not_called()
            stub_constructor.assert_called_once_with(first_channel)
            assert embedder._async_channel is first_channel
            assert embedder._channel is None

            await embedder.close_async()
            first_channel.close.assert_awaited_once_with()
            assert embedder._async_channel is None
            assert embedder._async_stub is None

            await embedder.warm_up_async()
            assert embedder._async_channel is second_channel
            await embedder.close_async()

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder()
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None
        assert embedder._channel is None
        assert embedder._async_channel is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_grpc", [False, True])
    async def test_close_and_close_async_are_independent(self, use_grpc: bool) -> None:
        sync_resource = MagicMock()
        async_resource = MagicMock(close=AsyncMock()) if use_grpc else async_http_client()
        with (
            patch(f"{MODULE}.httpx.Client", return_value=sync_resource),
            patch(f"{MODULE}.httpx.AsyncClient", return_value=async_resource),
            patch(f"{MODULE}.grpc.insecure_channel", return_value=sync_resource),
            patch(f"{MODULE}.grpc.aio.insecure_channel", return_value=async_resource),
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub"),
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(
                api_base_url="localhost:8082" if use_grpc else API_BASE_URL, use_grpc=use_grpc
            )
            embedder.warm_up()
            await embedder.warm_up_async()

            sync_attr = "_channel" if use_grpc else "_client"
            async_attr = "_async_channel" if use_grpc else "_async_client"
            async_close = async_resource.close if use_grpc else async_resource.aclose
            embedder.close()
            assert getattr(embedder, sync_attr) is None
            assert getattr(embedder, async_attr) is async_resource
            sync_resource.close.assert_called_once_with()
            async_close.assert_not_awaited()

            await embedder.close_async()
            assert getattr(embedder, async_attr) is None
            async_close.assert_awaited_once_with()


class TestRun:
    @pytest.mark.parametrize("invalid_text", [None, 42, ["text"]])
    def test_run_rejects_non_string_input(self, invalid_text: Any) -> None:
        with patch(f"{MODULE}.httpx.Client", return_value=sync_http_client()):
            embedder = HuggingFaceAPISparseTextEmbedder()
            try:
                with pytest.raises(TypeError, match="expects a string"):
                    embedder.run(invalid_text)
            finally:
                embedder.close()

    def test_run_http_request_and_response(self) -> None:
        client = sync_http_client([[{"index": 12, "value": 1}, {"index": 99, "value": 0.25}]])
        with patch(f"{MODULE}.httpx.Client", return_value=client) as constructor:
            embedder = HuggingFaceAPISparseTextEmbedder(
                api_base_url="https://tei.example.test/api/",
                token=Secret.from_token("secret"),
                prefix="query: ",
                suffix=" </s>",
                timeout=4.5,
                headers={"X-Tenant": "one"},
            )
            try:
                result = embedder.run("cheese")
            finally:
                embedder.close()

        constructor.assert_called_once_with(
            base_url="https://tei.example.test/api/",
            timeout=4.5,
            headers={"Authorization": "Bearer secret", "X-Tenant": "one"},
        )
        client.post.assert_called_once_with("embed_sparse", json={"inputs": "query: cheese </s>"})
        client.post.return_value.raise_for_status.assert_called_once_with()
        assert result == {"sparse_embedding": SparseEmbedding(indices=[12, 99], values=[1.0, 0.25])}

    @pytest.mark.asyncio
    async def test_run_async_http_request_and_response(self) -> None:
        client = async_http_client([[{"index": 7, "value": 2.5}]])
        with patch(f"{MODULE}.httpx.AsyncClient", return_value=client) as constructor:
            embedder = HuggingFaceAPISparseTextEmbedder(
                api_base_url="http://tei:8080/",
                token=Secret.from_token("token"),
                timeout=9,
                headers={"X-Test": "yes"},
            )
            try:
                result = await embedder.run_async("input")
            finally:
                await embedder.close_async()

        constructor.assert_called_once_with(
            base_url="http://tei:8080/",
            timeout=9,
            headers={"Authorization": "Bearer token", "X-Test": "yes"},
        )
        client.post.assert_awaited_once_with("embed_sparse", json={"inputs": "input"})
        client.post.return_value.raise_for_status.assert_called_once_with()
        assert result == {"sparse_embedding": SparseEmbedding(indices=[7], values=[2.5])}

    def test_run_grpc_request_and_response(self) -> None:
        channel = MagicMock()
        stub = MagicMock()
        stub.EmbedSparse.return_value = SimpleNamespace(
            sparse_embeddings=[SimpleNamespace(index=12, value=1.0), SimpleNamespace(index=99, value=0.25)]
        )
        with (
            patch(f"{MODULE}.grpc.insecure_channel", return_value=channel) as channel_constructor,
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub", return_value=stub) as stub_constructor,
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(
                api_base_url="localhost:8082", prefix="query: ", suffix=" </s>", use_grpc=True
            )
            try:
                result = embedder.run("cheese")
            finally:
                embedder.close()

        channel_constructor.assert_called_once_with("localhost:8082")
        stub_constructor.assert_called_once_with(channel)
        stub.EmbedSparse.assert_called_once_with(tei_pb2.EmbedSparseRequest(inputs="query: cheese </s>"))
        assert result == {"sparse_embedding": SparseEmbedding(indices=[12, 99], values=[1.0, 0.25])}

    @pytest.mark.asyncio
    async def test_run_async_grpc_request_and_response(self) -> None:
        channel = MagicMock(close=AsyncMock())
        stub = MagicMock()
        stub.EmbedSparse = AsyncMock(
            return_value=SimpleNamespace(sparse_embeddings=[SimpleNamespace(index=7, value=2.5)])
        )
        with (
            patch(f"{MODULE}.grpc.aio.insecure_channel", return_value=channel) as channel_constructor,
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub", return_value=stub) as stub_constructor,
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(
                api_base_url="localhost:8082", prefix="query: ", suffix="!", use_grpc=True
            )
            try:
                result = await embedder.run_async("input")
            finally:
                await embedder.close_async()

        channel_constructor.assert_called_once_with("localhost:8082")
        stub_constructor.assert_called_once_with(channel)
        stub.EmbedSparse.assert_awaited_once_with(tei_pb2.EmbedSparseRequest(inputs="query: input!"))
        assert result == {"sparse_embedding": SparseEmbedding(indices=[7], values=[2.5])}

    def test_token_is_refreshed_only_after_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "first-token")
        first_client = sync_http_client()
        second_client = sync_http_client()
        with patch(f"{MODULE}.httpx.Client", side_effect=[first_client, second_client]) as constructor:
            embedder = HuggingFaceAPISparseTextEmbedder()
            try:
                embedder.run("one")
                monkeypatch.setenv("HF_TOKEN", "second-token")
                embedder.run("two")
                assert constructor.call_count == 1

                embedder.close()
                embedder.run("three")
            finally:
                embedder.close()

        assert constructor.call_args_list[0].kwargs["headers"] == {"Authorization": "Bearer first-token"}
        assert constructor.call_args_list[1].kwargs["headers"] == {"Authorization": "Bearer second-token"}

    def test_explicit_authorization_header_wins_over_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "env-token")
        client = sync_http_client()
        with patch(f"{MODULE}.httpx.Client", return_value=client) as constructor:
            embedder = HuggingFaceAPISparseTextEmbedder(headers={"Authorization": "Basic test-key"})
            try:
                embedder.run("text")
            finally:
                embedder.close()

        assert constructor.call_args.kwargs["headers"] == {"Authorization": "Basic test-key"}

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
    def test_run_error_keeps_http_client_open_until_close(self, payload: Any) -> None:
        client = sync_http_client(payload)
        with patch(f"{MODULE}.httpx.Client", return_value=client):
            embedder = HuggingFaceAPISparseTextEmbedder()
            with pytest.raises(ValueError):
                embedder.run("text")

            assert embedder._client is client
            client.close.assert_not_called()
            embedder.close()
            client.close.assert_called_once_with()

    def test_run_propagates_http_error_and_keeps_client_open_until_close(self) -> None:
        request = httpx.Request("POST", "http://localhost:8080/embed_sparse")
        client = sync_http_client()
        client.post.return_value = httpx.Response(503, request=request)
        with patch(f"{MODULE}.httpx.Client", return_value=client):
            embedder = HuggingFaceAPISparseTextEmbedder()
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                embedder.run("text")

            assert exc_info.value.response.status_code == 503
            assert embedder._client is client
            client.close.assert_not_called()
            embedder.close()
            client.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_run_async_error_keeps_http_client_open_until_close(self) -> None:
        request = httpx.Request("POST", "http://localhost:8080/embed_sparse")
        client = async_http_client()
        client.post.return_value = httpx.Response(503, request=request)
        with patch(f"{MODULE}.httpx.AsyncClient", return_value=client):
            embedder = HuggingFaceAPISparseTextEmbedder()
            with pytest.raises(httpx.HTTPStatusError):
                await embedder.run_async("text")

            assert embedder._async_client is client
            client.aclose.assert_not_awaited()
            await embedder.close_async()
            client.aclose.assert_awaited_once_with()

    def test_run_error_keeps_grpc_channel_open_until_close(self) -> None:
        channel = MagicMock()
        stub = MagicMock()
        stub.EmbedSparse.side_effect = RuntimeError("request failed")
        with (
            patch(f"{MODULE}.grpc.insecure_channel", return_value=channel),
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub", return_value=stub),
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)
            with pytest.raises(RuntimeError, match="request failed"):
                embedder.run("text")

            assert embedder._channel is channel
            channel.close.assert_not_called()
            embedder.close()
            channel.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_run_async_error_keeps_grpc_channel_open_until_close(self) -> None:
        channel = MagicMock(close=AsyncMock())
        stub = MagicMock()
        stub.EmbedSparse = AsyncMock(side_effect=RuntimeError("request failed"))
        with (
            patch(f"{MODULE}.grpc.aio.insecure_channel", return_value=channel),
            patch(f"{MODULE}.tei_pb2_grpc.EmbedStub", return_value=stub),
        ):
            embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)
            with pytest.raises(RuntimeError, match="request failed"):
                await embedder.run_async("text")

            assert embedder._async_channel is channel
            channel.close.assert_not_awaited()
            await embedder.close_async()
            channel.close.assert_awaited_once_with()

    @pytest.mark.integration
    def test_live_run_tei_grpc(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)
        try:
            result = embedder.run("sparse retrieval")
        finally:
            embedder.close()

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async_tei_grpc(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="localhost:8082", use_grpc=True)
        try:
            result = await embedder.run_async("sparse retrieval")
        finally:
            await embedder.close_async()

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices

    @pytest.mark.integration
    def test_live_run_tei(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(api_base_url=API_BASE_URL)
        try:
            result = embedder.run("sparse retrieval")
        finally:
            embedder.close()

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async_tei(self) -> None:
        embedder = HuggingFaceAPISparseTextEmbedder(api_base_url=API_BASE_URL)
        try:
            result = await embedder.run_async("sparse retrieval")
        finally:
            await embedder.close_async()

        assert isinstance(result["sparse_embedding"], SparseEmbedding)
        assert result["sparse_embedding"].indices
