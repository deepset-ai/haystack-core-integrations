# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import SparseEmbedding
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret
from haystack.utils.url_validation import is_valid_http_url

from .sparse_embedding_utils import _build_client_kwargs, _embed_sparse, _embed_sparse_async

with LazyImport("Run 'pip install \"huggingface-api-haystack[grpc]\"' for grpc support.") as grpc_import:
    import grpc

    from haystack_integrations.components.embedders.huggingface_api._grpc import tei_pb2, tei_pb2_grpc


@component
class HuggingFaceAPISparseTextEmbedder:
    """
    Embeds text into a sparse vector using a Hugging Face Text Embeddings Inference (TEI) server.

    The TEI server must be running a sparse embedding model and expose the `/embed_sparse` endpoint.
    HTTP clients and gRPC channels are created lazily by `warm_up()`/`warm_up_async()` or the corresponding run
    method, then reused until `close()`/`close_async()` is called. HTTP tokens are resolved when the client is created.

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPISparseTextEmbedder

    embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="http://localhost:8080")
    result = embedder.run("What is sparse retrieval?")
    print(result["sparse_embedding"])
    ```
    """

    def __init__(
        self,
        *,
        api_base_url: str = "http://localhost:8080",
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        timeout: float | None = 30.0,
        headers: dict[str, str] | None = None,
        use_grpc: bool = False,
    ) -> None:
        """
        Create a sparse text embedder backed by TEI.

        :param api_base_url: Base URL of the TEI server.
        :param token: Token sent to TEI as HTTP bearer authorization, if set. The token is resolved when the HTTP
            client is created during warm-up. Close and warm up the embedder again to pick up a changed token.
        :param prefix: A string to add before the text.
        :param suffix: A string to add after the text.
        :param timeout: HTTP request timeout in seconds. Set to `None` to disable it.
        :param headers: Additional HTTP headers to send with each request.
        :param use_grpc: Use the gRPC API instead of HTTP. This is supported only by TEI and requires installing the
            `grpc` optional dependency. When enabled, `api_base_url` is used as the gRPC target.
        :raises ValueError: If `api_base_url` is not a valid HTTP URL when using HTTP.
        """
        if not use_grpc and not is_valid_http_url(api_base_url):
            msg = f"api_base_url must be a valid HTTP URL, but got {api_base_url}"
            raise ValueError(msg)

        if use_grpc:
            grpc_import.check()

        self.api_base_url = api_base_url
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.timeout = timeout
        self.headers = headers or {}
        self.use_grpc = use_grpc
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._channel: grpc.Channel | None = None
        self._async_channel: grpc.aio.Channel | None = None
        self._stub: tei_pb2_grpc.EmbedStub | None = None
        self._async_stub: tei_pb2_grpc.EmbedAsyncStub | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            api_base_url=self.api_base_url,
            token=self.token,
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            headers=self.headers,
            use_grpc=self.use_grpc,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HuggingFaceAPISparseTextEmbedder":
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)

    def _client_kwargs(self) -> dict[str, Any]:
        return _build_client_kwargs(
            api_base_url=self.api_base_url, timeout=self.timeout, headers=self.headers, token=self.token
        )

    def warm_up(self) -> None:
        """
        Create the synchronous HTTP client or gRPC channel if it has not been created yet.

        The token is resolved when creating an HTTP client. An existing resource is reused.

        :raises ValueError:
            If the configured token cannot be resolved.
        """
        if self.use_grpc:
            if self._channel is None:
                self._channel = grpc.insecure_channel(self.api_base_url)
                self._stub = tei_pb2_grpc.EmbedStub(self._channel)
            return

        if self._client is None:
            self._client = httpx.Client(**self._client_kwargs())

    async def warm_up_async(self) -> None:
        """
        Create the asynchronous HTTP client or gRPC channel if it has not been created yet.

        The token is resolved when creating an HTTP client. An existing resource is reused.

        :raises ValueError:
            If the configured token cannot be resolved.
        """
        if self.use_grpc:
            if self._async_channel is None:
                self._async_channel = grpc.aio.insecure_channel(self.api_base_url)
                self._async_stub = tei_pb2_grpc.EmbedStub(self._async_channel)
            return

        if self._async_client is None:
            self._async_client = httpx.AsyncClient(**self._client_kwargs())

    def close(self) -> None:
        """Close and reset synchronous HTTP and gRPC resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    async def close_async(self) -> None:
        """Close and reset asynchronous HTTP and gRPC resources."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
        if self._async_channel is not None:
            await self._async_channel.close()
            self._async_channel = None
            self._async_stub = None

    def _prepare_input(self, text: str) -> str:
        if not isinstance(text, str):
            msg = (
                "HuggingFaceAPISparseTextEmbedder expects a string as input. "
                "To embed Documents, use HuggingFaceAPISparseDocumentEmbedder."
            )
            raise TypeError(msg)
        return self.prefix + text + self.suffix

    @component.output_types(sparse_embedding=SparseEmbedding)
    def run(self, text: str) -> dict[str, SparseEmbedding]:
        """
        Embed a single string.

        :param text: Text to embed.
        :returns: The sparse embedding of the input text.
        """
        self.warm_up()
        text_to_embed = self._prepare_input(text)
        if self.use_grpc:
            assert self._stub is not None  # noqa: S101
            response = self._stub.EmbedSparse(tei_pb2.EmbedSparseRequest(inputs=text_to_embed))
            return {
                "sparse_embedding": SparseEmbedding(
                    indices=[item.index for item in response.sparse_embeddings],
                    values=[item.value for item in response.sparse_embeddings],
                )
            }

        assert self._client is not None  # noqa: S101
        embeddings = _embed_sparse(client=self._client, inputs=text_to_embed)
        return {"sparse_embedding": embeddings[0]}

    @component.output_types(sparse_embedding=SparseEmbedding)
    async def run_async(self, text: str) -> dict[str, SparseEmbedding]:
        """
        Embed a single string asynchronously.

        :param text: Text to embed.
        :returns: The sparse embedding of the input text.
        """
        await self.warm_up_async()
        text_to_embed = self._prepare_input(text)
        if self.use_grpc:
            assert self._async_stub is not None  # noqa: S101
            response = await self._async_stub.EmbedSparse(tei_pb2.EmbedSparseRequest(inputs=text_to_embed))
            return {
                "sparse_embedding": SparseEmbedding(
                    indices=[item.index for item in response.sparse_embeddings],
                    values=[item.value for item in response.sparse_embeddings],
                )
            }

        assert self._async_client is not None  # noqa: S101
        embeddings = await _embed_sparse_async(client=self._async_client, inputs=text_to_embed)
        return {"sparse_embedding": embeddings[0]}
