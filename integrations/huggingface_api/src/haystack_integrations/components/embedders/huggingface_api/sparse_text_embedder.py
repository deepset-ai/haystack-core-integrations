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

from haystack_integrations.common.huggingface_api.utils import _grpc_metadata, _parse_grpc_sparse_embedding

from .sparse_embedding_utils import _build_client_kwargs, _embed_sparse, _embed_sparse_async

with LazyImport("Run 'pip install \"huggingface-api-haystack[grpc]\"' for grpc support.") as grpc_import:
    from grpc_requests import Client
    from grpc_requests.aio import AsyncClient


@component
class HuggingFaceAPISparseTextEmbedder:
    """
    Embeds text into a sparse vector using a Hugging Face Text Embeddings Inference (TEI) server.

    The TEI server must be running a sparse embedding model and, when using HTTP, expose the `/embed_sparse` endpoint.

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

        :param api_base_url: Base URL of the TEI server, or gRPC target.
        :param token: Token sent to TEI as bearer authorization, if set.
        :param prefix: A string to add before the text.
        :param suffix: A string to add after the text.
        :param timeout: HTTP request timeout in seconds. Set to `None` to disable it.
        :param headers: Additional HTTP headers to send with each request.
        :param use_grpc: Use gRPC instead of HTTP. Requires `huggingface-api-haystack[grpc]`.
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
        self._grpc_client: Client | None = None
        self._async_grpc_client: AsyncClient | None = None

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
        """Create the synchronous client."""
        if self.use_grpc:
            if self._grpc_client is None:
                self._grpc_client = Client(self.api_base_url)
            return

        if self._client is None:
            self._client = httpx.Client(**self._client_kwargs())

    async def warm_up_async(self) -> None:
        """Create the asynchronous client."""
        if self.use_grpc:
            if self._async_grpc_client is None:
                self._async_grpc_client = await AsyncClient.create(self.api_base_url)
            return

        if self._async_client is None:
            self._async_client = httpx.AsyncClient(**self._client_kwargs())

    def close(self) -> None:
        """Close the synchronous client."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._grpc_client is not None:
            self._grpc_client.channel.close()
            self._grpc_client = None

    async def close_async(self) -> None:
        """Close the asynchronous client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
        if self._async_grpc_client is not None:
            await self._async_grpc_client.channel.close()
            self._async_grpc_client = None

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
            assert self._grpc_client is not None  # noqa: S101
            response = self._grpc_client.unary_unary(
                "tei.v1.Embed", "EmbedSparse", {"inputs": text_to_embed}, metadata=_grpc_metadata(self.token)
            )
            return {"sparse_embedding": _parse_grpc_sparse_embedding(response.get("sparse_embeddings", []))}

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
            assert self._async_grpc_client is not None  # noqa: S101
            response = await self._async_grpc_client.unary_unary(
                "tei.v1.Embed", "EmbedSparse", {"inputs": text_to_embed}, metadata=_grpc_metadata(self.token)
            )
            return {"sparse_embedding": _parse_grpc_sparse_embedding(response.get("sparse_embeddings", []))}

        assert self._async_client is not None  # noqa: S101
        embeddings = await _embed_sparse_async(client=self._async_client, inputs=text_to_embed)
        return {"sparse_embedding": embeddings[0]}
