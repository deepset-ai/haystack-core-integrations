# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from asyncio import Semaphore, gather
from collections.abc import AsyncIterator
from dataclasses import replace
from itertools import chain
from typing import Any

import httpx
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import SparseEmbedding
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret
from haystack.utils.url_validation import is_valid_http_url
from tqdm import tqdm

from .sparse_embedding_utils import _build_client_kwargs, _embed_sparse, _embed_sparse_async

with LazyImport("Run 'pip install \"huggingface-api-haystack[grpc]\"' for grpc support.") as grpc_import:
    import grpc

    from haystack_integrations.components.embedders.huggingface_api._grpc import tei_pb2, tei_pb2_grpc


@component
class HuggingFaceAPISparseDocumentEmbedder:
    """
    Embeds Documents into sparse vectors using a Hugging Face Text Embeddings Inference (TEI) server.

    The component batches requests and returns copies of the input Documents with `sparse_embedding` set.

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPISparseDocumentEmbedder

    embedder = HuggingFaceAPISparseDocumentEmbedder(api_base_url="http://localhost:8080")
    documents = embedder.run([Document(content="Sparse retrieval")])["documents"]
    print(documents[0].sparse_embedding)
    ```
    """

    def __init__(
        self,
        *,
        api_base_url: str = "http://localhost:8080",
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        timeout: float | None = 30.0,
        headers: dict[str, str] | None = None,
        concurrency_limit: int = 4,
        use_grpc: bool = False,
    ) -> None:
        """
        Create a sparse Document embedder backed by TEI.

        :param api_base_url: Base URL of the TEI server.
        :param token: Token sent to TEI as HTTP bearer authorization, if set.
        :param prefix: A string to add before each prepared Document text.
        :param suffix: A string to add after each prepared Document text.
        :param batch_size: Number of Documents sent in each HTTP request. This parameter is ignored when using gRPC.
        :param progress_bar: If `True`, show a progress bar while embedding.
        :param meta_fields_to_embed: Metadata fields to embed before the Document content.
        :param embedding_separator: Separator for metadata fields and Document content.
        :param timeout: HTTP request timeout in seconds. Set to `None` to disable it.
        :param headers: Additional HTTP headers to send with each request.
        :param concurrency_limit: Maximum concurrent requests made by `run_async`.
        :param use_grpc: Use the gRPC API instead of HTTP. This is supported only by TEI and requires installing the
            `grpc` optional dependency. When enabled, `api_base_url` is used as the gRPC target.
        :raises ValueError: If `api_base_url` is invalid when using HTTP or a numeric parameter is not positive.
        """
        if not use_grpc and not is_valid_http_url(api_base_url):
            msg = f"api_base_url must be a valid HTTP URL, but got {api_base_url}"
            raise ValueError(msg)
        if batch_size <= 0:
            msg = f"batch_size must be > 0, but got {batch_size}"
            raise ValueError(msg)
        if concurrency_limit <= 0:
            msg = f"concurrency_limit must be > 0, but got {concurrency_limit}"
            raise ValueError(msg)

        if use_grpc:
            grpc_import.check()
            self._channel = grpc.insecure_channel(api_base_url)
            self._stub = tei_pb2_grpc.EmbedStub(self._channel)
            self._async_channel = grpc.aio.insecure_channel(api_base_url)
            self._async_stub = tei_pb2_grpc.EmbedStub(self._async_channel)

        self.api_base_url = api_base_url
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.timeout = timeout
        self.headers = headers or {}
        self.concurrency_limit = concurrency_limit
        self.use_grpc = use_grpc

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            api_base_url=self.api_base_url,
            token=self.token,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            timeout=self.timeout,
            headers=self.headers,
            concurrency_limit=self.concurrency_limit,
            use_grpc=self.use_grpc,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HuggingFaceAPISparseDocumentEmbedder":
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)

    def _client_kwargs(self) -> dict[str, Any]:
        return _build_client_kwargs(
            api_base_url=self.api_base_url, timeout=self.timeout, headers=self.headers, token=self.token
        )

    def _prepare_texts_to_embed(self, documents: list[Document]) -> list[str]:
        texts = []
        for document in documents:
            meta = [
                str(document.meta[field])
                for field in self.meta_fields_to_embed
                if field in document.meta and document.meta[field] is not None
            ]
            texts.append(self.prefix + self.embedding_separator.join([*meta, document.content or ""]) + self.suffix)
        return texts

    def _embed_batch_grpc(self, texts: list[str]) -> list[SparseEmbedding]:
        """Embed all texts through a single gRPC stream."""
        if not texts:
            return []

        responses = self._stub.EmbedSparseStream(tei_pb2.EmbedSparseRequest(inputs=text) for text in texts)
        embeddings = [
            SparseEmbedding(
                indices=[sparse_value.index for sparse_value in response.sparse_embeddings],
                values=[sparse_value.value for sparse_value in response.sparse_embeddings],
            )
            for response in tqdm(
                responses,
                total=len(texts),
                disable=not self.progress_bar,
                desc="Calculating sparse embeddings",
            )
        ]
        if len(embeddings) != len(texts):
            msg = f"Expected {len(texts)} sparse embeddings, got {len(embeddings)}"
            raise ValueError(msg)
        return embeddings

    async def _embed_batch_grpc_async(self, texts: list[str]) -> list[SparseEmbedding]:
        """Embed texts through concurrent gRPC streams."""
        if not texts:
            return []

        stream_count = min(max(1, self.concurrency_limit), len(texts))
        streams = [
            texts[len(texts) * index // stream_count : len(texts) * (index + 1) // stream_count]
            for index in range(stream_count)
        ]
        progress = tqdm(total=stream_count, disable=not self.progress_bar, desc="Calculating sparse embeddings")

        async def embed_stream(stream_texts: list[str]) -> list[SparseEmbedding]:
            async def requests() -> AsyncIterator[tei_pb2.EmbedSparseRequest]:
                for text in stream_texts:
                    yield tei_pb2.EmbedSparseRequest(inputs=text)

            responses = self._async_stub.EmbedSparseStream(requests())
            embeddings = [
                SparseEmbedding(
                    indices=[sparse_value.index for sparse_value in response.sparse_embeddings],
                    values=[sparse_value.value for sparse_value in response.sparse_embeddings],
                )
                async for response in responses
            ]
            if len(embeddings) != len(stream_texts):
                msg = f"Expected {len(stream_texts)} sparse embeddings, got {len(embeddings)}"
                raise ValueError(msg)
            progress.update(1)
            return embeddings

        try:
            return list(chain.from_iterable(await gather(*(embed_stream(stream) for stream in streams))))
        finally:
            progress.close()

    def _embed_batches(self, client: httpx.Client, texts: list[str]) -> list[SparseEmbedding]:
        embeddings = []
        for start in tqdm(
            range(0, len(texts), self.batch_size),
            disable=not self.progress_bar,
            desc="Calculating sparse embeddings",
        ):
            embeddings.extend(_embed_sparse(client=client, inputs=texts[start : start + self.batch_size]))
        return embeddings

    async def _embed_batches_async(self, client: httpx.AsyncClient, texts: list[str]) -> list[SparseEmbedding]:
        semaphore = Semaphore(self.concurrency_limit)
        batches = [texts[start : start + self.batch_size] for start in range(0, len(texts), self.batch_size)]
        progress = tqdm(total=len(batches), disable=not self.progress_bar, desc="Calculating sparse embeddings")

        async def embed_batch(batch: list[str]) -> list[SparseEmbedding]:
            async with semaphore:
                result = await _embed_sparse_async(client=client, inputs=batch)
                progress.update(1)
                return result

        try:
            return list(chain.from_iterable(await gather(*(embed_batch(batch) for batch in batches))))
        finally:
            progress.close()

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        if not isinstance(documents, list) or any(not isinstance(document, Document) for document in documents):
            msg = (
                "HuggingFaceAPISparseDocumentEmbedder expects a list of Documents as input. "
                "To embed a string, use HuggingFaceAPISparseTextEmbedder."
            )
            raise TypeError(msg)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Embed a list of Documents.

        :param documents: Documents to embed.
        :returns: Copies of the Documents with sparse embeddings.
        """
        self._validate_documents(documents)
        texts = self._prepare_texts_to_embed(documents)
        if self.use_grpc:
            embeddings = self._embed_batch_grpc(texts)
        else:
            with httpx.Client(**self._client_kwargs()) as client:
                embeddings = self._embed_batches(client, texts)
        return {
            "documents": [
                replace(document, sparse_embedding=embedding)
                for document, embedding in zip(documents, embeddings, strict=True)
            ]
        }

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Embed a list of Documents asynchronously.

        :param documents: Documents to embed.
        :returns: Copies of the Documents with sparse embeddings.
        """
        self._validate_documents(documents)
        texts = self._prepare_texts_to_embed(documents)
        if self.use_grpc:
            embeddings = await self._embed_batch_grpc_async(texts)
        else:
            async with httpx.AsyncClient(**self._client_kwargs()) as client:
                embeddings = await self._embed_batches_async(client, texts)
        return {
            "documents": [
                replace(document, sparse_embedding=embedding)
                for document, embedding in zip(documents, embeddings, strict=True)
            ]
        }
