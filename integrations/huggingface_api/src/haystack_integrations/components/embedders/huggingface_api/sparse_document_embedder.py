# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from asyncio import Semaphore, gather
from dataclasses import replace
from itertools import chain
from typing import Any

import httpx
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret
from haystack.utils.url_validation import is_valid_http_url
from tqdm import tqdm

from .sparse_embedding import _embed_sparse, _embed_sparse_async, _request_headers


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
    ) -> None:
        """
        Create a sparse Document embedder backed by TEI.

        :param api_base_url: Base URL of the TEI server.
        :param token: Token sent to TEI as HTTP bearer authorization, if set.
        :param prefix: A string to add before each prepared Document text.
        :param suffix: A string to add after each prepared Document text.
        :param batch_size: Number of Documents sent in each request.
        :param progress_bar: If `True`, show a progress bar while embedding.
        :param meta_fields_to_embed: Metadata fields to embed before the Document content.
        :param embedding_separator: Separator for metadata fields and Document content.
        :param timeout: HTTP request timeout in seconds. Set to `None` to disable it.
        :param headers: Additional HTTP headers to send with each request.
        :param concurrency_limit: Maximum concurrent requests made by `run_async`.
        :raises ValueError: If `api_base_url` is invalid or a numeric parameter is not positive.
        """
        if not is_valid_http_url(api_base_url):
            msg = f"api_base_url must be a valid HTTP URL, but got {api_base_url}"
            raise ValueError(msg)
        if batch_size <= 0:
            msg = f"batch_size must be > 0, but got {batch_size}"
            raise ValueError(msg)
        if concurrency_limit <= 0:
            msg = f"concurrency_limit must be > 0, but got {concurrency_limit}"
            raise ValueError(msg)

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
        base_url = f"{self.api_base_url.rstrip('/')}/"
        client_headers = _request_headers(self.headers, self.token)
        self._client = httpx.Client(base_url=base_url, timeout=self.timeout, headers=client_headers)
        self._async_client = httpx.AsyncClient(base_url=base_url, timeout=self.timeout, headers=client_headers)

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
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HuggingFaceAPISparseDocumentEmbedder":
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)

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

    def _embed_batches(self, texts: list[str]) -> list[SparseEmbedding]:
        embeddings = []
        for start in tqdm(
            range(0, len(texts), self.batch_size),
            disable=not self.progress_bar,
            desc="Calculating sparse embeddings",
        ):
            embeddings.extend(_embed_sparse(client=self._client, inputs=texts[start : start + self.batch_size]))
        return embeddings

    async def _embed_batches_async(self, texts: list[str]) -> list[SparseEmbedding]:
        semaphore = Semaphore(self.concurrency_limit)
        batches = [texts[start : start + self.batch_size] for start in range(0, len(texts), self.batch_size)]
        progress = tqdm(total=len(batches), disable=not self.progress_bar, desc="Calculating sparse embeddings")

        async def embed_batch(batch: list[str]) -> list[SparseEmbedding]:
            async with semaphore:
                result = await _embed_sparse_async(client=self._async_client, inputs=batch)
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
        embeddings = self._embed_batches(self._prepare_texts_to_embed(documents))
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
        embeddings = await self._embed_batches_async(self._prepare_texts_to_embed(documents))
        return {
            "documents": [
                replace(document, sparse_embedding=embedding)
                for document, embedding in zip(documents, embeddings, strict=True)
            ]
        }
