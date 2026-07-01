# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from copy import copy
from dataclasses import replace
from typing import Any, Literal

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from dakera import AsyncDakeraClient, DakeraClient, DistanceMetric, NotFoundError, Vector

from .filters import _normalize_filters, _validate_filters

logger = logging.getLogger(__name__)

# Dakera returns at most this many vectors from a single similarity query.
# When plain filtering (querying with a dummy vector) hits this limit, more
# matching documents may exist in the namespace.
TOP_K_LIMIT = 1_000

# Only these metadata value types can be stored and filtered on server-side.
# ``list[str]`` is supported and checked separately.
METADATA_SUPPORTED_TYPES = str, int, bool, float

# Distance metric aliases accepted in the constructor, mapped to the SDK enum.
_METRIC_MAP: dict[str, DistanceMetric] = {
    "cosine": DistanceMetric.COSINE,
    "euclidean": DistanceMetric.EUCLIDEAN,
    "dot_product": DistanceMetric.DOT_PRODUCT,
}


class DakeraDocumentStore:
    """
    A Document Store backed by a self-hosted [Dakera](https://dakera.ai) memory server.

    Dakera stores documents as vectors inside a *namespace* and serves dense similarity
    search over them. This document store drives Dakera's raw vector-namespace API, so the
    embeddings are supplied by any Haystack embedder while Dakera handles storage and recall.

    Usage example:
    ```python
    import os
    from haystack import Document
    from haystack_integrations.document_stores.dakera import DakeraDocumentStore

    os.environ["DAKERA_API_KEY"] = "dk-..."
    document_store = DakeraDocumentStore(
        url="http://localhost:3000",
        namespace="my-docs",
        dimension=768,
    )
    document_store.write_documents([Document(content="Hello world", embedding=[0.1] * 768)])
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("DAKERA_API_KEY"),  # noqa: B008
        url: str = "http://localhost:3000",
        namespace: str = "default",
        dimension: int = 768,
        metric: Literal["cosine", "euclidean", "dot_product"] = "cosine",
        batch_size: int = 100,
    ) -> None:
        """
        Creates a new DakeraDocumentStore instance.

        :param api_key: The Dakera API key (a ``dk-...`` token). Read from the ``DAKERA_API_KEY``
            environment variable by default.
        :param url: Base URL of the Dakera server. Defaults to ``http://localhost:3000``, the port
            used by the `dakera-deploy` docker-compose stack.
        :param namespace: The Dakera namespace to read from and write to. It is created on the first
            write if it does not exist.
        :param dimension: The dimension of the embeddings. Only used when the namespace is created.
        :param metric: The distance metric used for similarity search. Only used when the namespace
            is created.
        :param batch_size: The number of documents to send to Dakera in a single upsert request.

        :raises ValueError: If ``metric`` is not one of ``cosine``, ``euclidean`` or ``dot_product``.
        """
        if metric not in _METRIC_MAP:
            msg = f"Unsupported metric '{metric}'. Choose one of {list(_METRIC_MAP)}."
            raise ValueError(msg)

        self.api_key = api_key
        self.url = url
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.batch_size = batch_size

        self._client: DakeraClient | None = None
        self._async_client: AsyncDakeraClient | None = None
        # Dummy embedding used to run metadata-only filter queries and as a placeholder
        # for documents written without an embedding.
        self._dummy_vector = [-10.0] * self.dimension

    def _initialize_client(self) -> DakeraClient:
        if self._client is not None:
            return self._client

        client = DakeraClient(base_url=self.url, api_key=self.api_key.resolve_value())
        self._ensure_namespace(client.get_namespace, client.configure_namespace)
        self._client = client
        return client

    async def _initialize_async_client(self) -> AsyncDakeraClient:
        if self._async_client is not None:
            return self._async_client

        client = AsyncDakeraClient(base_url=self.url, api_key=self.api_key.resolve_value())
        try:
            info = await client.get_namespace(self.namespace)
            self._adopt_existing_dimension(info.dimensions)
        except NotFoundError:
            await client.configure_namespace(
                self.namespace, dimension=self.dimension, distance=_METRIC_MAP[self.metric]
            )
        self._async_client = client
        return client

    def _ensure_namespace(self, get_namespace: Any, configure_namespace: Any) -> None:
        """Adopt the dimension of an existing namespace, or create it if missing."""
        try:
            info = get_namespace(self.namespace)
            self._adopt_existing_dimension(info.dimensions)
        except NotFoundError:
            configure_namespace(self.namespace, dimension=self.dimension, distance=_METRIC_MAP[self.metric])

    def _adopt_existing_dimension(self, actual_dimension: int | None) -> None:
        if actual_dimension and actual_dimension != self.dimension:
            logger.warning(
                "Namespace '{namespace}' has dimension {actual}, but {specified} was specified. "
                "The existing dimension will be used.",
                namespace=self.namespace,
                actual=actual_dimension,
                specified=self.dimension,
            )
            self.dimension = actual_dimension
            self._dummy_vector = [-10.0] * self.dimension

    def close(self) -> None:
        """Releases the synchronous client resources."""
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None

    async def close_async(self) -> None:
        """Releases the asynchronous client resources. Invoke it manually when done."""
        if self._async_client is not None:
            close = getattr(self._async_client, "close", None)
            if callable(close):
                await close()
            self._async_client = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DakeraDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            url=self.url,
            namespace=self.namespace,
            dimension=self.dimension,
            metric=self.metric,
            batch_size=self.batch_size,
        )

    def count_documents(self) -> int:
        """Returns how many documents are present in the namespace."""
        client = self._initialize_client()
        try:
            return client.get_namespace(self.namespace).vector_count
        except NotFoundError:
            return 0

    async def count_documents_async(self) -> int:
        """Asynchronously returns how many documents are present in the namespace."""
        client = await self._initialize_async_client()
        try:
            info = await client.get_namespace(self.namespace)
            return info.vector_count
        except NotFoundError:
            return 0

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes Documents to Dakera.

        :param documents: A list of Documents to write.
        :param policy: The duplicate policy. Dakera upserts by id, so only
            `DuplicatePolicy.OVERWRITE` semantics are supported.
        :returns: The number of documents written.
        """
        client = self._initialize_client()
        vectors = self._prepare_documents_for_writing(documents, policy)
        for batch in self._batched(vectors, self.batch_size):
            client.upsert(self.namespace, vectors=batch)
        return len(vectors)

    async def write_documents_async(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Asynchronously writes Documents to Dakera.

        :param documents: A list of Documents to write.
        :param policy: The duplicate policy. Dakera upserts by id, so only
            `DuplicatePolicy.OVERWRITE` semantics are supported.
        :returns: The number of documents written.
        """
        client = await self._initialize_async_client()
        vectors = self._prepare_documents_for_writing(documents, policy)
        for batch in self._batched(vectors, self.batch_size):
            await client.upsert(self.namespace, vectors=batch)
        return len(vectors)

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For the filter specification, refer to the
        [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

        :param filters: The filters to apply.
        :returns: A list of Documents that match the given filters.
        """
        _validate_filters(filters)
        documents = self._embedding_retrieval(query_embedding=self._dummy_vector, filters=filters, top_k=TOP_K_LIMIT)
        # A dummy query vector produces meaningless scores, so drop them.
        documents = [replace(doc, score=None) for doc in documents]
        self._warn_if_truncated(len(documents))
        return documents

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        :param filters: The filters to apply.
        :returns: A list of Documents that match the given filters.
        """
        _validate_filters(filters)
        documents = await self._embedding_retrieval_async(
            query_embedding=self._dummy_vector, filters=filters, top_k=TOP_K_LIMIT
        )
        documents = [replace(doc, score=None) for doc in documents]
        self._warn_if_truncated(len(documents))
        return documents

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents with the provided ids from the namespace.

        :param document_ids: The ids to delete.
        """
        if not document_ids:
            return
        client = self._initialize_client()
        try:
            client.delete(self.namespace, ids=document_ids)
        except NotFoundError:
            logger.debug("Namespace '{namespace}' not found. Nothing to delete.", namespace=self.namespace)

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Asynchronously deletes documents with the provided ids from the namespace.

        :param document_ids: The ids to delete.
        """
        if not document_ids:
            return
        client = await self._initialize_async_client()
        try:
            await client.delete(self.namespace, ids=document_ids)
        except NotFoundError:
            logger.debug("Namespace '{namespace}' not found. Nothing to delete.", namespace=self.namespace)

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Retrieves the documents most similar to ``query_embedding``.

        This method is not part of the public interface of `DakeraDocumentStore`; it is called by
        `DakeraEmbeddingRetriever`, which is the public entry point.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :returns: A list of Documents most similar to ``query_embedding``.
        :raises ValueError: If ``query_embedding`` is empty.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        _validate_filters(filters)
        normalized = _normalize_filters(filters) if filters else None
        client = self._initialize_client()
        result = client.query(
            self.namespace,
            vector=query_embedding,
            top_k=top_k,
            filter=normalized,
            include_values=True,
            include_metadata=True,
            distance_metric=_METRIC_MAP[self.metric],
        )
        return self._convert_query_result_to_documents(result)

    async def _embedding_retrieval_async(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Asynchronously retrieves the documents most similar to ``query_embedding``.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :returns: A list of Documents most similar to ``query_embedding``.
        :raises ValueError: If ``query_embedding`` is empty.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        _validate_filters(filters)
        normalized = _normalize_filters(filters) if filters else None
        client = await self._initialize_async_client()
        result = await client.query(
            self.namespace,
            vector=query_embedding,
            top_k=top_k,
            filter=normalized,
            include_values=True,
            include_metadata=True,
            distance_metric=_METRIC_MAP[self.metric],
        )
        return self._convert_query_result_to_documents(result)

    def _convert_query_result_to_documents(self, result: Any) -> list[Document]:
        documents = []
        for match in result.results:
            metadata = dict(match.metadata or {})
            content = metadata.pop("content", None)

            # We always store a vector when writing, but we don't want to surface the dummy
            # placeholder used for documents that had no embedding.
            embedding = match.values if match.values and match.values != self._dummy_vector else None

            documents.append(
                Document(
                    id=match.id,
                    content=content,
                    meta=metadata,
                    embedding=embedding,
                    score=match.score,
                )
            )
        return documents

    @staticmethod
    def _warn_if_truncated(count: int) -> None:
        if count == TOP_K_LIMIT:
            logger.warning(
                "DakeraDocumentStore returned the maximum of {limit} documents for this query. "
                "More matching documents may exist in the namespace.",
                limit=TOP_K_LIMIT,
            )

    @staticmethod
    def _batched(items: list[Any], size: int) -> list[list[Any]]:
        if size <= 0:
            return [items] if items else []
        return [items[i : i + size] for i in range(0, len(items), size)]

    def _discard_invalid_meta(self, document: Document) -> Document:
        """Removes metadata fields whose types Dakera cannot store or filter on."""

        def valid_type(value: Any) -> bool:
            return isinstance(value, METADATA_SUPPORTED_TYPES) or (
                isinstance(value, list) and all(isinstance(i, str) for i in value)
            )

        if not document.meta:
            return document

        discarded_keys = []
        new_meta = {}
        for key, value in document.meta.items():
            if valid_type(value):
                new_meta[key] = value
            else:
                discarded_keys.append(key)

        if discarded_keys:
            logger.warning(
                "Document {id} has metadata fields with unsupported types: {keys}. "
                "Only str, int, bool, float and list[str] are supported; these fields are discarded.",
                id=document.id,
                keys=discarded_keys,
            )
        return replace(document, meta=new_meta)

    def _convert_documents_to_vectors(self, documents: list[Document]) -> list[Vector]:
        vectors = []
        for document in documents:
            embedding = copy(document.embedding)
            if embedding is None:
                logger.warning(
                    "Document {id} has no embedding. Dakera is a vector store, so a dummy embedding is used. "
                    "This can affect search results.",
                    id=document.id,
                )
                embedding = self._dummy_vector

            metadata = dict(self._discard_invalid_meta(document).meta or {})
            if document.content is not None:
                metadata["content"] = document.content
            if document.blob is not None:
                logger.warning(
                    "Document {id} has a `blob` field, but storing ByteStream objects in Dakera is not "
                    "supported. The `blob` field is ignored.",
                    id=document.id,
                )

            vectors.append(Vector(id=document.id, values=embedding, metadata=metadata))
        return vectors

    def _prepare_documents_for_writing(self, documents: list[Document], policy: DuplicatePolicy) -> list[Vector]:
        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy not in (DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE):
            logger.warning(
                "DakeraDocumentStore only supports `DuplicatePolicy.OVERWRITE` but got {policy}. "
                "Duplicates are overwritten by id.",
                policy=str(policy),
            )
        return self._convert_documents_to_vectors(documents)
