from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, ClassVar, Literal

from glide import (
    BackoffStrategy,
    GlideClient,
    GlideClientConfiguration,
    GlideClusterClient,
    GlideClusterClientConfiguration,
    ft,
    glide_json,
)
from glide_shared import (
    NodeAddress,
    ServerCredentials,
)
from glide_shared.commands.server_modules.ft_options.ft_create_options import (
    DataType,
    DistanceMetricType,
    Field,
    FtCreateOptions,
    NumericField,
    TagField,
    VectorAlgorithm,
    VectorField,
    VectorFieldAttributesHnsw,
    VectorType,
)
from glide_shared.commands.server_modules.ft_options.ft_search_options import (
    FtSearchOptions,
    ReturnField,
)
from glide_sync import (
    BackoffStrategy as SyncBackoffStrategy,
)
from glide_sync import (
    GlideClient as SyncGlideClient,
)
from glide_sync import (
    GlideClientConfiguration as SyncGlideClientConfiguration,
)
from glide_sync import (
    GlideClusterClient as SyncGlideClusterClient,
)
from glide_sync import (
    GlideClusterClientConfiguration as SyncGlideClusterClientConfiguration,
)
from glide_sync import ft as sync_ft
from glide_sync import (
    glide_json as sync_glide_json,
)
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from haystack_integrations.document_stores.valkey.filters import _normalize_filters, _validate_filters

logger = logging.getLogger(__name__)


class ValkeyDocumentStoreError(DocumentStoreError):
    pass


class ValkeyDocumentStore(DocumentStore):
    """
    A document store implementation using Valkey (Redis-compatible) with vector search capabilities.

    This document store provides persistent storage for documents with embeddings and supports
    vector similarity search using the FT.SEARCH module. It's designed for high-performance
    retrieval applications requiring both semantic search and metadata filtering.

    Key features:
    - Vector similarity search with HNSW algorithm
    - Metadata filtering on tag and numeric fields
    - Configurable distance metrics (L2, cosine, inner product)
    - Batch operations for efficient document management
    - Both synchronous and asynchronous operations
    - Cluster and standalone mode support

    Supported filterable metadata fields:
    - meta_category (TagField): exact string matches
    - meta_status (TagField): status filtering
    - meta_priority (NumericField): numeric comparisons
    - meta_score (NumericField): score filtering
    - meta_timestamp (NumericField): date/time filtering

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.document_stores.valkey import ValkeyDocumentStore

    # Initialize document store
    document_store = ValkeyDocumentStore(
        nodes_list=[("localhost", 6379)],
        index_name="my_documents",
        embedding_dim=768,
        distance_metric="cosine"
    )

    # Store documents with embeddings
    documents = [
        Document(
            content="Valkey is a Redis-compatible database",
            embedding=[0.1, 0.2, ...],  # 768-dim vector
            meta={"category": "database", "priority": 1}
        )
    ]
    document_store.write_documents(documents)

    # Search with filters
    results = document_store.search(
        embedding=[0.1, 0.15, ...],
        filters={"field": "meta.category", "operator": "==", "value": "database"},
        limit=10
    )
    ```
    """

    _METRIC_MAP: ClassVar[dict[str, DistanceMetricType]] = {
        "l2": DistanceMetricType.L2,
        "cosine": DistanceMetricType.COSINE,
        "ip": DistanceMetricType.IP,
    }
    _DUMMY_VALUE = -10.0

    def __init__(
        self,
        # Connection configuration
        nodes_list: list[tuple[str, int]] | None = None,
        *,
        cluster_mode: bool = False,
        # Security and authentication configuration
        use_tls: bool = False,
        username: str | None = None,
        password: str | None = None,
        # Client timeout and retry configuration
        request_timeout: int = 500,
        retry_attempts: int = 3,
        retry_base_delay_ms: int = 1000,
        retry_exponent_base: int = 2,
        # Document store operation configuration
        batch_size: int = 100,
        # Index and vector configuration
        index_name: str = "haystack_document",
        distance_metric: Literal["l2", "cosine", "ip"] = "cosine",
        embedding_dim: int = 768,
    ):
        self._index_name = index_name
        self._distance_metric = self._parse_metric(distance_metric)
        self._embedding_dim = embedding_dim
        self._dummy_vector = [ValkeyDocumentStore._DUMMY_VALUE] * self._embedding_dim

        self._nodes_list: list[tuple[str, int]] = nodes_list or [("localhost", 6379)]
        self._cluster_mode: bool = cluster_mode
        self._use_tls: bool = use_tls
        self._username: str | None = username
        self._password: str | None = password
        self._request_timeout: int = request_timeout
        self._retry_attempts: int = retry_attempts
        self._retry_base_delay_ms: int = retry_base_delay_ms
        self._retry_exponent_base: int = retry_exponent_base
        self._batch_size: int = batch_size
        self._client: SyncGlideClient | SyncGlideClusterClient | None = None
        self._async_client: GlideClient | GlideClusterClient | None = None

    def _get_connection(self) -> SyncGlideClient | SyncGlideClusterClient:
        if self._client:
            return self._client

        self._verify_node_list(self._nodes_list)

        addresses = [NodeAddress(host, port) for host, port in self._nodes_list]
        reconnect_strategy = SyncBackoffStrategy(
            num_of_retries=self._retry_attempts,
            factor=self._retry_base_delay_ms,
            exponent_base=self._retry_exponent_base,
        )

        try:
            if self._cluster_mode:
                cluster_config = SyncGlideClusterClientConfiguration(
                    addresses=addresses,
                    use_tls=self._use_tls,
                    credentials=self._build_credentials(self._username, self._password),
                    request_timeout=self._request_timeout,
                    reconnect_strategy=reconnect_strategy,
                )
                self._client = SyncGlideClusterClient.create(cluster_config)
            else:
                client_config = SyncGlideClientConfiguration(
                    addresses=addresses,
                    use_tls=self._use_tls,
                    credentials=self._build_credentials(self._username, self._password),
                    request_timeout=self._request_timeout,
                    reconnect_strategy=reconnect_strategy,
                )
                self._client = SyncGlideClient.create(client_config)
            return self._client
        except Exception as e:
            msg = "Failed to connect to Valkey"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    async def _get_async_connection(self) -> GlideClient | GlideClusterClient:
        if self._async_client:
            return self._async_client

        self._verify_node_list(self._nodes_list)

        addresses = [NodeAddress(host, port) for host, port in self._nodes_list]
        reconnect_strategy = BackoffStrategy(
            num_of_retries=self._retry_attempts,
            factor=self._retry_base_delay_ms,
            exponent_base=self._retry_exponent_base,
        )

        try:
            if self._cluster_mode:
                cluster_config = GlideClusterClientConfiguration(
                    addresses=addresses,
                    use_tls=self._use_tls,
                    credentials=self._build_credentials(self._username, self._password),
                    request_timeout=self._request_timeout,
                    reconnect_strategy=reconnect_strategy,
                )
                self._async_client = await GlideClusterClient.create(cluster_config)
            else:
                client_config = GlideClientConfiguration(
                    addresses=addresses,
                    use_tls=self._use_tls,
                    credentials=self._build_credentials(self._username, self._password),
                    request_timeout=self._request_timeout,
                    reconnect_strategy=reconnect_strategy,
                )
                self._async_client = await GlideClient.create(client_config)
            return self._async_client
        except Exception as e:
            msg = "Failed to connect to Valkey"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.error(f"Failed to close Valkey client: {e}")
                pass
        self._client = None

    async def async_close(self) -> None:
        if self._async_client:
            try:
                await self._async_client.close()
            except Exception as e:
                logger.error(f"Failed to close Valkey client: {e}")
                pass
        self._async_client = None

    def _has_index(self) -> bool:
        client = self._get_connection()

        try:
            sync_ft.info(client, self._index_name)
            return True
        except Exception as e:
            logger.info(f"Valkey index check failed for {self._index_name}, {e}")
            return False

    async def _async_has_index(self) -> bool:
        client = await self._get_async_connection()

        try:
            await ft.info(client, self._index_name)
            return True
        except Exception as e:
            logger.info(f"Valkey index check failed for {self._index_name}, {e}")
            return False

    def _prepare_index_fields(self) -> list[Field]:
        """Prepare index fields configuration."""
        return [
            TagField("$.id", alias="id"),
            VectorField(
                name="vector",
                algorithm=VectorAlgorithm.HNSW,
                attributes=VectorFieldAttributesHnsw(
                    dimensions=self._embedding_dim,
                    distance_metric=self._distance_metric,
                    type=VectorType.FLOAT32,
                ),
            ),
            TagField("$.meta_category", alias="meta_category"),
            TagField("$.meta_status", alias="meta_status"),
            NumericField("$.meta_priority", alias="meta_priority"),
            NumericField("$.meta_score", alias="meta_score"),
            NumericField("$.meta_timestamp", alias="meta_timestamp"),
        ]

    def _create_index(self) -> None:
        client = self._get_connection()

        try:
            if self._has_index():
                logger.info(f"Index {self._index_name} already exists")
                return

            fields = self._prepare_index_fields()
            options = FtCreateOptions(DataType.JSON)
            ok = sync_ft.create(client, self._index_name, fields, options)
            if ok not in (b"OK", "OK"):
                msg = f"FT.CREATE failed for index '{self._index_name}': {ok!r}"
                raise ValkeyDocumentStoreError(msg)

        except Exception as e:
            msg = f"Error creating collection {self._index_name}"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    async def _async_create_index(self) -> None:
        try:
            client = await self._get_async_connection()

            if await self._async_has_index():
                logger.info(f"Index {self._index_name} already exists")
                return

            fields = self._prepare_index_fields()
            options = FtCreateOptions(DataType.JSON)
            ok = await ft.create(client, self._index_name, fields, options)
            if ok not in (b"OK", "OK"):
                msg = f"FT.CREATE failed for index '{self._index_name}': {ok!r}"
                raise ValkeyDocumentStoreError(msg) from None

        except Exception as e:
            msg = f"Error creating collection {self._index_name}"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        return default_to_dict(
            self,
            nodes_list=self._nodes_list,
            cluster_mode=self._cluster_mode,
            use_tls=self._use_tls,
            username=self._username,
            password=self._password,
            request_timeout=self._request_timeout,
            retry_attempts=self._retry_attempts,
            retry_base_delay_ms=self._retry_base_delay_ms,
            retry_exponent_base=self._retry_exponent_base,
            batch_size=self._batch_size,
            index_name=self._index_name,
            distance_metric=self._distance_metric.name.lower(),
            embedding_dim=self._embedding_dim,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValkeyDocumentStore:
        """
        Deserializes the store from a dictionary.
        """
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Return the number of documents stored in the document store.

        This method queries the Valkey Search index to get the total count of indexed documents.
        If the index doesn't exist, it returns 0.

        :return: The number of documents in the document store.
        :raises ValkeyDocumentStoreError: If there's an error accessing the index or counting documents.

        Example:
        ```python
        document_store = ValkeyDocumentStore()
        count = document_store.count_documents()
        print(f"Total documents: {count}")
        ```
        """
        client = self._get_connection()

        if not self._has_index():
            logger.info(f"Index {self._index_name} does not exist")
            return 0

        try:
            info = sync_ft.info(client, self._index_name)
            num_docs = info[b"num_docs"]
            if isinstance(num_docs, (str, bytes, int)):
                return int(num_docs)
            logger.warning(f"Unexpected type for num_docs: {type(num_docs)}")
            return 0
        except Exception as e:
            msg = f"Error counting documents in index '{self._index_name}'"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    async def async_count_documents(self) -> int:
        """
        Asynchronously return the number of documents stored in the document store.

        This method queries the Valkey Search index to get the total count of indexed documents.
        If the index doesn't exist, it returns 0. This is the async version of count_documents().

        :return: The number of documents in the document store.
        :raises ValkeyDocumentStoreError: If there's an error accessing the index or counting documents.

        Example:
        ```python
        document_store = ValkeyDocumentStore()
        count = await document_store.async_count_documents()
        print(f"Total documents: {count}")
        ```
        """
        client = await self._get_async_connection()

        if not await self._async_has_index():
            logger.info(f"Index {self._index_name} does not exist")
            return 0

        try:
            info = await ft.info(client, self._index_name)
            num_docs = info[b"num_docs"]
            if isinstance(num_docs, (str, bytes, int)):
                return int(num_docs)
            logger.warning(f"Unexpected type for num_docs: {type(num_docs)}")
            return 0
        except Exception as e:
            logger.error(f"Error counting documents in index '{self._index_name}': {e}")
            raise ValkeyDocumentStoreError(e) from e

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Filter documents by metadata without vector search.

        This method retrieves documents based on metadata filters without performing vector similarity search.
        Since Valkey Search requires vector queries, this method uses a dummy vector internally and removes
        the similarity scores from results.

        :param filters: Optional metadata filters in Haystack format. Supports filtering on:
            - meta.category (string equality)
            - meta.status (string equality)
            - meta.priority (numeric comparisons)
            - meta.score (numeric comparisons)
            - meta.timestamp (numeric comparisons)
        :return: List of documents matching the filters, with score set to None.
        :raises ValkeyDocumentStoreError: If there's an error filtering documents.

        Example:
        ```python
        # Filter by category
        docs = document_store.filter_documents(
            filters={"field": "meta.category", "operator": "==", "value": "news"}
        )

        # Filter by numeric range
        docs = document_store.filter_documents(
            filters={"field": "meta.priority", "operator": ">=", "value": 5}
        )
        ```
        """
        _validate_filters(filters)

        try:
            # Valkey only performs vector similarity search
            # here we are querying with a dummy vector and the current number of documents as limit
            limit = self.count_documents()
            documents = self.search(self._dummy_vector, filters, limit)

            # when simply filtering, we don't want to return any scores
            # furthermore, we are querying with a dummy vector, so the scores are meaningless
            for doc in documents:
                doc.score = None

            return documents

        except Exception as e:
            msg = f"Error filtering documents in index '{self._index_name}'"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    async def async_filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Asynchronously filter documents by metadata without vector search.

        This is the async version of filter_documents(). It retrieves documents based on metadata filters
        without performing vector similarity search. Since Valkey Search requires vector queries, this method
        uses a dummy vector internally and removes the similarity scores from results.

        :param filters: Optional metadata filters in Haystack format. Supports filtering on:
            - meta.category (string equality)
            - meta.status (string equality)
            - meta.priority (numeric comparisons)
            - meta.score (numeric comparisons)
            - meta.timestamp (numeric comparisons)
        :return: List of documents matching the filters, with score set to None.
        :raises ValkeyDocumentStoreError: If there's an error filtering documents.

        Example:
        ```python
        # Filter by category
        docs = await document_store.async_filter_documents(
            filters={"field": "meta.category", "operator": "==", "value": "news"}
        )

        # Filter by numeric range
        docs = await document_store.async_filter_documents(
            filters={"field": "meta.priority", "operator": ">=", "value": 5}
        )
        ```
        """
        _validate_filters(filters)

        try:
            # Valkey only performs vector similarity search
            # here we are querying with a dummy vector and the max compatible limit
            limit = self.count_documents()
            documents = await self.async_search(self._dummy_vector, filters, limit)

            # when simply filtering, we don't want to return any scores
            # furthermore, we are querying with a dummy vector, so the scores are meaningless
            for doc in documents:
                doc.score = None

            return documents

        except Exception as e:
            msg = f"Error filtering documents in index '{self._index_name}'"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Write documents to the document store.

        This method stores documents with their embeddings and metadata in Valkey. The search index is
        automatically created if it doesn't exist. Documents without embeddings will be assigned a
        dummy vector for indexing purposes.

        :param documents: List of Document objects to store. Each document should have:
            - content: The document text
            - embedding: Vector representation (optional, dummy vector used if missing)
            - meta: Optional metadata dict with supported fields (category, status, priority, score, timestamp)
        :param policy: How to handle duplicate documents. Only NONE and OVERWRITE are supported.
            Defaults to DuplicatePolicy.NONE.
        :return: Number of documents successfully written.
        :raises ValkeyDocumentStoreError: If there's an error writing documents.
        :raises ValueError: If documents list contains invalid objects.

        Example:
        ```python
        documents = [
            Document(
                content="First document",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "news", "priority": 1}
            ),
            Document(
                content="Second document",
                embedding=[0.4, 0.5, 0.6],
                meta={"category": "blog", "priority": 2}
            )
        ]
        count = document_store.write_documents(documents)
        print(f"Wrote {count} documents")
        ```
        """
        self._validate_documents(documents)
        if len(documents) == 0:
            logger.warning("Calling ValkeyDocumentStore.write_documents() with empty list")
            return 0
        self._validate_policy(policy)

        client = self._get_connection()
        self._create_index()

        written_count = 0
        for doc in documents:
            try:
                doc_dict = self._prepare_document_dict(doc)
                key = f"{self._index_name}:{doc.id}"
                sync_glide_json.set(client, key, "$", json.dumps(doc_dict))
                written_count += 1
            except Exception as e:
                msg = f"Failed to write document {doc.id}: {e}"
                logger.error(msg)
                raise ValkeyDocumentStoreError(msg) from e

        return written_count

    async def async_write_documents(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Asynchronously write documents to the document store.

        This is the async version of write_documents(). It stores documents with their embeddings and
        metadata in Valkey using batch processing for improved performance. The search index is
        automatically created if it doesn't exist.

        :param documents: List of Document objects to store. Each document should have:
            - content: The document text
            - embedding: Vector representation (optional, dummy vector used if missing)
            - meta: Optional metadata dict with supported fields (category, status, priority, score, timestamp)
        :param policy: How to handle duplicate documents. Only NONE and OVERWRITE are supported.
            Defaults to DuplicatePolicy.NONE.
        :return: Number of documents successfully written.
        :raises ValkeyDocumentStoreError: If there's an error writing documents.
        :raises ValueError: If documents list contains invalid objects.

        Example:
        ```python
        documents = [
            Document(
                content="First document",
                embedding=[0.1, 0.2, 0.3],
                meta={"category": "news", "priority": 1}
            ),
            Document(
                content="Second document",
                embedding=[0.4, 0.5, 0.6],
                meta={"category": "blog", "priority": 2}
            )
        ]
        count = await document_store.async_write_documents(documents)
        print(f"Wrote {count} documents")
        ```
        """
        self._validate_documents(documents)
        if len(documents) == 0:
            logger.warning("Calling ValkeyDocumentStore.write_documents() with empty list")
            return 0
        self._validate_policy(policy)

        client = await self._get_async_connection()
        await self._async_create_index()

        def write_single_doc(doc: Document) -> Any:
            doc_dict = self._prepare_document_dict(doc)
            key = f"{self._index_name}:{doc.id}"
            return glide_json.set(client, key, "$", json.dumps(doc_dict))

        written_count = 0
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            try:
                await asyncio.gather(*[write_single_doc(doc) for doc in batch])
                written_count += len(batch)
            except Exception as e:
                msg = f"Failed to write batch starting at index {i}: {e}"
                logger.error(msg)
                raise ValkeyDocumentStoreError(msg) from e

        return written_count

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents from the document store by their IDs.

        This method removes documents from both the Valkey database and the search index.
        If some documents are not found, a warning is logged but the operation continues.

        :param document_ids: List of document IDs to delete. These should be the same IDs
            used when the documents were originally stored.
        :raises ValkeyDocumentStoreError: If there's an error deleting documents.

        Example:
        ```python
        # Delete specific documents
        document_store.delete_documents(["doc1", "doc2", "doc3"])

        # Delete a single document
        document_store.delete_documents(["single_doc_id"])
        ```
        """
        client = self._get_connection()
        assert client is not None

        keys: list[str | bytes] = [f"{self._index_name}:{doc_id}" for doc_id in document_ids]
        try:
            result = client.delete(keys)
            if result < len(document_ids):
                logger.warning(f"Some documents not found. Expected {len(document_ids)}, deleted {result}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            msg = f"Failed to delete documents: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    async def async_delete_documents(self, document_ids: list[str]) -> None:
        """
        Asynchronously delete documents from the document store by their IDs.

        This is the async version of delete_documents(). It removes documents from both the Valkey
        database and the search index. If some documents are not found, a warning is logged but
        the operation continues.

        :param document_ids: List of document IDs to delete. These should be the same IDs
            used when the documents were originally stored.
        :raises ValkeyDocumentStoreError: If there's an error deleting documents.

        Example:
        ```python
        # Delete specific documents
        await document_store.async_delete_documents(["doc1", "doc2", "doc3"])

        # Delete a single document
        await document_store.async_delete_documents(["single_doc_id"])
        ```
        """
        client = await self._get_async_connection()

        keys: list[str | bytes] = [f"{self._index_name}:{doc_id}" for doc_id in document_ids]
        try:
            result = await client.delete(keys)
            if result < len(document_ids):
                logger.warning(f"Some documents not found. Expected {len(document_ids)}, deleted {result}")
        except Exception as e:
            msg = f"Failed to delete documents: {e}"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    def delete_all_documents(self) -> None:
        """
        Delete all documents from the document store.

        This method removes all documents by dropping the entire search index. This is an efficient
        way to clear all data but requires recreating the index for future operations. If the index
        doesn't exist, the operation completes without error.

        :raises ValkeyDocumentStoreError: If there's an error dropping the index.

        Warning:
            This operation is irreversible and will permanently delete all documents and the search index.

        Example:
        ```python
        # Clear all documents from the store
        document_store.delete_all_documents()

        # The index will be automatically recreated on next write operation
        document_store.write_documents(new_documents)
        ```
        """
        client = self._get_connection()

        try:
            if self._has_index():
                # Drop the existing index
                sync_ft.dropindex(client, self._index_name)
                logger.info(f"Dropped index {self._index_name} and all its documents")
            else:
                logger.info(f"Index {self._index_name} does not exist, nothing to delete")
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            msg = f"Failed to delete all documents: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    async def async_delete_all_documents(self) -> None:
        """
        Asynchronously delete all documents from the document store.

        This is the async version of delete_all_documents(). It removes all documents by dropping
        the entire search index. This is an efficient way to clear all data but requires recreating
        the index for future operations. If the index doesn't exist, the operation completes without error.

        :raises ValkeyDocumentStoreError: If there's an error dropping the index.

        Warning:
            This operation is irreversible and will permanently delete all documents and the search index.

        Example:
        ```python
        # Clear all documents from the store
        await document_store.async_delete_all_documents()

        # The index will be automatically recreated on next write operation
        await document_store.async_write_documents(new_documents)
        ```
        """
        client = await self._get_async_connection()

        try:
            if await self._async_has_index():
                # Drop the existing index
                await ft.dropindex(client, self._index_name)
                logger.info(f"Dropped index {self._index_name} and all its documents")
            else:
                logger.info(f"Index {self._index_name} does not exist, nothing to delete")
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            msg = f"Failed to delete all documents: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    def search(
        self,
        embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        *,
        with_embedding: bool = True,
    ) -> list[Document]:
        """
        Search for documents using vector similarity.

        This method performs vector similarity search using the configured distance metric
        (cosine, L2, or inner product).
        Results are ranked by similarity score and can be filtered by metadata. If the search index doesn't exist,
        an empty list is returned with a warning logged.

        :param embedding: Query embedding vector for similarity search. Must match the embedding_dim
            configured for the document store.
        :param filters: Optional metadata filters in Haystack format. Supports filtering on:
            - meta.category (string equality)
            - meta.status (string equality)
            - meta.priority (numeric comparisons: ==, !=, >, >=, <, <=)
            - meta.score (numeric comparisons)
            - meta.timestamp (numeric comparisons)
        :param limit: Maximum number of documents to return. Must be > 0.
        :param with_embedding: Whether to include document embeddings in the results. Set to False
            to reduce response size when embeddings are not needed.
        :return: List of Document objects ranked by similarity score (highest first). Each document
            includes a similarity score in the score attribute. Returns empty list if index doesn't exist.
        :raises ValkeyDocumentStoreError: If there's an error performing the search.

        Example:
        ```python
        # Basic vector search
        results = document_store.search(
            embedding=[0.1, 0.2, 0.3],
            limit=5
        )

        # Search with metadata filters
        results = document_store.search(
            embedding=[0.1, 0.2, 0.3],
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            limit=10,
            with_embedding=False
        )

        # Access similarity scores
        for doc in results:
            print(f"Document: {doc.content}, Score: {doc.score}")
        ```
        """
        client = self._get_connection()

        if not self._has_index():
            logger.warning(f"Index {self._index_name} does not exist, returning empty results")
            return []

        if limit <= 0:
            logger.warning("Limit is <= 0, returning empty result")
            return []

        try:
            query, query_options = self._build_search_query_and_options(
                embedding, filters, limit, with_embedding=with_embedding
            )
            results = sync_ft.search(client, self._index_name, query, query_options)

            return self._parse_documents_from_ft(results, with_embedding=with_embedding)

        except Exception as e:
            logger.error(f"Failed to retrieve documents by embedding: {e}")
            msg = f"Failed to retrieve documents by embedding: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    async def async_search(
        self,
        embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        *,
        with_embedding: bool = True,
    ) -> list[Document]:
        """
        Asynchronously search for documents using vector similarity.

        This is the async version of search(). It performs vector similarity search using the configured
        distance metric (cosine, L2, or inner product). Results are ranked by similarity score and can be
        filtered by metadata. If the search index doesn't exist, an empty list is returned with a warning logged.

        :param embedding: Query embedding vector for similarity search. Must match the embedding_dim
            configured for the document store.
        :param filters: Optional metadata filters in Haystack format. Supports filtering on:
            - meta.category (string equality)
            - meta.status (string equality)
            - meta.priority (numeric comparisons: ==, !=, >, >=, <, <=)
            - meta.score (numeric comparisons)
            - meta.timestamp (numeric comparisons)
        :param limit: Maximum number of documents to return. Must be > 0.
        :param with_embedding: Whether to include document embeddings in the results. Set to False
            to reduce response size when embeddings are not needed.
        :return: List of Document objects ranked by similarity score (highest first). Each document
            includes a similarity score in the score attribute. Returns empty list if index doesn't exist.
        :raises ValkeyDocumentStoreError: If there's an error performing the search.

        Example:
        ```python
        # Basic vector search
        results = await document_store.async_search(
            embedding=[0.1, 0.2, 0.3],
            limit=5
        )

        # Search with metadata filters
        results = await document_store.async_search(
            embedding=[0.1, 0.2, 0.3],
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            limit=10,
            with_embedding=False
        )

        # Access similarity scores
        for doc in results:
            print(f"Document: {doc.content}, Score: {doc.score}")
        ```
        """
        client = await self._get_async_connection()

        if not await self._async_has_index():
            logger.warning(f"Index {self._index_name} does not exist, returning empty results")
            return []

        if limit == 0:
            logger.warning(f"Limit cannot be zero: {limit}")
            return []

        try:
            query, query_options = self._build_search_query_and_options(
                embedding, filters, limit, with_embedding=with_embedding
            )
            results = await ft.search(client, self._index_name, query, query_options)

            return self._parse_documents_from_ft(results, with_embedding=with_embedding)

        except Exception as e:
            msg = f"Failed to retrieve documents by embedding: {e}"
            logger.error(msg)
            raise ValkeyDocumentStoreError(msg) from e

    def _prepare_document_dict(self, doc: Document) -> dict:
        """Prepare document dictionary for storage."""
        payload = doc.to_dict(flatten=False)
        payload.pop("embedding", None)

        meta = doc.meta or {}
        doc_dict = {
            "id": doc.id,
            "payload": payload,
            "meta_category": meta.get("category", ""),
            "meta_status": meta.get("status", ""),
            "meta_priority": meta.get("priority", 0),
            "meta_score": meta.get("score", 0.0),
            "meta_timestamp": meta.get("timestamp", 0),
        }

        if doc.embedding is not None:
            doc_dict["vector"] = doc.embedding
        else:
            doc_dict["vector"] = [ValkeyDocumentStore._DUMMY_VALUE] * self._embedding_dim

        return doc_dict

    @staticmethod
    def _parse_documents_from_ft(raw: Any, *, with_embedding: bool) -> list[Document]:
        documents: list[Document] = []
        if not raw or raw[0] == 0:
            return documents
        if raw[1] is None:
            return documents
        for _, doc_info in raw[1].items():
            # doc_key_str = doc_key.decode() if isinstance(doc_key, bytes) else doc_key
            # doc_id = doc_key_str.split(":")[-1]  # Extract ID from key

            # Get payload from doc_info
            payload_data = doc_info.get(b"payload")
            if payload_data:
                if isinstance(payload_data, bytes):
                    payload_str = payload_data.decode()
                    payload = json.loads(payload_str)
                else:
                    payload = payload_data
            else:
                continue

            # Get embedding from doc_info if requested
            embedding = None
            if with_embedding:
                vector_data = doc_info.get(b"vector")
                if vector_data:
                    vector_str = vector_data.decode() if isinstance(vector_data, bytes) else vector_data
                    embedding = json.loads(vector_str)

            # Get similarity score from search results
            score_data = doc_info.get(b"__vector_score")
            similarity_score = None
            if score_data:
                score_str = score_data.decode() if isinstance(score_data, bytes) else score_data
                similarity_score = float(score_str)

            # Reconstruct document using from_dict for proper deserialization
            doc = Document.from_dict(payload)
            if embedding and not all(x == ValkeyDocumentStore._DUMMY_VALUE for x in embedding):
                doc.embedding = embedding
            # Set similarity score from vector search
            doc.score = similarity_score
            documents.append(doc)

        return documents

    @staticmethod
    def _build_search_query_and_options(
        embedding: list[float], filters: dict[str, Any] | None, limit: int, *, with_embedding: bool
    ) -> tuple[str, FtSearchOptions]:
        # Validate and normalize filters
        if filters:
            _validate_filters(filters)
            filter_query = _normalize_filters(filters)
        else:
            filter_query = "*"

        # Set return fields
        return_fields = [
            ReturnField("$.id", alias="id"),
            ReturnField("$.payload", alias="payload"),
            ReturnField("__vector_score", alias="score"),
        ]
        if with_embedding:
            return_fields.append(ReturnField("$.vector", alias="vector"))

        vector_param_name = "query_vector"

        # Combine filters with vector search
        query = f"{filter_query}=>[KNN {limit} @vector ${vector_param_name}]"
        query_options = FtSearchOptions(
            params={vector_param_name: ValkeyDocumentStore._to_float32_bytes(embedding)}, return_fields=return_fields
        )

        return query, query_options

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        """Validate document list."""
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"DocumentStore.write_documents() expects a list of Documents but got an element of {type(doc)}."
                raise ValueError(msg)

    @staticmethod
    def _validate_policy(policy: DuplicatePolicy) -> None:
        """Validate duplicate policy."""
        if policy not in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            logger.warning(
                f"ValkeyDocumentStore only supports `DuplicatePolicy.OVERWRITE`"
                f"but got {policy}. Overwriting duplicates is enabled by default."
            )

    @staticmethod
    def _to_float32_bytes(vec: list[float]) -> bytes:
        return b"".join(struct.pack("<f", x) for x in vec)

    @staticmethod
    def _verify_node_list(node_list: list[tuple[str, int]]) -> None:
        if not node_list:
            msg = "Node list is empty. Cannot create Valkey client"
            logger.error(msg)
            raise DocumentStoreError(msg) from None

    @staticmethod
    def _build_credentials(username: str | None, password: str | None) -> ServerCredentials | None:
        # Setup credentials if password is provided (username is optional, defaults to "default")
        if password:
            return ServerCredentials(username=username, password=password)
        return None

    @staticmethod
    def _parse_metric(metric: str) -> DistanceMetricType:
        try:
            return ValkeyDocumentStore._METRIC_MAP[metric]
        except KeyError as err:
            allowed_metrics = list(ValkeyDocumentStore._METRIC_MAP.keys())
            msg = f"Unsupported metric: {metric}. Allowed: {allowed_metrics}"
            raise ValueError(msg) from err
