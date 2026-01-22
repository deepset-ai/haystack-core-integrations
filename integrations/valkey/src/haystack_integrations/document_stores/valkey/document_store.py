# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, ClassVar, Literal

from dataclasses import replace

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
    FtSearchLimit,
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
from haystack.errors import FilterError
from haystack.utils import Secret

from haystack_integrations.document_stores.valkey.filters import _normalize_filters, _validate_filters

logger = logging.getLogger(__name__)


class ValkeyDocumentStoreError(DocumentStoreError):
    pass


class ValkeyDocumentStore(DocumentStore):
    """
    A document store implementation using Valkey with vector search capabilities.

    This document store provides persistent storage for documents with embeddings and supports
    vector similarity search using the Valkey Search module. It's designed for high-performance
    retrieval applications requiring both semantic search and metadata filtering.

    Key features:
    - Vector similarity search with HNSW algorithm
    - Metadata filtering on tag and numeric fields
    - Configurable distance metrics (L2, cosine, inner product)
    - Batch operations for efficient document management
    - Both synchronous and asynchronous operations
    - Cluster and standalone mode support

    Supported filterable Document metadata fields:
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
    results = document_store._embedding_retrival(
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
        username: Secret | None = Secret.from_env_var("VALKEY_USERNAME", strict=False),  # noqa: B008
        password: Secret | None = Secret.from_env_var("VALKEY_PASSWORD", strict=False),  # noqa: B008
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
        metadata_fields: dict[str, type[str] | type[int]] | None = None,
    ):
        """
        Creates a new ValkeyDocumentStore instance.

        :param nodes_list: List of (host, port) tuples for Valkey nodes. Defaults to [("localhost", 6379)].
        :param cluster_mode: Whether to connect in cluster mode. Defaults to False.
        :param use_tls: Whether to use TLS for connections. Defaults to False.
        :param username: Username for authentication. If not provided, reads from VALKEY_USERNAME environment variable.
            Defaults to None.
        :param password: Password for authentication. If not provided, reads from VALKEY_PASSWORD environment variable.
            Defaults to None.
        :param request_timeout: Request timeout in milliseconds. Defaults to 500.
        :param retry_attempts: Number of retry attempts for failed operations. Defaults to 3.
        :param retry_base_delay_ms: Base delay in milliseconds for exponential backoff. Defaults to 1000.
        :param retry_exponent_base: Exponent base for exponential backoff calculation. Defaults to 2.
        :param batch_size: Number of documents to process in a single batch for async operations. Defaults to 100.
        :param index_name: Name of the search index. Defaults to "haystack_document".
        :param distance_metric: Distance metric for vector similarity. Options: "l2", "cosine", "ip" (inner product).
            Defaults to "cosine".
        :param embedding_dim: Dimension of document embeddings. Defaults to 768.
        :param metadata_fields: Dictionary mapping metadata field names to Python types for filtering.
            Supported types: str (for exact matching), int (for numeric comparisons).
            Example: {"category": str, "priority": int}.
            If not provided, no metadata fields will be indexed for filtering.
        """
        self._index_name = index_name
        self._distance_metric = self._parse_metric(distance_metric)
        self._embedding_dim = embedding_dim
        self._dummy_vector = [ValkeyDocumentStore._DUMMY_VALUE] * self._embedding_dim
        
        # Validate and normalize metadata fields
        self._metadata_fields = self._validate_and_normalize_metadata_fields(metadata_fields or {})

        self._nodes_list: list[tuple[str, int]] = nodes_list or [("localhost", 6379)]
        self._cluster_mode: bool = cluster_mode
        self._use_tls: bool = use_tls
        self._username: Secret | None = username
        self._password: Secret | None = password
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
            raise ValkeyDocumentStoreError(msg) from e

    async def _get_connection_async(self) -> GlideClient | GlideClusterClient:
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
            raise ValkeyDocumentStoreError(msg) from e

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.error("Failed to close Valkey client: {error}", error=e)
                pass
        self._client = None

    async def close_async(self) -> None:
        if self._async_client:
            try:
                await self._async_client.close()
            except Exception as e:
                logger.error("Failed to close Valkey client: {error}", error=e)
                pass
        self._async_client = None

    def _has_index(self) -> bool:
        client = self._get_connection()

        try:
            sync_ft.info(client, self._index_name)
            return True
        except Exception as e:
            logger.info("Valkey index check failed for {index_name}, {error}", index_name=self._index_name, error=e)
            return False

    async def _has_index_async(self) -> bool:
        client = await self._get_connection_async()

        try:
            await ft.info(client, self._index_name)
            return True
        except Exception as e:
            logger.info("Valkey index check failed for {index_name}, {error}", index_name=self._index_name, error=e)
            return False

    def _prepare_index_fields(self) -> list[Field]:
        """Prepare index fields configuration."""
        fields = [
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
        ]
        
        # _metadata_fields keys already have meta_ prefix
        for field_name, field_type in self._metadata_fields.items():
            field_path = f"$.{field_name}"
            
            if field_type == "tag":
                fields.append(TagField(field_path, alias=field_name))
            elif field_type == "numeric":
                fields.append(NumericField(field_path, alias=field_name))
        
        return fields

    def _create_index(self) -> None:
        client = self._get_connection()

        try:
            if self._has_index():
                logger.info("Index {index_name} already exists", index_name=self._index_name)
                return

            fields = self._prepare_index_fields()
            options = FtCreateOptions(DataType.JSON)
            ok = sync_ft.create(client, self._index_name, fields, options)
            if ok not in (b"OK", "OK"):
                msg = f"FT.CREATE failed for index '{self._index_name}': {ok!r}"
                raise ValkeyDocumentStoreError(msg)

        except Exception as e:
            msg = f"Error creating collection {self._index_name}"
            raise ValkeyDocumentStoreError(msg) from e

    async def _create_index_async(self) -> None:
        try:
            client = await self._get_connection_async()

            if await self._has_index_async():
                logger.info("Index {index_name} already exists", index_name=self._index_name)
                return

            fields = self._prepare_index_fields()
            options = FtCreateOptions(DataType.JSON)
            ok = await ft.create(client, self._index_name, fields, options)
            if ok not in (b"OK", "OK"):
                msg = f"FT.CREATE failed for index '{self._index_name}': {ok!r}"
                raise ValkeyDocumentStoreError(msg) from None

        except Exception as e:
            msg = f"Error creating collection {self._index_name}"
            raise ValkeyDocumentStoreError(msg) from e

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        metadata_fields_for_ser = {}
        for field_name, field_type in self._metadata_fields.items():
            # Remove meta_ prefix: meta_category -> category
            clean_name = field_name[5:] if field_name.startswith("meta_") else field_name
            metadata_fields_for_ser[clean_name] = str if field_type == "tag" else int
        
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
            metadata_fields=metadata_fields_for_ser if metadata_fields_for_ser else None,
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
            logger.info("Index {index_name} does not exist", index_name=self._index_name)
            return 0

        try:
            info = sync_ft.info(client, self._index_name)
            num_docs = info[b"num_docs"]
            if isinstance(num_docs, (str, bytes, int)):
                return int(num_docs)
            logger.warning("Unexpected type for num_docs: {type_num_docs}", type_num_docs=type(num_docs))
            return 0
        except Exception as e:
            msg = f"Error counting documents in index '{self._index_name}'"
            raise ValkeyDocumentStoreError(msg) from e

    async def count_documents_async(self) -> int:
        """
        Asynchronously return the number of documents stored in the document store.

        This method queries the Valkey Search index to get the total count of indexed documents.
        If the index doesn't exist, it returns 0. This is the async version of count_documents().

        :return: The number of documents in the document store.
        :raises ValkeyDocumentStoreError: If there's an error accessing the index or counting documents.

        Example:
        ```python
        document_store = ValkeyDocumentStore()
        count = await document_store.count_documents_async()
        print(f"Total documents: {count}")
        ```
        """
        client = await self._get_connection_async()

        if not await self._has_index_async():
            logger.info("Index {index_name} does not exist", index_name=self._index_name)
            return 0

        try:
            info = await ft.info(client, self._index_name)
            num_docs = info[b"num_docs"]
            if isinstance(num_docs, (str, bytes, int)):
                return int(num_docs)
            logger.warning("Unexpected type for num_docs: {type_num_docs}", type_num_docs=type(num_docs))
            return 0
        except Exception as e:
            msg = f"Error counting documents in index '{self._index_name}': {e}"
            raise ValkeyDocumentStoreError(msg) from e

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
            documents = self._embedding_retrieval(self._dummy_vector, filters, limit)

            # when simply filtering, we don't want to return any scores
            # furthermore, we are querying with a dummy vector, so the scores are meaningless
            docs_no_score = []
            for doc in documents:
                docs_no_score.append(replace(doc, score=None))

            return docs_no_score

        except FilterError:
            raise
        except Exception as e:
            msg = f"Error filtering documents in index '{self._index_name}'"
            raise ValkeyDocumentStoreError(msg) from e

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
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
        docs = await document_store.filter_documents_async(
            filters={"field": "meta.category", "operator": "==", "value": "news"}
        )

        # Filter by numeric range
        docs = await document_store.filter_documents_async(
            filters={"field": "meta.priority", "operator": ">=", "value": 5}
        )
        ```
        """
        _validate_filters(filters)

        try:
            # Valkey only performs vector similarity search
            # here we are querying with a dummy vector and the max compatible limit
            limit = await self.count_documents_async()
            documents = await self._embedding_retrieval_async(self._dummy_vector, filters, limit)

            # when simply filtering, we don't want to return any scores
            # furthermore, we are querying with a dummy vector, so the scores are meaningless
            docs_no_score = []
            for doc in documents:
                docs_no_score.append(replace(doc, score = None))

            return docs_no_score

        except FilterError:
            raise
        except Exception as e:
            msg = f"Error filtering documents in index '{self._index_name}'"
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
                raise ValkeyDocumentStoreError(msg) from e

        return written_count

    async def write_documents_async(
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
        count = await document_store.write_documents_async(documents)
        print(f"Wrote {count} documents")
        ```
        """
        self._validate_documents(documents)
        if len(documents) == 0:
            logger.warning("Calling ValkeyDocumentStore.write_documents_async() with empty list")
            return 0
        self._validate_policy(policy)

        client = await self._get_connection_async()
        await self._create_index_async()

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

        keys: list[str | bytes] = [f"{self._index_name}:{doc_id}" for doc_id in document_ids]
        try:
            result = client.delete(keys)
            if result < len(document_ids):
                logger.warning("Some documents not found. Expected {len_document_ids}, deleted {result}", len_document_ids=len(document_ids), result=result)
        except Exception as e:
            msg = f"Failed to delete documents: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    async def delete_documents_async(self, document_ids: list[str]) -> None:
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
        await document_store.delete_documents_async(["doc1", "doc2", "doc3"])

        # Delete a single document
        await document_store.delete_documents_async(["single_doc_id"])
        ```
        """
        client = await self._get_connection_async()

        keys: list[str | bytes] = [f"{self._index_name}:{doc_id}" for doc_id in document_ids]
        try:
            result = await client.delete(keys)
            if result < len(document_ids):
                logger.warning("Some documents not found. Expected {len_document_ids}, deleted {result}", len_document_ids=len(document_ids), result=result)
        except Exception as e:
            msg = f"Failed to delete documents: {e}"
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
                logger.info("Dropped index {index_name} and all its documents", index_name=self._index_name)
            else:
                logger.info("Index {index_name} does not exist, nothing to delete", index_name=self._index_name)
        except Exception as e:
            msg = f"Failed to delete all documents: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    async def delete_all_documents_async(self) -> None:
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
        await document_store.delete_all_documents_async()

        # The index will be automatically recreated on next write operation
        await document_store.write_documents_async(new_documents)
        ```
        """
        client = await self._get_connection_async()

        try:
            if await self._has_index_async():
                # Drop the existing index
                await ft.dropindex(client, self._index_name)
                logger.info("Dropped index {index_name} and all its documents", index_name=self._index_name)
            else:
                logger.info("Index {index_name} does not exist, nothing to delete", index_name=self._index_name)
        except Exception as e:
            msg = f"Failed to delete all documents: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    def _embedding_retrieval(
        self,
        embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        *,
        with_embedding: bool = True,
    ) -> list[Document]:
        """
        Retrieve documents using vector similarity.

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
        # Basic vector retrieval
        results = document_store._embedding_retrieval(
            embedding=[0.1, 0.2, 0.3],
            limit=5
        )

        # Retrieve with metadata filters
        results = document_store._embedding_retrieval(
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
            logger.warning("Index {index_name} does not exist, returning empty results", index_name=self._index_name)
            return []

        if limit <= 0:
            logger.warning("Limit is <= 0, returning empty result")
            return []

        try:
            query, query_options = self._build_search_query_and_options(
                embedding, filters, limit, with_embedding=with_embedding, supported_fields=self._metadata_fields
            )
            results = sync_ft.search(client, self._index_name, query, query_options)

            return self._parse_documents_from_ft(results, with_embedding=with_embedding)

        except FilterError:
            raise
        except Exception as e:
            msg = f"Failed to retrieve documents by embedding: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    async def _embedding_retrieval_async(
        self,
        embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        *,
        with_embedding: bool = True,
    ) -> list[Document]:
        """
        Asynchronously retrieve documents using vector similarity.

        This is the async version of _embedding_retrieval(). It performs vector similarity search using the configured
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
        # Basic vector retrieval
        results = await document_store._embedding_retrieval_async(
            embedding=[0.1, 0.2, 0.3],
            limit=5
        )

        # Retrieval with metadata filters
        results = await document_store._embedding_retrieval_async(
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
        client = await self._get_connection_async()

        if not await self._has_index_async():
            logger.warning("Index {index_name} does not exist, returning empty results", index_name=self._index_name)
            return []

        if limit == 0:
            logger.warning("Limit cannot be zero: {limit}", limit=limit)
            return []

        try:
            query, query_options = self._build_search_query_and_options(
                embedding, filters, limit, with_embedding=with_embedding, supported_fields=self._metadata_fields
            )
            results = await ft.search(client, self._index_name, query, query_options)

            return self._parse_documents_from_ft(results, with_embedding=with_embedding)

        except FilterError:
            raise
        except Exception as e:
            msg = f"Failed to retrieve documents by embedding: {e}"
            raise ValkeyDocumentStoreError(msg) from e

    def _prepare_document_dict(self, doc: Document) -> dict:
        """Prepare document dictionary for storage."""
        payload = doc.to_dict(flatten=False)
        payload.pop("embedding", None)

        meta = doc.meta or {}
        doc_dict = {"id": doc.id, "payload": payload}
        
        # _metadata_fields keys already have meta_ prefix
        for field_name_with_prefix, field_type in self._metadata_fields.items():
            # Extract original field name: meta_category -> category
            field_name = field_name_with_prefix[5:]  # Remove "meta_"
            doc_dict[field_name_with_prefix] = meta.get(field_name, None)

        doc_dict["vector"] = doc.embedding if doc.embedding else [self._DUMMY_VALUE] * self._embedding_dim
        return doc_dict

    @staticmethod
    def _parse_documents_from_ft(raw: Any, *, with_embedding: bool) -> list[Document]:
        """
        Parse raw Valkey FT.SEARCH results into Document objects.

        :param raw: Raw search results from Valkey FT.SEARCH command. Expected format is a list where
            raw[0] is the total count (int) and raw[1] is a dict mapping document keys to field dictionaries
            containing b"payload", b"vector", and b"__vector_score" as byte keys.
        :param with_embedding: Whether to include embeddings in parsed documents.
        :return: List of Document objects with scores and optional embeddings.
        """
        documents: list[Document] = []
        # Handle empty results: raw is None/empty, or raw[0] (result count) is 0
        # This occurs when no documents match the query filters or the index is empty
        if not raw or raw[0] == 0:
            return documents
        
        for doc_info in raw[1].values():
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

            # Get embedding from doc_info if requested and add to payload
            if with_embedding:
                vector_data = doc_info.get(b"vector")
                if vector_data:
                    vector_str = vector_data.decode() if isinstance(vector_data, bytes) else vector_data
                    embedding = json.loads(vector_str)
                    # Only add embedding if it's not a dummy vector
                    if not all(x == ValkeyDocumentStore._DUMMY_VALUE for x in embedding):
                        payload["embedding"] = embedding

            # Get similarity score from search results and add to payload
            score_data = doc_info.get(b"__vector_score")
            if score_data:
                score_str = score_data.decode() if isinstance(score_data, bytes) else score_data
                payload["score"] = float(score_str)

            # Reconstruct document using from_dict with updated payload
            doc = Document.from_dict(payload)
            documents.append(doc)

        return documents

    @staticmethod
    def _build_search_query_and_options(
        embedding: list[float], filters: dict[str, Any] | None, limit: int, *, with_embedding: bool,
        supported_fields: dict[str, str]
    ) -> tuple[str, FtSearchOptions]:
        if filters:
            _validate_filters(filters)
            filter_query = _normalize_filters(filters, supported_fields)
        else:
            filter_query = "*"

        return_fields = [
            ReturnField("$.id", alias="id"),
            ReturnField("$.payload", alias="payload"),
            ReturnField("__vector_score", alias="score"),
        ]
        if with_embedding:
            return_fields.append(ReturnField("$.vector", alias="vector"))

        query = f"{filter_query}=>[KNN {limit} @vector $query_vector]"
        query_options = FtSearchOptions(
            params={"query_vector": ValkeyDocumentStore._to_float32_bytes(embedding)}, 
            return_fields=return_fields,
            limit=FtSearchLimit(offset=0, count=limit)
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
                "ValkeyDocumentStore only supports `DuplicatePolicy.OVERWRITE`"
                "but got {policy}. Overwriting duplicates is enabled by default.",
                policy=policy
            )

    @staticmethod
    def _to_float32_bytes(vec: list[float]) -> bytes:
        return b"".join(struct.pack("<f", x) for x in vec)

    @staticmethod
    def _verify_node_list(node_list: list[tuple[str, int]]) -> None:
        if not node_list:
            msg = "Node list is empty. Cannot create Valkey client"
            raise DocumentStoreError(msg) from None

    @staticmethod
    def _build_credentials(username: Secret | None, password: Secret | None) -> ServerCredentials | None:
        # Setup credentials if password is provided (username is optional, defaults to "default")
        if password and password.resolve_value():
            username_str = username.resolve_value() if username else None
            password_str = password.resolve_value()
            return ServerCredentials(username=username_str, password=password_str)
        return None

    @staticmethod
    def _parse_metric(metric: str) -> DistanceMetricType:
        try:
            return ValkeyDocumentStore._METRIC_MAP[metric]
        except KeyError as err:
            allowed_metrics = list(ValkeyDocumentStore._METRIC_MAP.keys())
            msg = f"Unsupported metric: {metric}. Allowed: {allowed_metrics}"
            raise ValueError(msg) from err

    @staticmethod
    def _validate_and_normalize_metadata_fields(metadata_fields: dict[str, type[str] | type[int]]) -> dict[str, str]:
        """
        Validate and normalize metadata field definitions.
        
        :param metadata_fields: User-provided metadata field definitions mapping field names to Python types.
        :return: Normalized metadata fields with meta_ prefix mapping to Valkey field types ("tag" or "numeric").
        :raises ValueError: If field definitions are invalid.
        """
        if not isinstance(metadata_fields, dict):
            msg = "metadata_fields must be a dictionary"
            raise ValueError(msg)
        
        TYPE_MAPPING = {str: "tag", int: "numeric"}
        
        normalized = {}
        for field_name, field_type in metadata_fields.items():
            if not isinstance(field_name, str) or not field_name:
                msg = f"Field name must be a non-empty string, got {field_name!r}"
                raise ValueError(msg)
            
            if field_type not in TYPE_MAPPING:
                msg = f"Unsupported field type {field_type!r} for field '{field_name}'. Supported: {list(TYPE_MAPPING.keys())}"
                raise ValueError(msg)
            
            normalized[f"meta_{field_name}"] = TYPE_MAPPING[field_type]
        
        return normalized
