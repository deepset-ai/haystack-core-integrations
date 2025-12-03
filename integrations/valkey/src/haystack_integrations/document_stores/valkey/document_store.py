import json
import asyncio
from functools import singledispatch
from typing import List, Optional, Dict, Any, Literal
from haystack.document_stores.types import DocumentStore
from haystack.dataclasses.document import Document
from glide import (
    BackoffStrategy,
    GlideClient,
    GlideClientConfiguration,
    GlideClusterClientConfiguration,
    GlideClusterClient,
    glide_json,
)
from glide import ft
from glide_sync import (
    BackoffStrategy as SyncBackoffStrategy,
    GlideClient as SyncGlideClient,
    GlideClientConfiguration as SyncGlideClientConfiguration,
    GlideClusterClientConfiguration as SyncGlideClusterClientConfiguration,
    GlideClusterClient as SyncGlideClusterClient,
    glide_json as sync_glide_json,
)
from glide_sync import ft as sync_ft
from glide_shared import (
    NodeAddress,
    ServerCredentials,
    # GlideClientConfiguration,
    # GlideClusterClientConfiguration
)
from glide_shared.commands.server_modules.ft_options.ft_create_options import (
    DataType,
    DistanceMetricType,
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
from glide_shared.exceptions import RequestError
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.errors import DocumentStoreError
from haystack import logging, default_to_dict, default_from_dict

from haystack_integrations.document_stores.valkey.utils import _serialize_for_json, _to_float32_bytes
from haystack_integrations.document_stores.valkey.filters import _normalize_filters, _validate_filters

logger = logging.getLogger(__name__)


class ValkeyDocumentStoreError(DocumentStoreError):
    pass


class ValkeyDocumentStore(DocumentStore):
    _METRIC_MAP: dict[str, DistanceMetricType] = {
        "l2": DistanceMetricType.L2,
        "cosine": DistanceMetricType.COSINE,
        "ip": DistanceMetricType.IP,
    }

    def __init__(
        self,
        # Connection configuration
        nodes_list: list[tuple[str, int]] | None = [("localhost", 6379)],
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
        self._dummy_vector = [-10.0] * self._embedding_dim

        self._nodes_list: list[tuple[str, int]] = nodes_list
        self._cluster_mode: bool = cluster_mode
        self._use_tls: bool = use_tls
        self._username: str | None = username
        self._password: str | None = password
        self._request_timeout: int = request_timeout
        self._retry_attempts: int = retry_attempts
        self._retry_base_delay_ms: int = retry_base_delay_ms
        self._retry_exponent_base: int = retry_exponent_base
        self._batch_size: int = batch_size
        self._client: SyncGlideClient | None = None
        self._async_client: GlideClient | None = None

    def _parse_metric(self, metric: str) -> DistanceMetricType:
        try:
            return self._METRIC_MAP[metric]
        except KeyError:
            raise ValueError(f"Unsupported metric: {metric}. Allowed: {list(self._METRIC_MAP.keys())}")

    def _get_connection(self) -> SyncGlideClient:
        if self._client:
            return self._client

        if not self._nodes_list:
            logger.error("Node list is empty. Cannot create Valkey addresses.")
            raise DocumentStoreError("Node list is empty. Cannot create Valkey client")

        addresses = [NodeAddress(host, port) for host, port in self._nodes_list]
        reconnect_strategy = SyncBackoffStrategy(
            num_of_retries=self._retry_attempts,
            factor=self._retry_base_delay_ms,
            exponent_base=self._retry_exponent_base,
        )

        # Setup credentials if provided
        credentials = None
        if self._username or self._password:
            credentials = ServerCredentials(username=self._username, password=self._password)

        try:
            if self._cluster_mode:
                config = SyncGlideClusterClientConfiguration(
                    addresses=addresses,
                    use_tls=self._use_tls,
                    credentials=credentials,
                    request_timeout=self._request_timeout,
                    reconnect_strategy=reconnect_strategy,
                )
                self._client = SyncGlideClusterClient.create(config)
            else:
                config = SyncGlideClientConfiguration(
                    addresses=addresses,
                    use_tls=self._use_tls,
                    credentials=credentials,
                    request_timeout=self._request_timeout,
                    reconnect_strategy=reconnect_strategy,
                )
                self._client = SyncGlideClient.create(config)
            return self._client
        except Exception as e:
            logger.error(f"Failed to connect to Valkey: {e}")
            raise ValkeyDocumentStoreError(f"Connection failed: {e}")

    async def _get_async_connection(self) -> GlideClient:
        if self._async_client:
            return self._async_client

        if not self._nodes_list:
            logger.error("Node list is empty. Cannot create Valkey addresses.")
            raise DocumentStoreError("Node list is empty. Cannot create Valkey client")

        addresses = [NodeAddress(host, port) for host, port in self._nodes_list]
        reconnect_strategy = BackoffStrategy(
            num_of_retries=self._retry_attempts,
            factor=self._retry_base_delay_ms,
            exponent_base=self._retry_exponent_base,
        )

        # Setup credentials if provided
        credentials = None
        if self._username or self._password:
            credentials = ServerCredentials(username=self._username, password=self._password)

        try:
            if self._cluster_mode:
                config = GlideClusterClientConfiguration(
                    addresses=addresses,
                    use_tls=self._use_tls,
                    credentials=credentials,
                    request_timeout=self._request_timeout,
                    reconnect_strategy=reconnect_strategy,
                )
                self._async_client = await GlideClusterClient.create(config)
            else:
                config = GlideClientConfiguration(
                    addresses=addresses,
                    use_tls=self._use_tls,
                    credentials=credentials,
                    request_timeout=self._request_timeout,
                    reconnect_strategy=reconnect_strategy,
                )
                self._async_client = await GlideClient.create(config)
            return self._async_client
        except Exception as e:
            logger.error(f"Failed to connect to Valkey: {e}")
            raise ValkeyDocumentStoreError(f"Connection failed: {e}")

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.error(f"Failed to close Valkey client: {e}")
                pass
        self._client = None

    async def async_close(self) -> None:
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.error(f"Failed to close Valkey client: {e}")
                pass
        self._async_client = None

    def _has_index(self) -> bool:
        client = self._get_connection()
        assert client is not None

        try:
            sync_ft.info(client, self._index_name)
            return True
        except Exception as e:
            logger.info(f"Valkey index check failed for {self._index_name}, {e}")
            return False

    async def _async_has_index(self) -> bool:
        client = await self._get_async_connection()
        assert client is not None

        try:
            await ft.info(client, self._index_name)
            return True
        except Exception as e:
            logger.info(f"Valkey index check failed for {self._index_name}, {e}")
            return False

    def _create_index(self) -> None:
        try:
            if self._has_index():
                logger.info(f"Index {self._index_name} already exists")
                return

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
                # Filterable metadata fields
                TagField("$.meta_category", alias="meta_category"),
                TagField("$.meta_status", alias="meta_status"),
                NumericField("$.meta_priority", alias="meta_priority"),
                NumericField("$.meta_score", alias="meta_score"),
                NumericField("$.meta_timestamp", alias="meta_timestamp"),
            ]

            options = FtCreateOptions(DataType.JSON)

            ok = sync_ft.create(self._client, self._index_name, fields, options)
            if ok not in (b"OK", "OK"):
                raise ValkeyDocumentStoreError(f"FT.CREATE failed for index '{self._index_name}': {ok!r}")

        except Exception as e:
            logger.error(f"Error creating collection {self._index_name}: {e}")
            raise ValkeyDocumentStoreError(e)

    async def _async_create_index(self) -> None:
        try:
            if await self._async_has_index():
                logger.info(f"Index {self._index_name} already exists")
                return

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
                # Filterable metadata fields
                TagField("$.meta_category", alias="meta_category"),
                TagField("$.meta_status", alias="meta_status"),
                NumericField("$.meta_priority", alias="meta_priority"),
                NumericField("$.meta_score", alias="meta_score"),
                NumericField("$.meta_timestamp", alias="meta_timestamp"),
            ]

            options = FtCreateOptions(DataType.JSON)

            ok = await ft.create(self._async_client, self._index_name, fields, options)
            if ok not in (b"OK", "OK"):
                raise ValkeyDocumentStoreError(f"FT.CREATE failed for index '{self._index_name}': {ok!r}")

        except Exception as e:
            logger.error(f"Error creating collection {self._index_name}: {e}")
            raise ValkeyDocumentStoreError(e)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        return default_to_dict(
            self,
            nodes_list=self._nodes_list,
            cluster_mode=self._cluster_mode,
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
    def from_dict(cls, data: dict[str, Any]) -> "ValkeyDocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        client = self._get_connection()
        assert client is not None

        if not self._has_index():
            logger.info(f"Index {self._index_name} does not exist")
            return 0

        try:
            info = sync_ft.info(client, self._index_name)
            return int(info[b"num_docs"])
        except Exception as e:
            logger.error(f"Error counting documents in index '{self._index_name}': {e}")
            raise ValkeyDocumentStoreError(e)

    async def async_count_documents(self) -> int:
        client = await self._get_async_connection()
        assert client is not None

        if not await self._async_has_index():
            logger.info(f"Index {self._index_name} does not exist")
            return 0

        try:
            info = await ft.info(client, self._index_name)
            return int(info[b"num_docs"])
        except Exception as e:
            logger.error(f"Error counting documents in index '{self._index_name}': {e}")
            raise ValkeyDocumentStoreError(e)

    def filter_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        Valkey Search supports filtering on the following fields:
        - meta_category (TagField): exact string matches (==, !=, in, not in)
        - meta_status (TagField): status filtering (==, !=, in, not in)
        - meta_priority (NumericField): numeric comparisons (==, !=, >, >=, <, <=, in, not in)
        - meta_score (NumericField): score filtering (==, !=, >, >=, <, <=, in, not in)
        - meta_timestamp (NumericField): date/time filtering (==, !=, >, >=, <, <=, in, not in)

        Logical operators: AND, OR

        Example filters:
        ```python
        # Simple filter
        filters = {"field": "meta.category", "operator": "==", "value": "news"}

        # Complex filter
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {"field": "meta.priority", "operator": ">=", "value": 5},
            ]
        }
        ```

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        """
        client = self._get_connection()
        assert client is not None
        self._create_index()

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
            logger.error(f"Failed to retrieve documents by embedding: {e}")
            raise ValkeyDocumentStoreError(f"Failed to retrieve documents by embedding: {e}")

    async def async_filter_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        Valkey Search supports filtering on the following fields:
        - meta_category (TagField): exact string matches (==, !=, in, not in)
        - meta_status (TagField): status filtering (==, !=, in, not in)
        - meta_priority (NumericField): numeric comparisons (==, !=, >, >=, <, <=, in, not in)
        - meta_score (NumericField): score filtering (==, !=, >, >=, <, <=, in, not in)
        - meta_timestamp (NumericField): date/time filtering (==, !=, >, >=, <, <=, in, not in)

        Logical operators: AND, OR

        Example filters:
        ```python
        # Simple filter
        filters = {"field": "meta.category", "operator": "==", "value": "news"}

        # Complex filter
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {"field": "meta.priority", "operator": ">=", "value": 5},
            ]
        }
        ```

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        """
        client = await self._get_async_connection()
        assert client is not None
        await self._async_create_index()

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
            logger.error(f"Failed to retrieve documents by embedding: {e}")
            raise ValkeyDocumentStoreError(f"Failed to retrieve documents by embedding: {e}")

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes Documents into the DocumentStore.

        :param documents: a list of Document objects.
        :param policy: the policy to apply when a Document with the same id already exists in the DocumentStore.
            - `DuplicatePolicy.NONE`: Default policy, behaviour depends on the Document Store.
            - `DuplicatePolicy.SKIP`: If a Document with the same id already exists, it is skipped and not written.
            - `DuplicatePolicy.OVERWRITE`: If a Document with the same id already exists, it is overwritten.
            - `DuplicatePolicy.FAIL`: If a Document with the same id already exists, an error is raised.
        :raises DuplicateError: If `policy` is set to `DuplicatePolicy.FAIL` and a Document with the same id already
            exists.
        :returns: The number of Documents written.
            If `DuplicatePolicy.OVERWRITE` is used, this number is always equal to the number of documents in input.
            If `DuplicatePolicy.SKIP` is used, this number can be lower than the number of documents in the input list.
        """
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"DocumentStore.write_documents() expects a list of Documents but got an element of {type(doc)}."
                raise ValueError(msg)

        if len(documents) == 0:
            logger.warning("Calling ValkeyDocumentStore.write_documents() with empty list")
            return 0

        if policy not in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            logger.warning(
                f"ValkeyDocumentStore only supports `DuplicatePolicy.OVERWRITE`"
                f"but got {policy}. Overwriting duplicates is enabled by default."
            )

        client = self._get_connection()
        assert client is not None
        self._create_index()

        written_count = 0
        for doc in documents:
            try:
                payload = doc.to_dict(flatten=False)
                payload.pop("embedding", None)

                # Extract filterable metadata fields
                meta = doc.meta or {}
                doc_dict = {
                    "id": doc.id,
                    "payload": json.dumps(payload),
                    "meta_category": meta.get("category", ""),
                    "meta_status": meta.get("status", ""),
                    "meta_priority": meta.get("priority", 0),
                    "meta_score": meta.get("score", 0.0),
                    "meta_timestamp": meta.get("timestamp", 0),
                }

                if doc.embedding is not None:
                    doc_dict["vector"] = doc.embedding
                else:
                    doc_dict["vector"] = [-1000.0] * self._embedding_dim

                key = f"{self._index_name}:{doc.id}"
                sync_glide_json.set(client, key, "$", json.dumps(doc_dict))
                written_count += 1

            except Exception as e:
                logger.error(f"Failed to write document {doc.id}: {e}")
                raise ValkeyDocumentStoreError(f"Failed to write document {doc.id}: {e}")

        return written_count

    async def async_write_documents(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"DocumentStore.write_documents() expects a list of Documents but got an element of {type(doc)}."
                raise ValueError(msg)

        if len(documents) == 0:
            logger.warning("Calling ValkeyDocumentStore.write_documents() with empty list")
            return 0

        if policy not in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            logger.warning(
                f"ValkeyDocumentStore only supports `DuplicatePolicy.OVERWRITE`"
                f"but got {policy}. Overwriting duplicates is enabled by default."
            )

        client = await self._get_async_connection()
        assert client is not None
        await self._async_create_index()

        def write_single_doc(doc: Document):
            payload = doc.to_dict(flatten=False)
            payload.pop("embedding", None)

            meta = doc.meta or {}
            doc_dict = {
                "id": doc.id,
                "payload": json.dumps(payload),
                "meta_category": meta.get("category", ""),
                "meta_status": meta.get("status", ""),
                "meta_priority": meta.get("priority", 0),
                "meta_score": meta.get("score", 0.0),
                "meta_timestamp": meta.get("timestamp", 0),
            }

            if doc.embedding is not None:
                doc_dict["vector"] = doc.embedding
            else:
                doc_dict["vector"] = [-1000.0] * self._embedding_dim

            key = f"{self._index_name}:{doc.id}"
            return glide_json.set(client, key, "$", json.dumps(doc_dict))

        written_count = 0
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            try:
                await asyncio.gather(*[write_single_doc(doc) for doc in batch])
                written_count += len(batch)
            except Exception as e:
                logger.error(f"Failed to write batch starting at index {i}: {e}")
                raise ValkeyDocumentStoreError(f"Failed to write batch: {e}")

        return written_count

    def delete_documents(self, document_ids: list[str]) -> None:
        client = self._get_connection()
        assert client is not None

        keys = [f"{self._index_name}:{doc_id}" for doc_id in document_ids]
        try:
            result = client.delete(keys)
            if result < len(document_ids):
                logger.warning(f"Some documents not found. Expected {len(document_ids)}, deleted {result}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise ValkeyDocumentStoreError(f"Failed to delete documents: {e}")

    async def async_delete_documents(self, document_ids: list[str]) -> None:
        client = await self._get_async_connection()
        assert client is not None

        keys = [f"{self._index_name}:{doc_id}" for doc_id in document_ids]
        try:
            result = await client.delete(keys)
            if result < len(document_ids):
                logger.warning(f"Some documents not found. Expected {len(document_ids)}, deleted {result}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise ValkeyDocumentStoreError(f"Failed to delete documents: {e}")

    def delete_all_documents(self) -> None:
        """
        Deletes all documents in the document store by dropping the index.
        """
        client = self._get_connection()
        assert client is not None

        try:
            if self._has_index():
                # Drop the existing index
                sync_ft.dropindex(client, self._index_name)
                logger.info(f"Dropped index {self._index_name} and all its documents")
            else:
                logger.info(f"Index {self._index_name} does not exist, nothing to delete")
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            raise ValkeyDocumentStoreError(f"Failed to delete all documents: {e}")

    async def async_delete_all_documents(self) -> None:
        """
        Asynchronously deletes all documents in the document store by dropping the index.
        """
        client = await self._get_async_connection()
        assert client is not None

        try:
            if await self._async_has_index():
                # Drop the existing index
                await ft.dropindex(client, self._index_name)
                logger.info(f"Dropped index {self._index_name} and all its documents")
            else:
                logger.info(f"Index {self._index_name} does not exist, nothing to delete")
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            raise ValkeyDocumentStoreError(f"Failed to delete all documents: {e}")

    def search(
        self,
        embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        limit: int = 10,
        with_embedding: bool = True,
    ) -> list[Document]:
        client = self._get_connection()
        assert client is not None
        self._create_index()

        if limit <= 0:
            logger.warning("Limit is <= 0, returning empty result")
            return []

        # Validate and normalize filters
        if filters:
            _validate_filters(filters)
            filter_query = _normalize_filters(filters)
        else:
            filter_query = "*"

        try:
            # Set return fields
            return_fields = [
                ReturnField("$.id", alias="id"),
                ReturnField("$.payload", alias="payload"),
                ReturnField("$.meta_category", alias="meta_category"),
                ReturnField("__vector_score", alias="score"),
            ]
            if with_embedding:
                return_fields.append(ReturnField("$.vector", alias="vector"))

            vector_param_name = "query_vector"
            # Combine filters with vector search
            query = f"{filter_query}=>[KNN {limit} @vector ${vector_param_name}]"
            query_options = FtSearchOptions(
                params={vector_param_name: _to_float32_bytes(embedding)}, return_fields=return_fields
            )

            results = sync_ft.search(client, self._index_name, query, query_options)

            documents = []
            if results[0] == 0:
                return documents
            for doc_key, doc_info in results[1].items():
                # doc_key_str = doc_key.decode() if isinstance(doc_key, bytes) else doc_key
                # doc_id = doc_key_str.split(":")[-1]  # Extract ID from key

                # Get payload from doc_info
                payload_data = doc_info.get(b"payload")
                if payload_data:
                    payload_str = payload_data.decode() if isinstance(payload_data, bytes) else payload_data
                    # Unescape the JSON string
                    payload_str = payload_str.replace('\\"', '"')
                    payload = json.loads(payload_str)
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
                if embedding != self._dummy_vector:
                    doc.embedding = embedding
                doc.embedding = embedding
                # Set similarity score from vector search
                doc.score = similarity_score
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve documents by embedding: {e}")
            raise ValkeyDocumentStoreError(f"Failed to retrieve documents by embedding: {e}")

    async def async_search(
        self,
        embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        limit: int = 10,
        with_embedding: bool = True,
    ) -> list[Document]:
        client = await self._get_async_connection()
        assert client is not None
        await self._async_create_index()

        if limit == 0:
            logger.warning(f"Limit cannot be zero: {limit}")
            return []

        # Validate and normalize filters
        if filters:
            _validate_filters(filters)
            filter_query = _normalize_filters(filters)
        else:
            filter_query = "*"

        try:
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
            if limit < 0:
                # No KNN limit - return all matching documents
                query = f"{filter_query}=>[KNN 10000 @vector ${vector_param_name}]"
            else:
                query = f"{filter_query}=>[KNN {limit} @vector ${vector_param_name}]"
            query_options = FtSearchOptions(
                params={vector_param_name: _to_float32_bytes(embedding)}, return_fields=return_fields
            )

            results = await ft.search(client, self._index_name, query, query_options)

            documents = []
            for doc_key, doc_info in results[1].items():
                # doc_key_str = doc_key.decode() if isinstance(doc_key, bytes) else doc_key
                # doc_id = doc_key_str.split(":")[-1]  # Extract ID from key

                # Get payload from doc_info
                payload_data = doc_info.get(b"payload")
                if payload_data:
                    payload_str = payload_data.decode() if isinstance(payload_data, bytes) else payload_data
                    # Unescape the JSON string
                    payload_str = payload_str.replace('\\"', '"')
                    payload = json.loads(payload_str)
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
                doc.embedding = embedding
                # Set similarity score from vector search
                doc.score = similarity_score
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve documents by embedding: {e}")
            raise ValkeyDocumentStoreError(f"Failed to retrieve documents by embedding: {e}")
