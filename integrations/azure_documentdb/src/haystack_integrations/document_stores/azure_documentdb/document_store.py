# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from contextlib import suppress
from typing import Any, Literal

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from pymongo import AsyncMongoClient, InsertOne, MongoClient, ReplaceOne, UpdateOne
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.auth_oidc import OIDCCallback, OIDCCallbackContext, OIDCCallbackResult
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo
from pymongo.errors import BulkWriteError

from haystack_integrations.document_stores.azure_documentdb.filters import _normalize_filters

logger = logging.getLogger(__name__)

_DOCUMENTDB_TOKEN_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"
_DOCUMENTDB_HOST_SUFFIX = "global.mongocluster.cosmos.azure.com"
_DRIVER_INFO = DriverInfo(name="AzureDocumentDBHaystackIntegration")


class AzureIdentityTokenCallback(OIDCCallback):
    """Fetch Microsoft Entra access tokens for PyMongo's OIDC authentication."""

    def __init__(self, credential: TokenCredential) -> None:
        self._credential = credential

    def fetch(self, context: OIDCCallbackContext) -> OIDCCallbackResult:  # noqa: ARG002
        """Fetch an access token for Azure DocumentDB.

        :param context: PyMongo OIDC callback context.
        :returns: The OIDC callback result containing a Microsoft Entra access token.
        """
        token = self._credential.get_token(_DOCUMENTDB_TOKEN_SCOPE)
        return OIDCCallbackResult(access_token=token.token)


class AzureDocumentDBDocumentStore:
    """
    A Haystack document store backed by Azure DocumentDB.

    The default authentication mode uses Microsoft Entra ID through `DefaultAzureCredential`. Supply the Azure
    DocumentDB cluster name with `cluster_name` or the `AZURE_DOCUMENTDB_CLUSTER_NAME` environment variable.

    A connection string can be supplied through `mongo_connection_string` or
    `AZURE_DOCUMENTDB_CONNECTION_STRING` for local development and integration tests. Connection strings can contain
    credentials and aren't recommended for production workloads.

    The collection must already exist. For embedding retrieval, create a `cosmosSearch` vector index by calling
    `create_vector_index` or provisioning it separately. Filtered vector search also requires a regular index for
    every filtered metadata field, such as `meta.category`. Values used with `>`, `>=`, `<`, or `<=` must be numbers
    or ISO-formatted date strings.

    Usage:

    ```python
    from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore

    document_store = AzureDocumentDBDocumentStore(database_name="haystack", collection_name="documents")
    document_store.create_vector_index(dimensions=1536, similarity="COS")
    ```
    """

    def __init__(
        self,
        *,
        database_name: str,
        collection_name: str,
        vector_search_index: str = "haystack_vector_index",
        full_text_search_index: str | None = None,
        cluster_name: str | None = None,
        mongo_connection_string: Secret | None = Secret.from_env_var(  # noqa: B008
            "AZURE_DOCUMENTDB_CONNECTION_STRING", strict=False
        ),
        azure_token_credential: TokenCredential | None = None,
        embedding_field: str = "embedding",
        content_field: str = "content",
    ) -> None:
        """
        Create an Azure DocumentDB document store.

        :param database_name: Name of the existing database.
        :param collection_name: Name of the existing collection.
        :param vector_search_index: Name used when creating the vector index. Azure DocumentDB selects vector indexes
            by path at query time, so this name is not included in vector search queries.
        :param full_text_search_index: Name of an Azure DocumentDB full-text search index. Full-text search is currently
            a gated preview and must be enabled on the cluster before using the full-text retriever.
        :param cluster_name: Azure DocumentDB cluster name. If omitted, `AZURE_DOCUMENTDB_CLUSTER_NAME` is used.
        :param mongo_connection_string: Optional MongoDB connection string intended only for local development and
            integration tests. Microsoft Entra authentication is used when this value is absent.
        :param azure_token_credential: Azure credential used for Microsoft Entra authentication. If omitted,
            `DefaultAzureCredential` is used.
        :param embedding_field: Field containing document embeddings.
        :param content_field: Field containing document content.
        :raises ValueError: If database, collection, or field names are invalid.
        """
        for name, value in (("database_name", database_name), ("collection_name", collection_name)):
            if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                msg = f"Invalid {name}: {value!r}. It can only contain letters, numbers, hyphens, or underscores."
                raise ValueError(msg)
        if not embedding_field or embedding_field.startswith("$"):
            msg = "embedding_field must be a non-empty MongoDB field path and cannot start with '$'."
            raise ValueError(msg)
        if not content_field or content_field.startswith("$"):
            msg = "content_field must be a non-empty MongoDB field path and cannot start with '$'."
            raise ValueError(msg)

        self.database_name = database_name
        self.collection_name = collection_name
        self.vector_search_index = vector_search_index
        self.full_text_search_index = full_text_search_index
        self.cluster_name = cluster_name
        self.mongo_connection_string = mongo_connection_string
        self.azure_token_credential = azure_token_credential
        self.embedding_field = embedding_field
        self.content_field = content_field

        self._connection: MongoClient | None = None
        self._connection_async: AsyncMongoClient | None = None
        self._collection: Collection | None = None
        self._collection_async: AsyncCollection | None = None
        self._credential: TokenCredential | None = None

    def _client_kwargs(self) -> tuple[str, dict[str, Any]]:
        connection_string = self.mongo_connection_string.resolve_value() if self.mongo_connection_string else None
        if connection_string:
            logger.warning(
                "Azure DocumentDB is using connection-string authentication. This fallback is intended only for "
                "local development and integration tests. Use Microsoft Entra managed identity in production."
            )
            return connection_string, {"retryWrites": False, "driver": _DRIVER_INFO}

        cluster_name = self.cluster_name or os.getenv("AZURE_DOCUMENTDB_CLUSTER_NAME")
        if not cluster_name:
            msg = (
                "Azure DocumentDB cluster name is required. Set `cluster_name` or the "
                "AZURE_DOCUMENTDB_CLUSTER_NAME environment variable."
            )
            raise DocumentStoreError(msg)

        if self._credential is None:
            self._credential = self.azure_token_credential or DefaultAzureCredential()
        callback = AzureIdentityTokenCallback(self._credential)
        uri = f"mongodb+srv://{cluster_name}.{_DOCUMENTDB_HOST_SUFFIX}/"
        return uri, {
            "authMechanism": "MONGODB-OIDC",
            "authMechanismProperties": {"OIDC_CALLBACK": callback},
            "retryWrites": False,
            "tls": True,
            "driver": _DRIVER_INFO,
        }

    def _connection_is_valid(self, connection: MongoClient) -> bool:
        try:
            connection.admin.command("ping")
            return True
        except Exception as error:
            logger.error("Connection to Azure DocumentDB failed: {error}", error=error)
            return False

    async def _connection_is_valid_async(self, connection: AsyncMongoClient) -> bool:
        try:
            await connection.admin.command("ping")
            return True
        except Exception as error:
            logger.error("Connection to Azure DocumentDB failed: {error}", error=error)
            return False

    def _ensure_connection_setup(self) -> None:
        if self._connection is None:
            uri, kwargs = self._client_kwargs()
            self._connection = MongoClient(uri, **kwargs)
        if not self._connection_is_valid(self._connection):
            msg = "Connection to Azure DocumentDB failed."
            raise DocumentStoreError(msg)
        database = self._connection[self.database_name]
        if self.collection_name not in database.list_collection_names():
            msg = f"Collection '{self.collection_name}' does not exist in database '{self.database_name}'."
            raise DocumentStoreError(msg)
        self._collection = database[self.collection_name]
        self._collection.create_index("id", unique=True)

    async def _ensure_connection_setup_async(self) -> None:
        if self._connection_async is None:
            uri, kwargs = self._client_kwargs()
            self._connection_async = AsyncMongoClient(uri, **kwargs)
        if not await self._connection_is_valid_async(self._connection_async):
            msg = "Connection to Azure DocumentDB failed."
            raise DocumentStoreError(msg)
        database = self._connection_async[self.database_name]
        if self.collection_name not in await database.list_collection_names():
            msg = f"Collection '{self.collection_name}' does not exist in database '{self.database_name}'."
            raise DocumentStoreError(msg)
        self._collection_async = database[self.collection_name]
        await self._collection_async.create_index("id", unique=True)

    @property
    def connection(self) -> MongoClient | AsyncMongoClient:
        """Return the active Azure DocumentDB client.

        :returns: The synchronous or asynchronous PyMongo client.
        :raises DocumentStoreError: If no connection has been established.
        """
        if self._connection is not None:
            return self._connection
        if self._connection_async is not None:
            return self._connection_async
        msg = "The connection is not established yet."
        raise DocumentStoreError(msg)

    @property
    def collection(self) -> Collection | AsyncCollection:
        """Return the active Azure DocumentDB collection.

        :returns: The synchronous or asynchronous PyMongo collection.
        :raises DocumentStoreError: If no collection has been initialized.
        """
        if self._collection is not None:
            return self._collection
        if self._collection_async is not None:
            return self._collection_async
        msg = "The collection is not established yet."
        raise DocumentStoreError(msg)

    def close(self) -> None:
        """Release synchronous client resources."""
        if self._connection is not None:
            with suppress(Exception):
                self._connection.close()
            self._connection = None
            self._collection = None

    async def close_async(self) -> None:
        """Release asynchronous client resources."""
        if self._connection_async is not None:
            with suppress(Exception):
                await self._connection_async.close()
            self._connection_async = None
            self._collection_async = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this document store to a dictionary.

        :returns: Serialized document-store configuration.
        """
        if self.azure_token_credential is not None:
            logger.warning(
                "AzureDocumentDBDocumentStore was initialized with `azure_token_credential`, which cannot be "
                "serialized and must be provided again after deserialization."
            )
        return default_to_dict(
            self,
            database_name=self.database_name,
            collection_name=self.collection_name,
            vector_search_index=self.vector_search_index,
            full_text_search_index=self.full_text_search_index,
            cluster_name=self.cluster_name,
            mongo_connection_string=self.mongo_connection_string.to_dict() if self.mongo_connection_string else None,
            embedding_field=self.embedding_field,
            content_field=self.content_field,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AzureDocumentDBDocumentStore":
        """Deserialize this document store from a dictionary.

        :param data: Serialized document-store configuration.
        :returns: The deserialized document store.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["mongo_connection_string"])
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """Return the number of documents in the store.

        :returns: The number of documents.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        return self._collection.count_documents({})

    async def count_documents_async(self) -> int:
        """Asynchronously return the number of documents in the store.

        :returns: The number of documents.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        return await self._collection_async.count_documents({})

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """Return documents matching Haystack metadata filters.

        :param filters: Haystack metadata filters. Strings in ordered comparisons must be ISO-formatted dates.
        :returns: Documents matching the filters.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        query = _normalize_filters(filters) if filters else {}
        return [self._mongo_doc_to_haystack_doc(doc) for doc in self._collection.find(query)]

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """Asynchronously return documents matching Haystack metadata filters.

        :param filters: Haystack metadata filters. Strings in ordered comparisons must be ISO-formatted dates.
        :returns: Documents matching the filters.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        query = _normalize_filters(filters) if filters else {}
        documents = await self._collection_async.find(query).to_list(length=None)
        return [self._mongo_doc_to_haystack_doc(doc) for doc in documents]

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """Write documents to Azure DocumentDB using the requested duplicate policy.

        :param documents: Documents to write.
        :param policy: How to handle documents whose IDs already exist.
        :returns: The number of documents written.
        :raises ValueError: If `documents` contains an object that is not a `Document`.
        :raises DuplicateDocumentError: If a duplicate ID is written with `DuplicatePolicy.FAIL`.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        return self._write_documents(self._collection, documents, policy)

    def _write_documents(self, collection: Collection, documents: list[Document], policy: DuplicatePolicy) -> int:
        existing = 0
        if policy == DuplicatePolicy.SKIP:
            existing = collection.count_documents({"id": {"$in": [document.id for document in documents]}})
        operations, written = self._prepare_write_operations(documents, policy, existing)
        if not operations:
            return 0
        try:
            collection.bulk_write(operations)
        except BulkWriteError as error:
            details = error.details.get("writeErrors", []) if error.details else []
            msg = f"Duplicate documents found: {details}"
            raise DuplicateDocumentError(msg) from error
        return written

    def _prepare_write_operations(
        self, documents: list[Document], policy: DuplicatePolicy, existing: int = 0
    ) -> tuple[list[InsertOne[dict[str, Any]] | ReplaceOne[dict[str, Any]] | UpdateOne], int]:
        if any(not isinstance(document, Document) for document in documents):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)
        if not documents:
            return [], 0
        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL
        mongo_documents = [self._haystack_doc_to_mongo_doc(document) for document in documents]
        operations: list[InsertOne[dict[str, Any]] | ReplaceOne[dict[str, Any]] | UpdateOne]
        if policy == DuplicatePolicy.SKIP:
            operations = [UpdateOne({"id": doc["id"]}, {"$setOnInsert": doc}, upsert=True) for doc in mongo_documents]
            written = len(documents) - existing
        elif policy == DuplicatePolicy.FAIL:
            operations = [InsertOne(doc) for doc in mongo_documents]
            written = len(documents)
        else:
            operations = [ReplaceOne({"id": doc["id"]}, doc, upsert=True) for doc in mongo_documents]
            written = len(documents)
        return operations, written

    async def write_documents_async(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """Asynchronously write documents using the requested duplicate policy.

        :param documents: Documents to write.
        :param policy: How to handle documents whose IDs already exist.
        :returns: The number of documents written.
        :raises ValueError: If `documents` contains an object that is not a `Document`.
        :raises DuplicateDocumentError: If a duplicate ID is written with `DuplicatePolicy.FAIL`.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        existing = 0
        if policy == DuplicatePolicy.SKIP:
            existing = await self._collection_async.count_documents(
                {"id": {"$in": [document.id for document in documents]}}
            )
        operations, written = self._prepare_write_operations(documents, policy, existing)
        if not operations:
            return 0
        try:
            await self._collection_async.bulk_write(operations)
        except BulkWriteError as error:
            details = error.details.get("writeErrors", []) if error.details else []
            msg = f"Duplicate documents found: {details}"
            raise DuplicateDocumentError(msg) from error
        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        """Delete documents with matching Haystack IDs.

        :param document_ids: IDs of documents to delete.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        if document_ids:
            self._collection.delete_many({"id": {"$in": document_ids}})

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """Asynchronously delete documents with matching Haystack IDs.

        :param document_ids: IDs of documents to delete.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        if document_ids:
            await self._collection_async.delete_many({"id": {"$in": document_ids}})

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """Delete documents matching filters.

        :param filters: Haystack metadata filters selecting documents to delete.
        :returns: The number of documents deleted.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        result = self._collection.delete_many(_normalize_filters(filters))
        return result.deleted_count

    async def delete_by_filter_async(self, filters: dict[str, Any]) -> int:
        """Asynchronously delete documents matching filters.

        :param filters: Haystack metadata filters selecting documents to delete.
        :returns: The number of documents deleted.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        result = await self._collection_async.delete_many(_normalize_filters(filters))
        return result.deleted_count

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """Update metadata on documents matching filters.

        :param filters: Haystack metadata filters selecting documents to update.
        :param meta: Metadata fields and values to set.
        :returns: The number of documents updated.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        update = {"$set": {f"meta.{key}": value for key, value in meta.items()}}
        result = self._collection.update_many(_normalize_filters(filters), update)
        return result.modified_count

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """Asynchronously update metadata on documents matching filters.

        :param filters: Haystack metadata filters selecting documents to update.
        :param meta: Metadata fields and values to set.
        :returns: The number of documents updated.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        update = {"$set": {f"meta.{key}": value for key, value in meta.items()}}
        result = await self._collection_async.update_many(_normalize_filters(filters), update)
        return result.modified_count

    def delete_all_documents(self, *, recreate_collection: bool = False) -> None:
        """Delete all documents, optionally recreating the collection.

        :param recreate_collection: Drop and recreate the collection instead of deleting documents individually.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        assert self._connection is not None
        if recreate_collection:
            database = self._connection[self.database_name]
            collection_info = database.list_collections(filter={"name": self.collection_name})
            options = next(collection_info, {}).get("options", {})
            indexes = [index for index in self._collection.list_indexes() if index["name"] != "_id_"]
            self._collection.drop()
            database.create_collection(self.collection_name, **options)
            self._collection = database[self.collection_name]
            for index in indexes:
                keys = list(index["key"].items())
                index_options = {key: value for key, value in index.items() if key not in {"key", "v", "ns"}}
                self._collection.create_index(keys, **index_options)
        else:
            self._collection.delete_many({})

    async def delete_all_documents_async(self, *, recreate_collection: bool = False) -> None:
        """Asynchronously delete all documents, optionally recreating the collection.

        :param recreate_collection: Drop and recreate the collection instead of deleting documents individually.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        assert self._connection_async is not None
        if recreate_collection:
            database = self._connection_async[self.database_name]
            collection_info = await database.list_collections(filter={"name": self.collection_name})
            collection_info_list = await collection_info.to_list(length=1)
            options = collection_info_list[0].get("options", {}) if collection_info_list else {}
            indexes_cursor = await self._collection_async.list_indexes()
            indexes = [index for index in await indexes_cursor.to_list(length=None) if index["name"] != "_id_"]
            await self._collection_async.drop()
            await database.create_collection(self.collection_name, **options)
            self._collection_async = database[self.collection_name]
            for index in indexes:
                keys = list(index["key"].items())
                index_options = {key: value for key, value in index.items() if key not in {"key", "v", "ns"}}
                await self._collection_async.create_index(keys, **index_options)
        else:
            await self._collection_async.delete_many({})

    def create_vector_index(
        self,
        *,
        dimensions: int,
        similarity: Literal["COS", "L2", "IP"] = "COS",
        kind: Literal["vector-ivf", "vector-hnsw", "vector-diskann"] = "vector-hnsw",
        **index_options: Any,
    ) -> None:
        """Create the configured Azure DocumentDB `cosmosSearch` vector index.

        :param dimensions: Number of dimensions in each embedding.
        :param similarity: Similarity metric: cosine (`COS`), Euclidean (`L2`), or inner product (`IP`).
        :param kind: Vector index algorithm.
        :param index_options: Algorithm-specific Azure DocumentDB index options.
        :raises ValueError: If `dimensions` is not positive.
        :raises DocumentStoreError: If index creation fails.
        """
        if dimensions <= 0:
            msg = "dimensions must be greater than zero"
            raise ValueError(msg)
        self._ensure_connection_setup()
        assert self._connection is not None
        options = {"kind": kind, "dimensions": dimensions, "similarity": similarity, **index_options}
        command = {
            "createIndexes": self.collection_name,
            "indexes": [
                {
                    "name": self.vector_search_index,
                    "key": {self.embedding_field: "cosmosSearch"},
                    "cosmosSearchOptions": options,
                }
            ],
        }
        try:
            self._connection[self.database_name].command(command)
        except Exception as error:
            msg = f"Failed to create Azure DocumentDB vector index: {error}"
            raise DocumentStoreError(msg) from error

    async def create_vector_index_async(
        self,
        *,
        dimensions: int,
        similarity: Literal["COS", "L2", "IP"] = "COS",
        kind: Literal["vector-ivf", "vector-hnsw", "vector-diskann"] = "vector-hnsw",
        **index_options: Any,
    ) -> None:
        """Asynchronously create the configured `cosmosSearch` vector index.

        :param dimensions: Number of dimensions in each embedding.
        :param similarity: Similarity metric: cosine (`COS`), Euclidean (`L2`), or inner product (`IP`).
        :param kind: Vector index algorithm.
        :param index_options: Algorithm-specific Azure DocumentDB index options.
        :raises ValueError: If `dimensions` is not positive.
        :raises DocumentStoreError: If index creation fails.
        """
        if dimensions <= 0:
            msg = "dimensions must be greater than zero"
            raise ValueError(msg)
        await self._ensure_connection_setup_async()
        assert self._connection_async is not None
        options = {"kind": kind, "dimensions": dimensions, "similarity": similarity, **index_options}
        command = {
            "createIndexes": self.collection_name,
            "indexes": [
                {
                    "name": self.vector_search_index,
                    "key": {self.embedding_field: "cosmosSearch"},
                    "cosmosSearchOptions": options,
                }
            ],
        }
        try:
            await self._connection_async[self.database_name].command(command)
        except Exception as error:
            msg = f"Failed to create Azure DocumentDB vector index: {error}"
            raise DocumentStoreError(msg) from error

    def _embedding_pipeline(
        self, query_embedding: list[float], filters: dict[str, Any] | None, top_k: int
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            msg = "Query embedding must not be empty"
            raise ValueError(msg)
        if top_k <= 0:
            msg = "top_k must be greater than zero"
            raise ValueError(msg)
        cosmos_search: dict[str, Any] = {
            "vector": query_embedding,
            "path": self.embedding_field,
            "k": top_k,
        }
        if filters:
            cosmos_search["filter"] = _normalize_filters(filters)
        return [
            {"$search": {"cosmosSearch": cosmos_search, "returnStoredSource": True}},
            {"$project": {"document": "$$ROOT", "score": {"$meta": "searchScore"}}},
        ]

    def _embedding_retrieval(
        self, query_embedding: list[float], filters: dict[str, Any] | None = None, top_k: int = 10
    ) -> list[Document]:
        """Retrieve documents by Azure DocumentDB vector similarity."""
        pipeline = self._embedding_pipeline(query_embedding, filters, top_k)
        self._ensure_connection_setup()
        assert self._collection is not None
        try:
            results = list(self._collection.aggregate(pipeline))
        except Exception as error:
            msg = f"Vector retrieval from Azure DocumentDB failed: {error}"
            raise DocumentStoreError(msg) from error
        return [self._search_result_to_haystack_doc(result) for result in results]

    async def _embedding_retrieval_async(
        self, query_embedding: list[float], filters: dict[str, Any] | None = None, top_k: int = 10
    ) -> list[Document]:
        """Asynchronously retrieve documents by Azure DocumentDB vector similarity."""
        pipeline = self._embedding_pipeline(query_embedding, filters, top_k)
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        try:
            cursor = await self._collection_async.aggregate(pipeline)
            results = await cursor.to_list(length=None)
        except Exception as error:
            msg = f"Vector retrieval from Azure DocumentDB failed: {error}"
            raise DocumentStoreError(msg) from error
        return [self._search_result_to_haystack_doc(result) for result in results]

    def _full_text_pipeline(
        self, query: str | list[str], fuzzy: dict[str, int] | None, filters: dict[str, Any] | None, top_k: int
    ) -> list[dict[str, Any]]:
        if not query:
            msg = "Argument query must not be empty."
            raise ValueError(msg)
        if not self.full_text_search_index:
            msg = "full_text_search_index must be configured to use full-text retrieval."
            raise ValueError(msg)
        if top_k <= 0:
            msg = "top_k must be greater than zero"
            raise ValueError(msg)
        text_search: dict[str, Any] = {"query": query, "path": self.content_field}
        if fuzzy:
            text_search["fuzzy"] = fuzzy
        pipeline: list[dict[str, Any]] = [
            {"$search": {"index": self.full_text_search_index, "text": text_search}},
        ]
        if filters:
            pipeline.append({"$match": _normalize_filters(filters)})
        pipeline.extend(
            [
                {"$limit": top_k},
                {"$addFields": {"score": {"$meta": "searchScore"}}},
                {"$project": {"_id": 0}},
            ]
        )
        return pipeline

    def _full_text_retrieval(
        self,
        query: str | list[str],
        fuzzy: dict[str, int] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """Retrieve documents with Azure DocumentDB BM25 full-text search (gated preview)."""
        pipeline = self._full_text_pipeline(query, fuzzy, filters, top_k)
        self._ensure_connection_setup()
        assert self._collection is not None
        try:
            results = list(self._collection.aggregate(pipeline))
        except Exception as error:
            msg = f"Full-text retrieval from Azure DocumentDB failed: {error}"
            raise DocumentStoreError(msg) from error
        return [self._mongo_doc_to_haystack_doc(result) for result in results]

    async def _full_text_retrieval_async(
        self,
        query: str | list[str],
        fuzzy: dict[str, int] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """Asynchronously retrieve documents with Azure DocumentDB BM25 search (gated preview)."""
        pipeline = self._full_text_pipeline(query, fuzzy, filters, top_k)
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        try:
            cursor = await self._collection_async.aggregate(pipeline)
            results = await cursor.to_list(length=None)
        except Exception as error:
            msg = f"Full-text retrieval from Azure DocumentDB failed: {error}"
            raise DocumentStoreError(msg) from error
        return [self._mongo_doc_to_haystack_doc(result) for result in results]

    def _search_result_to_haystack_doc(self, result: dict[str, Any]) -> Document:
        document = dict(result.get("document", result))
        if "score" in result:
            document["score"] = result["score"]
        return self._mongo_doc_to_haystack_doc(document)

    def _mongo_doc_to_haystack_doc(self, mongo_doc: dict[str, Any]) -> Document:
        document = dict(mongo_doc)
        document.pop("_id", None)
        if self.content_field != "content":
            document["content"] = document.pop(self.content_field, None)
        if self.embedding_field != "embedding":
            document["embedding"] = document.pop(self.embedding_field, None)
        return Document.from_dict(document)

    def _haystack_doc_to_mongo_doc(self, haystack_doc: Document) -> dict[str, Any]:
        document = haystack_doc.to_dict(flatten=False)
        if self.content_field != "content":
            document[self.content_field] = document.pop("content", None)
        if self.embedding_field != "embedding":
            document[self.embedding_field] = document.pop("embedding", None)
        sparse_embedding = document.pop("sparse_embedding", None)
        if sparse_embedding:
            logger.warning(
                "Document {id} has a sparse embedding, but Azure DocumentDB integration does not support sparse "
                "embeddings. The field will be ignored.",
                id=haystack_doc.id,
            )
        document.pop("_id", None)
        return document
