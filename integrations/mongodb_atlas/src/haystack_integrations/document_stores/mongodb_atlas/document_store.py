# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from pymongo import AsyncMongoClient, InsertOne, MongoClient, ReplaceOne, UpdateOne
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo
from pymongo.errors import BulkWriteError

from haystack_integrations.document_stores.mongodb_atlas.filters import _normalize_filters

logger = logging.getLogger(__name__)


class MongoDBAtlasDocumentStore:
    """
    A MongoDBAtlasDocumentStore implementation that uses the
    [MongoDB Atlas](https://www.mongodb.com/atlas/database) service that is easy to deploy, operate, and scale.

    To connect to MongoDB Atlas, you need to provide a connection string in the format:
    `"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"`.

    This connection string can be obtained on the MongoDB Atlas Dashboard by clicking on the `CONNECT` button, selecting
    Python as the driver, and copying the connection string. The connection string can be provided as an environment
    variable `MONGO_CONNECTION_STRING` or directly as a parameter to the `MongoDBAtlasDocumentStore` constructor.

    After providing the connection string, you'll need to specify the `database_name` and `collection_name` to use.
    Most likely that you'll create these via the MongoDB Atlas web UI but one can also create them via the MongoDB
    Python driver. Creating databases and collections is beyond the scope of MongoDBAtlasDocumentStore. The primary
    purpose of this document store is to read and write documents to an existing collection.

    Users must provide both a `vector_search_index` for vector search operations and a `full_text_search_index`
    for full-text search operations. The `vector_search_index` supports a chosen metric
    (e.g., cosine, dot product, or Euclidean), while the `full_text_search_index` enables efficient text-based searches.
    Both indexes can be created through the Atlas web UI.

    For more details on MongoDB Atlas, see the official
    MongoDB Atlas [documentation](https://www.mongodb.com/docs/atlas/getting-started/).

    Usage example:
    ```python
    from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

    store = MongoDBAtlasDocumentStore(database_name="your_existing_db",
                                      collection_name="your_existing_collection",
                                      vector_search_index="your_existing_index",
                                      full_text_search_index="your_existing_index")
    print(store.count_documents())
    ```
    """

    def __init__(
        self,
        *,
        mongo_connection_string: Secret = Secret.from_env_var("MONGO_CONNECTION_STRING"),  # noqa: B008
        database_name: str,
        collection_name: str,
        vector_search_index: str,
        full_text_search_index: str,
        embedding_field: str = "embedding",
        content_field: str = "content",
    ):
        """
        Creates a new MongoDBAtlasDocumentStore instance.

        :param mongo_connection_string: MongoDB Atlas connection string in the format:
            `"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"`.
            This can be obtained on the MongoDB Atlas Dashboard by clicking on the `CONNECT` button.
            This value will be read automatically from the env var "MONGO_CONNECTION_STRING".
        :param database_name: Name of the database to use.
        :param collection_name: Name of the collection to use. To use this document store for embedding retrieval,
            this collection needs to have a vector search index set up on the `embedding` field.
        :param vector_search_index: The name of the vector search index to use for vector search operations.
            Create a vector_search_index in the Atlas web UI and specify the init params of MongoDBAtlasDocumentStore. \
            For more details refer to MongoDB
            Atlas [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#std-label-avs-create-index).
        :param full_text_search_index: The name of the search index to use for full-text search operations.
            Create a full_text_search_index in the Atlas web UI and specify the init params of
            MongoDBAtlasDocumentStore. For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/create-index/).
        :param embedding_field: The name of the field containing document embeddings. Default is "embedding".
        :param content_field: The name of the field containing the document content. Default is "content".
            This field is allows defining which field to load into the Haystack Document object as content.
            It can be particularly useful when integrating with an existing collection for retrieval. We discourage
            using this parameter when working with collections created by Haystack.
        :raises ValueError: If the collection name contains invalid characters.
        """
        if collection_name and not bool(re.match(r"^[a-zA-Z0-9\-_]+$", collection_name)):
            msg = f'Invalid collection name: "{collection_name}". It can only contain letters, numbers, -, or _.'
            raise ValueError(msg)

        self.mongo_connection_string = mongo_connection_string

        self.database_name = database_name
        self.collection_name = collection_name
        self.vector_search_index = vector_search_index
        self.full_text_search_index = full_text_search_index
        self.embedding_field = embedding_field
        self.content_field = content_field
        self._connection: Optional[MongoClient] = None
        self._connection_async: Optional[AsyncMongoClient] = None
        self._collection: Optional[Collection] = None
        self._collection_async: Optional[AsyncCollection] = None

    def __del__(self) -> None:
        """
        Destructor method to close MongoDB connections when the instance is destroyed.
        """
        if self._connection:
            self._connection.close()

    @property
    def connection(self) -> Union[AsyncMongoClient, MongoClient]:
        if self._connection:
            return self._connection
        if self._connection_async:
            return self._connection_async
        msg = "The connection is not established yet."
        raise DocumentStoreError(msg)

    @property
    def collection(self) -> Union[AsyncCollection, Collection]:
        if self._collection:
            return self._collection
        if self._collection_async:
            return self._collection_async
        msg = "The collection is not established yet."
        raise DocumentStoreError(msg)

    def _connection_is_valid(self, connection: MongoClient) -> bool:
        """
        Checks if the connection to MongoDB Atlas is valid.

        :returns: True if the connection is valid, False otherwise.
        """
        try:
            connection.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"Connection to MongoDB Atlas failed: {e}")
            return False

    async def _connection_is_valid_async(self, connection: AsyncMongoClient) -> bool:
        """
        Asynchronously checks if the connection to MongoDB Atlas is valid.

        :returns: True if the connection is valid, False otherwise.
        """
        try:
            await connection.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"Connection to MongoDB Atlas failed: {e}")
            return False

    def _collection_exists(self, connection: MongoClient, database_name: str, collection_name: str) -> bool:
        """
        Checks if the collection exists in the MongoDB Atlas database.

        :returns: True if the collection exists, False otherwise.
        """
        database = connection[database_name]
        return collection_name in database.list_collection_names()

    async def _collection_exists_async(
        self, connection: AsyncMongoClient, database_name: str, collection_name: str
    ) -> bool:
        """
        Asynchronously checks if the collection exists in the MongoDB Atlas database.

        :returns: True if the collection exists, False otherwise.
        """
        database = connection[database_name]
        return collection_name in await database.list_collection_names()

    def _ensure_connection_setup(self) -> None:
        """
        Ensures that the connection to MongoDB Atlas is set up and the collection exists.

        :raises DocumentStoreError: If the connection to MongoDB Atlas fails.
        :raises DocumentStoreError: If the collection does not exist.
        """
        if not self._connection:
            self._connection = MongoClient(
                self.mongo_connection_string.resolve_value(), driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
            )

        if not self._connection_is_valid(self._connection):
            msg = "Connection to MongoDB Atlas failed."
            raise DocumentStoreError(msg)

        if not self._collection_exists(self._connection, self.database_name, self.collection_name):
            msg = f"Collection '{self.collection_name}' does not exist in database '{self.database_name}'."
            raise DocumentStoreError(msg)

        if self._collection is None:
            database = self._connection[self.database_name]
            self._collection = database[self.collection_name]

    async def _ensure_connection_setup_async(self) -> None:
        """
        Asynchronously ensures that the connection to MongoDB Atlas is set up and the collection exists.

        :raises DocumentStoreError: If the connection to MongoDB Atlas fails.
        :raises DocumentStoreError: If the collection does not exist.
        """
        if not self._connection_async:
            self._connection_async = AsyncMongoClient(
                self.mongo_connection_string.resolve_value(), driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
            )

        if not await self._connection_is_valid_async(self._connection_async):
            msg = "Connection to MongoDB Atlas failed."
            raise DocumentStoreError(msg)

        if not await self._collection_exists_async(self._connection_async, self.database_name, self.collection_name):
            msg = f"Collection '{self.collection_name}' does not exist in database '{self.database_name}'."
            raise DocumentStoreError(msg)

        if self._collection_async is None:
            database = self._connection_async[self.database_name]
            self._collection_async = database[self.collection_name]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            mongo_connection_string=self.mongo_connection_string.to_dict(),
            database_name=self.database_name,
            collection_name=self.collection_name,
            vector_search_index=self.vector_search_index,
            full_text_search_index=self.full_text_search_index,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
              Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["mongo_connection_string"])
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: The number of documents in the document store.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        return self._collection.count_documents({})

    async def count_documents_async(self) -> int:
        """
        Asynchronously returns how many documents are present in the document store.

        :returns: The number of documents in the document store.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        return await self._collection_async.count_documents({})

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the Haystack [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: The filters to apply. It returns only the documents that match the filters.
        :returns: A list of Documents that match the given filters.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        filters = _normalize_filters(filters) if filters else None
        documents = list(self._collection.find(filters))
        return [self._mongo_doc_to_haystack_doc(doc) for doc in documents]

    async def filter_documents_async(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the Haystack [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: The filters to apply. It returns only the documents that match the filters.
        :returns: A list of Documents that match the given filters.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        filters = _normalize_filters(filters) if filters else None
        documents = await self._collection_async.find(filters).to_list()
        return [self._mongo_doc_to_haystack_doc(doc) for doc in documents]

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents into the MongoDB Atlas collection.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same ID already exists in the document store
             and the policy is set to DuplicatePolicy.FAIL (or not specified).
        :raises ValueError: If the documents are not of type Document.
        :returns: The number of documents written to the document store.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        mongo_documents = [self._haystack_doc_to_mongo_doc(doc) for doc in documents]
        operations: List[Union[UpdateOne, InsertOne, ReplaceOne]]
        written_docs = len(documents)

        if policy == DuplicatePolicy.SKIP:
            operations = [UpdateOne({"id": doc["id"]}, {"$setOnInsert": doc}, upsert=True) for doc in mongo_documents]
            existing_documents = self._collection.count_documents({"id": {"$in": [doc.id for doc in documents]}})
            written_docs -= existing_documents
        elif policy == DuplicatePolicy.FAIL:
            operations = [InsertOne(doc) for doc in mongo_documents]
        else:
            operations = [ReplaceOne({"id": doc["id"]}, upsert=True, replacement=doc) for doc in mongo_documents]

        try:
            self._collection.bulk_write(operations)
        except BulkWriteError as e:
            msg = f"Duplicate documents found: {e.details['writeErrors']}"
            raise DuplicateDocumentError(msg) from e

        return written_docs

    async def write_documents_async(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Writes documents into the MongoDB Atlas collection.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same ID already exists in the document store
             and the policy is set to DuplicatePolicy.FAIL (or not specified).
        :raises ValueError: If the documents are not of type Document.
        :returns: The number of documents written to the document store.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        mongo_documents = [self._haystack_doc_to_mongo_doc(doc) for doc in documents]

        operations: List[Union[UpdateOne, InsertOne, ReplaceOne]]
        written_docs = len(documents)

        if policy == DuplicatePolicy.SKIP:
            operations = [UpdateOne({"id": doc["id"]}, {"$setOnInsert": doc}, upsert=True) for doc in mongo_documents]
            existing_documents = await self._collection_async.count_documents(
                {"id": {"$in": [doc.id for doc in documents]}}
            )
            written_docs -= existing_documents
        elif policy == DuplicatePolicy.FAIL:
            operations = [InsertOne(doc) for doc in mongo_documents]
        else:
            operations = [ReplaceOne({"id": doc["id"]}, upsert=True, replacement=doc) for doc in mongo_documents]

        try:
            await self._collection_async.bulk_write(operations)
        except BulkWriteError as e:
            msg = f"Duplicate documents found: {e.details['writeErrors']}"
            raise DuplicateDocumentError(msg) from e

        return written_docs

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        if not document_ids:
            return
        self._collection.delete_many(filter={"id": {"$in": document_ids}})

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        """
        Asynchronously deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        if not document_ids:
            return
        await self._collection_async.delete_many(filter={"id": {"$in": document_ids}})

    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """
        Deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        self._ensure_connection_setup()
        assert self._collection is not None

        try:
            normalized_filters = _normalize_filters(filters)
            result = self._collection.delete_many(filter=normalized_filters)
            deleted_count = result.deleted_count
            logger.info(
                "Deleted {n_docs} documents from collection '{collection}' using filters.",
                n_docs=deleted_count,
                collection=self.collection_name,
            )
            return deleted_count
        except Exception as e:
            msg = f"Failed to delete documents by filter from MongoDB Atlas: {e!s}"
            raise DocumentStoreError(msg) from e

    async def delete_by_filter_async(self, filters: Dict[str, Any]) -> int:
        """
        Asynchronously deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None

        try:
            normalized_filters = _normalize_filters(filters)
            result = await self._collection_async.delete_many(filter=normalized_filters)
            deleted_count = result.deleted_count
            logger.info(
                "Deleted {n_docs} documents from collection '{collection}' using filters.",
                n_docs=deleted_count,
                collection=self.collection_name,
            )
            return deleted_count
        except Exception as e:
            msg = f"Failed to delete documents by filter from MongoDB Atlas: {e!s}"
            raise DocumentStoreError(msg) from e

    def update_by_filter(self, filters: Dict[str, Any], meta: Dict[str, Any]) -> int:
        """
        Updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
        :returns: The number of documents updated.
        """
        self._ensure_connection_setup()
        assert self._collection is not None

        try:
            normalized_filters = _normalize_filters(filters)
            # Build update operation to set metadata fields
            # MongoDB stores documents with flatten=False, so metadata is in the "meta" field
            update_fields = {f"meta.{key}": value for key, value in meta.items()}
            result = self._collection.update_many(filter=normalized_filters, update={"$set": update_fields})
            updated_count = result.modified_count
            logger.info(
                "Updated {n_docs} documents in collection '{collection}' using filters.",
                n_docs=updated_count,
                collection=self.collection_name,
            )
            return updated_count
        except Exception as e:
            msg = f"Failed to update documents by filter in MongoDB Atlas: {e!s}"
            raise DocumentStoreError(msg) from e

    async def update_by_filter_async(self, filters: Dict[str, Any], meta: Dict[str, Any]) -> int:
        """
        Asynchronously updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
        :returns: The number of documents updated.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None

        try:
            normalized_filters = _normalize_filters(filters)
            # Build update operation to set metadata fields
            # MongoDB stores documents with flatten=False, so metadata is in the "meta" field
            update_fields = {f"meta.{key}": value for key, value in meta.items()}
            result = await self._collection_async.update_many(filter=normalized_filters, update={"$set": update_fields})
            updated_count = result.modified_count
            logger.info(
                "Updated {n_docs} documents in collection '{collection}' using filters.",
                n_docs=updated_count,
                collection=self.collection_name,
            )
            return updated_count
        except Exception as e:
            msg = f"Failed to update documents by filter in MongoDB Atlas: {e!s}"
            raise DocumentStoreError(msg) from e

    def delete_all_documents(self, *, recreate_collection: bool = False) -> None:
        """
        Deletes all documents in the document store.

        :param recreate_collection: If True, the collection will be dropped and recreated with the original
            configuration and indexes. If False, all documents will be deleted while preserving the collection.
            Recreating the collection is faster for very large collections.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        assert self._connection is not None

        try:
            if recreate_collection:
                database = self._connection[self.database_name]

                # Save collection configuration
                collection_info = database.list_collections(filter={"name": self.collection_name})
                config = next(collection_info, {}).get("options", {})

                # Save index definitions (excluding default _id index)
                indexes = list(self._collection.list_indexes())
                custom_indexes = [idx for idx in indexes if idx["name"] != "_id_"]

                # Drop and recreate collection
                self._collection.drop()
                database.create_collection(self.collection_name, **config)

                # Recreate indexes
                for idx in custom_indexes:
                    keys = list(idx["key"].items())
                    index_options = {k: v for k, v in idx.items() if k not in ["key", "v", "ns"]}
                    self._collection.create_index(keys, **index_options)

                logger.info(
                    "Collection '{collection}' recreated with original configuration.",
                    collection=self.collection_name,
                )
            else:
                # Delete all documents without recreating collection
                result = self._collection.delete_many({})
                logger.info(
                    "Deleted {n_docs} documents from collection '{collection}'.",
                    n_docs=result.deleted_count,
                    collection=self.collection_name,
                )
        except Exception as e:
            msg = f"Failed to delete all documents from MongoDB Atlas: {e!s}"
            raise DocumentStoreError(msg) from e

    async def delete_all_documents_async(self, *, recreate_collection: bool = False) -> None:
        """
        Asynchronously deletes all documents in the document store.

        :param recreate_collection: If True, the collection will be dropped and recreated with the original
            configuration and indexes. If False, all documents will be deleted while preserving the collection.
            Recreating the collection is faster for very large collections.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        assert self._connection_async is not None

        try:
            if recreate_collection:
                database = self._connection_async[self.database_name]

                # Save collection configuration
                collection_info_cursor = await database.list_collections(filter={"name": self.collection_name})
                config_list = await collection_info_cursor.to_list(length=1)
                config = config_list[0].get("options", {}) if config_list else {}

                # Save index definitions (excluding default _id index)
                indexes_cursor = await self._collection_async.list_indexes()
                indexes = await indexes_cursor.to_list(length=None)
                custom_indexes = [idx for idx in indexes if idx["name"] != "_id_"]

                # Drop and recreate collection
                await self._collection_async.drop()
                await database.create_collection(self.collection_name, **config)

                # Recreate indexes
                for idx in custom_indexes:
                    keys = list(idx["key"].items())
                    index_options = {k: v for k, v in idx.items() if k not in ["key", "v", "ns"]}
                    await self._collection_async.create_index(keys, **index_options)

                logger.info(
                    "Collection '{collection}' recreated with original configuration.",
                    collection=self.collection_name,
                )
            else:
                # Delete all documents without recreating collection
                result = await self._collection_async.delete_many({})
                logger.info(
                    "Deleted {n_docs} documents from collection '{collection}'.",
                    n_docs=result.deleted_count,
                    collection=self.collection_name,
                )
        except Exception as e:
            msg = f"Failed to delete all documents from MongoDB Atlas: {e!s}"
            raise DocumentStoreError(msg) from e

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Find the documents that are most similar to the provided `query_embedding` by using a vector similarity metric.

        :param query_embedding: Embedding of the query
        :param filters: Optional filters.
        :param top_k: How many documents to return.
        :returns: A list of Documents that are most similar to the given `query_embedding`
        :raises ValueError: If `query_embedding` is empty.
        :raises DocumentStoreError: If the retrieval of documents from MongoDB Atlas fails.
        """
        self._ensure_connection_setup()
        assert self._collection is not None
        if not query_embedding:
            msg = "Query embedding must not be empty"
            raise ValueError(msg)

        filters = _normalize_filters(filters) if filters else {}

        pipeline: List[Dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.vector_search_index,
                    "path": self.embedding_field,
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k,
                    "filter": filters,
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"_id": 0}},
        ]
        try:
            documents = list(self._collection.aggregate(pipeline))
        except Exception as e:
            msg = f"Retrieval of documents from MongoDB Atlas failed: {e}"
            if filters:
                msg += (
                    "\nMake sure that the fields used in the filters are included "
                    "in the `vector_search_index` configuration"
                )
            raise DocumentStoreError(msg) from e

        documents = [self._mongo_doc_to_haystack_doc(doc) for doc in documents]
        return documents

    async def _embedding_retrieval_async(
        self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: int = 10
    ) -> List[Document]:
        """
        Asynchronously find the documents that are most similar to the provided `query_embedding` by using a vector
        similarity metric.

        :param query_embedding: Embedding of the query
        :param filters: Optional filters.
        :param top_k: How many documents to return.
        :returns: A list of Documents that are most similar to the given `query_embedding`
        :raises ValueError: If `query_embedding` is empty.
        :raises DocumentStoreError: If the retrieval of documents from MongoDB Atlas fails.
        """
        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        if not query_embedding:
            msg = "Query embedding must not be empty"
            raise ValueError(msg)

        filters = _normalize_filters(filters) if filters else {}

        pipeline: List[Dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.vector_search_index,
                    "path": self.embedding_field,
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k,
                    "filter": filters,
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"_id": 0}},
        ]
        try:
            cursor = await self._collection_async.aggregate(pipeline)
            documents = await cursor.to_list(length=None)
        except Exception as e:
            msg = f"Retrieval of documents from MongoDB Atlas failed: {e}"
            if filters:
                msg += (
                    "\nMake sure that the fields used in the filters are included "
                    "in the `vector_search_index` configuration"
                )
            raise DocumentStoreError(msg) from e

        documents = [self._mongo_doc_to_haystack_doc(doc) for doc in documents]
        return documents

    def _fulltext_retrieval(
        self,
        query: Union[str, List[str]],
        fuzzy: Optional[Dict[str, int]] = None,
        match_criteria: Optional[Literal["any", "all"]] = None,
        score: Optional[Dict[str, Dict]] = None,
        synonyms: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Retrieve documents similar to the provided `query` using a full-text search.

        :param query: The query string or a list of query strings to search for.
            If the query contains multiple terms, Atlas Search evaluates each term separately for matches.
        :param fuzzy: Enables finding strings similar to the search term(s).
            Note, `fuzzy` cannot be used with `synonyms`. Configurable options include `maxEdits`, `prefixLength`,
            and `maxExpansions`. For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param match_criteria: Defines how terms in the query are matched. Supported options are `"any"` and `"all"`.
            For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param score: Specifies the scoring method for matching results. Supported options include `boost`, `constant`,
            and `function`. For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param synonyms: The name of the synonym mapping definition in the index. This value cannot be an empty string.
            Note, `synonyms` can not be used with `fuzzy`.
        :param filters: Optional filters.
        :param top_k: How many documents to return.
        :returns: A list of Documents that are most similar to the given `query`
        :raises ValueError: If `query` or `synonyms` is empty.
        :raises ValueError: If `synonyms` and `fuzzy` are used together.
        :raises DocumentStoreError: If the retrieval of documents from MongoDB Atlas fails.
        """
        # Validate user input according to MongoDB Atlas Search requirements
        if not query:
            msg = "Argument query must not be empty."
            raise ValueError(msg)

        if isinstance(synonyms, str) and not synonyms:
            msg = "Argument synonyms cannot be an empty string."
            raise ValueError(msg)

        if synonyms and fuzzy:
            msg = "Cannot use both synonyms and fuzzy search together."
            raise ValueError(msg)

        if synonyms and not match_criteria:
            logger.warning(
                "Specify matchCriteria when using synonyms. "
                "Atlas Search matches terms in exact order by default, which may change in future versions."
            )

        filters = _normalize_filters(filters) if filters else {}

        # Build the text search options
        text_search: Dict[str, Any] = {"path": self.content_field or "content", "query": query}
        if match_criteria:
            text_search["matchCriteria"] = match_criteria
        if synonyms:
            text_search["synonyms"] = synonyms
        if fuzzy:
            text_search["fuzzy"] = fuzzy
        if score:
            text_search["score"] = score

        # Define the pipeline for MongoDB aggregation
        pipeline: List[Dict[str, Any]] = [
            {
                "$search": {
                    "index": self.full_text_search_index,
                    "compound": {"must": [{"text": text_search}]},
                }
            },
            # TODO: Use compound filter. See: (https://www.mongodb.com/docs/atlas/atlas-search/performance/query-performance/#avoid--match-after--search)
            {"$match": filters},
            {"$limit": top_k},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$project": {"_id": 0}},
        ]

        self._ensure_connection_setup()
        assert self._collection is not None
        try:
            documents = list(self._collection.aggregate(pipeline))
        except Exception as e:
            error_msg = f"Failed to retrieve documents from MongoDB Atlas: {e}"
            if filters:
                error_msg += "\nEnsure fields in filters are included in the `full_text_search_index` configuration."
            raise DocumentStoreError(error_msg) from e

        return [self._mongo_doc_to_haystack_doc(doc) for doc in documents]

    async def _fulltext_retrieval_async(
        self,
        query: Union[str, List[str]],
        fuzzy: Optional[Dict[str, int]] = None,
        match_criteria: Optional[Literal["any", "all"]] = None,
        score: Optional[Dict[str, Dict]] = None,
        synonyms: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Asynchronously retrieve documents similar to the provided `query` using a full-text search asynchronously.

        :param query: The query string or a list of query strings to search for.
            If the query contains multiple terms, Atlas Search evaluates each term separately for matches.
        :param fuzzy: Enables finding strings similar to the search term(s).
            Note, `fuzzy` cannot be used with `synonyms`. Configurable options include `maxEdits`, `prefixLength`,
            and `maxExpansions`. For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param match_criteria: Defines how terms in the query are matched. Supported options are `"any"` and `"all"`.
            For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param score: Specifies the scoring method for matching results. Supported options include `boost`, `constant`,
            and `function`. For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param synonyms: The name of the synonym mapping definition in the index. This value cannot be an empty string.
            Note, `synonyms` can not be used with `fuzzy`.
        :param filters: Optional filters.
        :param top_k: How many documents to return.
        :returns: A list of Documents that are most similar to the given `query`
        :raises ValueError: If `query` or `synonyms` is empty.
        :raises ValueError: If `synonyms` and `fuzzy` are used together.
        :raises DocumentStoreError: If the retrieval of documents from MongoDB Atlas fails.
        """
        # Validate user input according to MongoDB Atlas Search requirements
        if not query:
            msg = "Argument query must not be empty."
            raise ValueError(msg)

        if isinstance(synonyms, str) and not synonyms:
            msg = "Argument synonyms cannot be an empty string."
            raise ValueError(msg)

        if synonyms and fuzzy:
            msg = "Cannot use both synonyms and fuzzy search together."
            raise ValueError(msg)

        if synonyms and not match_criteria:
            logger.warning(
                "Specify matchCriteria when using synonyms. "
                "Atlas Search matches terms in exact order by default, which may change in future versions."
            )

        filters = _normalize_filters(filters) if filters else {}

        # Build the text search options
        text_search: Dict[str, Any] = {"path": self.content_field or "content", "query": query}
        if match_criteria:
            text_search["matchCriteria"] = match_criteria
        if synonyms:
            text_search["synonyms"] = synonyms
        if fuzzy:
            text_search["fuzzy"] = fuzzy
        if score:
            text_search["score"] = score

        # Define the pipeline for MongoDB aggregation
        pipeline: List[Dict[str, Any]] = [
            {
                "$search": {
                    "index": self.full_text_search_index,
                    "compound": {"must": [{"text": text_search}]},
                }
            },
            # TODO: Use compound filter. See: (https://www.mongodb.com/docs/atlas/atlas-search/performance/query-performance/#avoid--match-after--search)
            {"$match": filters},
            {"$limit": top_k},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$project": {"_id": 0}},
        ]

        await self._ensure_connection_setup_async()
        assert self._collection_async is not None
        try:
            cursor = await self._collection_async.aggregate(pipeline)
            documents = await cursor.to_list(length=None)
        except Exception as e:
            error_msg = f"Failed to retrieve documents from MongoDB Atlas: {e}"
            if filters:
                error_msg += "\nEnsure fields in filters are included in the `full_text_search_index` configuration."
            raise DocumentStoreError(error_msg) from e

        return [self._mongo_doc_to_haystack_doc(doc) for doc in documents]

    def _mongo_doc_to_haystack_doc(self, mongo_doc: Dict[str, Any]) -> Document:
        """
        Converts the dictionary coming out of MongoDB into a Haystack document

        :param mongo_doc: A dictionary representing a document as stored in MongoDB
        :returns: A Haystack Document object
        """
        mongo_doc.pop("_id", None)  # MongoDB's internal id doesn't belong into a Haystack document, so we remove it.
        if self.content_field != "content":
            mongo_doc["content"] = mongo_doc.pop(self.content_field, None)
        if self.embedding_field != "embedding":
            mongo_doc["embedding"] = mongo_doc.pop(self.embedding_field, None)
        return Document.from_dict(mongo_doc)

    def _haystack_doc_to_mongo_doc(self, haystack_doc: Document) -> Dict[str, Any]:
        """
        Parses a Haystack Document to a MongoDB document.

        :param haystack_doc: the Haystack Document to convert
        :returns: the MongoDB document in a dictionary representation
        """
        mongo_doc = haystack_doc.to_dict(flatten=False)
        if self.content_field != "content":
            mongo_doc[self.content_field] = mongo_doc.pop("content", None)
        if self.embedding_field != "embedding":
            mongo_doc[self.embedding_field] = mongo_doc.pop("embedding", None)
        if "sparse_embedding" in mongo_doc:
            sparse_embedding = mongo_doc.pop("sparse_embedding", None)
            if sparse_embedding:
                logger.warning(
                    "Document {id} has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in MongoDB Atlas is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    id=haystack_doc.id,
                )
        mongo_doc.pop("_id", None)
        return mongo_doc
