# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from asyncio import AbstractEventLoop, Lock, get_running_loop
from collections.abc import Generator
from typing import Any
from warnings import warn

from astrapy import AsyncCollection, Collection
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError, MissingDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .astra_client import (
    _collection_definition,
    _collection_indexing_warning,
    _data_api_client,
    _find_collection,
    _find_kwargs,
    _to_documents,
)
from .errors import AstraDocumentStoreFilterError
from .filters import _convert_filters

logger = logging.getLogger(__name__)


MAX_BATCH_SIZE = 20


def _batches(input_list: list[Any], batch_size: int) -> Generator[list[Any], None, None]:
    input_length = len(input_list)
    for ndx in range(0, input_length, batch_size):
        yield input_list[ndx : min(ndx + batch_size, input_length)]


class AstraDocumentStore:
    """
    An AstraDocumentStore document store for Haystack.

    Example Usage:
    ```python
    from haystack_integrations.document_stores.astra import AstraDocumentStore

    document_store = AstraDocumentStore(
        api_endpoint=api_endpoint,
        token=token,
        collection_name=collection_name,
        duplicates_policy=DuplicatePolicy.SKIP,
        embedding_dim=384,
    )
    ```
    """

    def __init__(
        self,
        api_endpoint: Secret = Secret.from_env_var("ASTRA_DB_API_ENDPOINT"),  # noqa: B008
        token: Secret = Secret.from_env_var("ASTRA_DB_APPLICATION_TOKEN"),  # noqa: B008
        collection_name: str = "documents",
        embedding_dimension: int = 768,
        duplicates_policy: DuplicatePolicy = DuplicatePolicy.NONE,
        similarity: str = "cosine",
        namespace: str | None = None,
    ) -> None:
        """
        The connection to Astra DB is established and managed through the JSON API.

        The required credentials (api endpoint and application token) can be generated
        through the UI by clicking and the connect tab, and then selecting JSON API and
        Generate Configuration.

        :param api_endpoint: the Astra DB API endpoint.
        :param token: the Astra DB application token.
        :param collection_name: the current collection in the keyspace in the current Astra DB.
        :param embedding_dimension: dimension of embedding vector.
        :param duplicates_policy: handle duplicate documents based on DuplicatePolicy parameter options.
              Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
              - `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
                    it is skipped and not written.
              - `DuplicatePolicy.SKIP`: if a Document with the same ID already exists, it is skipped and not written.
              - `DuplicatePolicy.OVERWRITE`: if a Document with the same ID already exists, it is overwritten.
              - `DuplicatePolicy.FAIL`: if a Document with the same ID already exists, an error is raised.
        :param similarity: Similarity metric for new collections: `cosine`, `dot_product`, or `euclidean`.
            Existing collections retain their configured metric.
        :param namespace: The keyspace containing the collection, or the SDK default when omitted.

        :raises ValueError: if the API endpoint or token is not set.
        """
        resolved_api_endpoint = api_endpoint.resolve_value()
        if resolved_api_endpoint is None:
            msg = (
                "AstraDocumentStore expects the API endpoint. "
                "Set the ASTRA_DB_API_ENDPOINT environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)
        self.resolved_api_endpoint = resolved_api_endpoint

        resolved_token = token.resolve_value()
        if resolved_token is None:
            msg = (
                "AstraDocumentStore expects an authentication token. "
                "Set the ASTRA_DB_APPLICATION_TOKEN environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)
        self.resolved_token = resolved_token

        self.api_endpoint = api_endpoint
        self.token = token
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.duplicates_policy = duplicates_policy
        self.similarity = similarity
        self.namespace = namespace
        self._collection: Collection | None = None
        self._async_collection: AsyncCollection | None = None
        self._async_collection_lock = Lock()
        self._async_loop: AbstractEventLoop | None = None

    def _get_collection(self) -> Collection:
        if self._collection is None:
            database = _data_api_client().get_database(
                api_endpoint=self.resolved_api_endpoint,
                token=self.resolved_token,
                keyspace=self.namespace,
            )
            descriptor = _find_collection(self.collection_name, database.list_collections())
            if descriptor is not None:
                warning = _collection_indexing_warning(descriptor)
                if warning is not None:
                    warn(warning, UserWarning, stacklevel=3)
                self._collection = database.get_collection(self.collection_name)
            else:
                # Listing and creation are not atomic; propagate concurrent configuration conflicts.
                self._collection = database.create_collection(
                    name=self.collection_name,
                    definition=_collection_definition(self.embedding_dimension, self.similarity),
                )
        return self._collection

    def _reset_async_state_on_loop_change(self) -> None:
        # The cached collection's HTTP client and the lock are bound to the loop that first used them, e.g.
        # repeated `asyncio.run(...)` calls each get a new loop. A stale collection can't be closed from
        # another loop, so it is dropped.
        loop = get_running_loop()
        if self._async_loop is not loop:
            self._async_collection = None
            self._async_collection_lock = Lock()
            self._async_loop = loop

    async def _get_async_collection(self) -> AsyncCollection:
        self._reset_async_state_on_loop_change()
        async with self._async_collection_lock:
            if self._async_collection is None:
                async with _data_api_client().get_async_database(
                    api_endpoint=self.resolved_api_endpoint,
                    token=self.resolved_token,
                    keyspace=self.namespace,
                ) as database:
                    descriptor = _find_collection(self.collection_name, await database.list_collections())
                    if descriptor is not None:
                        warning = _collection_indexing_warning(descriptor)
                        if warning is not None:
                            warn(warning, UserWarning, stacklevel=3)
                        self._async_collection = database.get_collection(self.collection_name)
                    else:
                        # Listing and creation are not atomic; propagate concurrent configuration conflicts.
                        self._async_collection = await database.create_collection(
                            name=self.collection_name,
                            definition=_collection_definition(self.embedding_dimension, self.similarity),
                        )
            return self._async_collection

    def close(self) -> None:
        """
        Drop the cached synchronous collection without deleting documents.

        AstraPy 2 exposes no way to release synchronous connections, so this only discards the collection; the next
        synchronous operation creates a new one.
        """
        self._collection = None

    async def close_async(self) -> None:
        """
        Release the cached async collection connection without deleting documents.

        Call this on the same event loop as the async methods, after they have finished and before
        closing the loop. Repeated calls are safe; a later call opens a new connection. Calls on a new
        event loop open a new connection automatically.
        """
        self._reset_async_state_on_loop_change()
        async with self._async_collection_lock:
            if self._async_collection is not None:
                # AstraPy 2 exposes connection cleanup through its async context manager protocol.
                await self._async_collection.__aexit__()
                self._async_collection = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AstraDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        :raises ValueError: The serialized `duplicates_policy` is not a valid policy name.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_endpoint", "token"])
        policy = data["init_parameters"].get("duplicates_policy")
        if isinstance(policy, str):
            try:
                data["init_parameters"]["duplicates_policy"] = DuplicatePolicy[policy]
            except KeyError as e:
                msg = f"Invalid duplicates_policy '{policy}'. Expected one of {[p.name for p in DuplicatePolicy]}."
                raise ValueError(msg) from e
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        return default_to_dict(
            self,
            api_endpoint=self.api_endpoint.to_dict(),
            token=self.token.to_dict(),
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_dimension,
            duplicates_policy=self.duplicates_policy.name,
            similarity=self.similarity,
            namespace=self.namespace,
        )

    def _resolve_policy(self, policy: DuplicatePolicy | None) -> DuplicatePolicy:
        if policy is None or policy == DuplicatePolicy.NONE:
            if self.duplicates_policy is not None and self.duplicates_policy != DuplicatePolicy.NONE:
                return self.duplicates_policy
            return DuplicatePolicy.SKIP
        return policy

    @staticmethod
    def _convert_input_document(document: dict | Document) -> dict[str, Any]:
        if isinstance(document, Document):
            document_dict = document.to_dict(flatten=False)
        elif isinstance(document, dict):
            document_dict = document
        else:
            msg = f"Unsupported type for documents, documents is of type {type(document)}."
            raise ValueError(msg)

        if "id" in document_dict:
            if "_id" not in document_dict:
                document_dict["_id"] = document_dict.pop("id")
            elif "_id" in document_dict:
                msg = f"Duplicate id definitions, both 'id' and '_id' present in document {document_dict}"
                raise Exception(msg)
        if "_id" in document_dict:
            if not isinstance(document_dict["_id"], str):
                msg = f"Document id {document_dict['_id']} is not a string, but is of type {type(document_dict['_id'])}"
                raise Exception(msg)

        if embedding := document_dict.pop("embedding", []):
            document_dict["$vector"] = embedding

        if "sparse_embedding" in document_dict:
            sparse_embedding = document_dict.pop("sparse_embedding", None)
            if sparse_embedding:
                logger.warning(
                    "Document {id} has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Astra is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    id=document_dict["_id"],
                )

        return document_dict

    @staticmethod
    def _overwrite_operation(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        fields = {key: value for key, value in document.items() if key != "_id"}
        return {"_id": document["_id"]}, {"$set": fields}

    @staticmethod
    def _log_written(policy: DuplicatePolicy, inserted_ids: list[str], updated_ids: list[str]) -> None:
        if inserted_ids:
            logger.info("write_documents inserted documents with id {ids}", ids=inserted_ids)
        else:
            logger.warning("No documents written. Argument policy set to {policy}", policy=policy.name)
        if policy == DuplicatePolicy.OVERWRITE:
            logger.info("write_documents updated documents with id {ids}", ids=updated_ids)

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Indexes documents for later queries.

        :param documents: a list of Haystack Document objects.
        :param policy: handle duplicate documents based on DuplicatePolicy parameter options.
            Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
            - `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
                it is skipped and not written.
            - `DuplicatePolicy.SKIP`: If a Document with the same ID already exists,
                it is skipped and not written.
            - `DuplicatePolicy.OVERWRITE`: If a Document with the same ID already exists, it is overwritten.
            - `DuplicatePolicy.FAIL`: If a Document with the same ID already exists, an error is raised.
        :returns: number of documents written.
        :raises ValueError: if the documents are not of type Document or dict.
        :raises DuplicateDocumentError: if a document with the same ID already exists and policy is set to FAIL.
        :raises Exception: if the document ID is not a string or if `id` and `_id` are both present in the document.
        """
        policy = self._resolve_policy(policy)
        documents_to_write = [self._convert_input_document(doc) for doc in documents]
        collection = self._get_collection()

        new_documents: list[dict[str, Any]] = []
        duplicate_documents: list[dict[str, Any]] = []
        for doc in documents_to_write:
            in_batch = any(d["_id"] == doc["_id"] for d in new_documents)
            if in_batch or collection.find_one({"_id": doc["_id"]}, projection={"_id": True}) is not None:
                if policy == DuplicatePolicy.FAIL:
                    msg = f"ID '{doc['_id']}' already exists."
                    raise DuplicateDocumentError(msg)
                duplicate_documents.append(doc)
            else:
                new_documents.append(doc)

        inserted_ids: list[str] = []
        for batch in _batches(new_documents, MAX_BATCH_SIZE):
            inserted_ids.extend(str(_id) for _id in collection.insert_many(documents=batch).inserted_ids)

        updated_ids: list[str] = []
        if policy == DuplicatePolicy.OVERWRITE:
            for doc in duplicate_documents:
                filter_, update = self._overwrite_operation(doc)
                if collection.find_one_and_update(filter_, update, projection={"_id": True}) is None:
                    logger.warning("Document {document_id} not updated in Astra DB.", document_id=doc["_id"])
                else:
                    updated_ids.append(doc["_id"])

        self._log_written(policy, inserted_ids, updated_ids)
        return len(inserted_ids) + len(updated_ids)

    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Asynchronously indexes documents for later queries.

        :param documents: a list of Haystack Document objects.
        :param policy: handle duplicate documents based on DuplicatePolicy parameter options.
            Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
            - `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
                it is skipped and not written.
            - `DuplicatePolicy.SKIP`: If a Document with the same ID already exists,
                it is skipped and not written.
            - `DuplicatePolicy.OVERWRITE`: If a Document with the same ID already exists, it is overwritten.
            - `DuplicatePolicy.FAIL`: If a Document with the same ID already exists, an error is raised.
        :returns: number of documents written.
        :raises ValueError: if the documents are not of type Document or dict.
        :raises DuplicateDocumentError: if a document with the same ID already exists and policy is set to FAIL.
        :raises Exception: if the document ID is not a string or if `id` and `_id` are both present in the document.
        """
        policy = self._resolve_policy(policy)
        documents_to_write = [self._convert_input_document(doc) for doc in documents]
        collection = await self._get_async_collection()

        new_documents: list[dict[str, Any]] = []
        duplicate_documents: list[dict[str, Any]] = []
        for doc in documents_to_write:
            in_batch = any(d["_id"] == doc["_id"] for d in new_documents)
            if in_batch or await collection.find_one({"_id": doc["_id"]}, projection={"_id": True}) is not None:
                if policy == DuplicatePolicy.FAIL:
                    msg = f"ID '{doc['_id']}' already exists."
                    raise DuplicateDocumentError(msg)
                duplicate_documents.append(doc)
            else:
                new_documents.append(doc)

        inserted_ids: list[str] = []
        for batch in _batches(new_documents, MAX_BATCH_SIZE):
            result = await collection.insert_many(documents=batch)
            inserted_ids.extend(str(_id) for _id in result.inserted_ids)

        updated_ids: list[str] = []
        if policy == DuplicatePolicy.OVERWRITE:
            for doc in duplicate_documents:
                filter_, update = self._overwrite_operation(doc)
                if await collection.find_one_and_update(filter_, update, projection={"_id": True}) is None:
                    logger.warning("Document {document_id} not updated in Astra DB.", document_id=doc["_id"])
                else:
                    updated_ids.append(doc["_id"])

        self._log_written(policy, inserted_ids, updated_ids)
        return len(inserted_ids) + len(updated_ids)

    def count_documents(self) -> int:
        """
        Counts the number of documents in the document store.

        :returns: the number of documents in the document store.
        """
        return self._get_collection().count_documents({}, upper_bound=10_000)

    async def count_documents_async(self) -> int:
        """
        Asynchronously counts the number of documents in the document store.

        :returns: the number of documents in the document store.
        """
        collection = await self._get_async_collection()
        return await collection.count_documents({}, upper_bound=10_000)

    @staticmethod
    def _normalize_new_filter_input(filters: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(filters, dict):
            msg = "Filters must be a dictionary"
            raise AstraDocumentStoreFilterError(msg)

        normalized_filters = filters.copy()
        if "id" in normalized_filters:
            normalized_filters["_id"] = normalized_filters.pop("id")

        return normalized_filters

    @staticmethod
    def _infer_metadata_field_type(values: list[Any]) -> str:
        inferred_types = set()
        for value in values:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, bool):
                        inferred_types.add("boolean")
                    elif isinstance(item, int | float):
                        inferred_types.add("long")
                    elif isinstance(item, str):
                        inferred_types.add("keyword")
            elif isinstance(value, bool):
                inferred_types.add("boolean")
            elif isinstance(value, int | float):
                inferred_types.add("long")
            elif isinstance(value, str):
                inferred_types.add("keyword")

        if not inferred_types:
            return "keyword"

        if len(inferred_types) > 1:
            logger.warning("Field has mixed metadata types {types}. Defaulting to 'keyword'.", types=inferred_types)
            return "keyword"

        return next(iter(inferred_types))

    @staticmethod
    def _normalize_distinct_values(values: list[Any]) -> list[Any]:
        # Values of different types can be equal in Python (`1 == True`, `1 == 1.0`), so a plain set/sorted()
        # would silently merge them or crash comparing incomparable types (e.g. str vs bool). Dedupe and sort
        # by (type, value) instead to keep values of different types distinct.
        #
        # This can't recover a whole-number float's distinctness from int though: AstraDB's Data API
        # canonicalizes any whole-number float (e.g. 1.0) to an int on storage, unconditionally - not
        # only when it shares a field with an actual int - so `values` here may already contain int
        # where the caller wrote a whole-number float, before we ever see it.
        seen: set[tuple[str, Any]] = set()
        normalized_values: list[Any] = []
        for value in values:
            items = value if isinstance(value, list) else [value]
            for item in items:
                if item is None:
                    continue
                dedup_key = (type(item).__name__, item)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    normalized_values.append(item)
        return sorted(normalized_values, key=lambda value: (type(value).__name__, str(value)))

    # The find helpers take the collection so that public methods fetch it themselves: this keeps the
    # legacy-indexing warning's `stacklevel` pointing at the caller of the public method.
    @staticmethod
    def _find_documents(
        collection: Collection,
        filters: dict[str, Any] | None,
        *,
        vector: list[float] | None = None,
        limit: int | None = None,
    ) -> list[Document]:
        responses = list(collection.find(**_find_kwargs(filters, vector=vector, limit=limit)))
        return _to_documents(responses)

    @staticmethod
    async def _find_documents_async(
        collection: AsyncCollection,
        filters: dict[str, Any] | None,
        *,
        vector: list[float] | None = None,
        limit: int | None = None,
    ) -> list[Document]:
        responses = [
            response async for response in collection.find(**_find_kwargs(filters, vector=vector, limit=limit))
        ]
        return _to_documents(responses)

    @staticmethod
    def _filter_queries(filters: dict[str, Any] | None) -> list[tuple[dict[str, Any] | None, list[float] | None]]:
        # Returns one (converted filters, query vector) pair per `find` call that `filter_documents` needs.
        if not isinstance(filters, dict) and filters is not None:
            msg = "Filters must be a dictionary or None"
            raise AstraDocumentStoreFilterError(msg)
        if filters is None:
            return [(None, None)]

        filters = filters.copy()
        if "id" in filters:
            filters["_id"] = filters.pop("id")
        if "embedding" not in filters:
            return [(_convert_filters(filters), None)]

        embedding = filters.pop("embedding")
        vectors = embedding["$in"] if "$in" in embedding else [embedding]
        converted_filters = _convert_filters(filters)
        return [(converted_filters, vector) for vector in vectors]

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns at most 1000 documents that match the filter.

        :param filters: filters to apply.
        :returns: matching documents.
        :raises AstraDocumentStoreFilterError: if the filter is invalid or not supported by this class.
        """
        queries = self._filter_queries(filters)
        collection = self._get_collection()
        documents = []
        for converted_filters, vector in queries:
            documents.extend(self._find_documents(collection, converted_filters, vector=vector, limit=1000))
        return documents

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Asynchronously returns at most 1000 documents that match the filter.

        :param filters: filters to apply.
        :returns: matching documents.
        :raises AstraDocumentStoreFilterError: if the filter is invalid or not supported by this class.
        """
        queries = self._filter_queries(filters)
        collection = await self._get_async_collection()
        documents = []
        for converted_filters, vector in queries:
            documents.extend(await self._find_documents_async(collection, converted_filters, vector=vector, limit=1000))
        return documents

    def get_documents_by_id(self, ids: list[str]) -> list[Document]:
        """
        Gets documents by their IDs.

        :param ids: the IDs of the documents to retrieve.
        :returns: the matching documents.
        """
        collection = self._get_collection()
        documents = []
        for batch in _batches(ids, MAX_BATCH_SIZE):
            documents.extend(self._find_documents(collection, {"_id": {"$in": batch}}))
        return documents

    async def get_documents_by_id_async(self, ids: list[str]) -> list[Document]:
        """
        Asynchronously gets documents by their IDs.

        :param ids: the IDs of the documents to retrieve.
        :returns: the matching documents.
        """
        collection = await self._get_async_collection()
        documents = []
        for batch in _batches(ids, MAX_BATCH_SIZE):
            documents.extend(await self._find_documents_async(collection, {"_id": {"$in": batch}}))
        return documents

    @staticmethod
    def _single_document(documents: list[Document], document_id: str) -> Document:
        if not documents:
            msg = f"Document {document_id} does not exist"
            raise MissingDocumentError(msg)
        return documents[0]

    def get_document_by_id(self, document_id: str) -> Document:
        """
        Gets a document by its ID.

        :param document_id: the ID to filter by
        :returns: the found document
        :raises MissingDocumentError: if the document is not found
        """
        documents = self._find_documents(self._get_collection(), {"_id": {"$in": [document_id]}})
        return self._single_document(documents, document_id)

    async def get_document_by_id_async(self, document_id: str) -> Document:
        """
        Asynchronously gets a document by its ID.

        :param document_id: the ID to filter by
        :returns: the found document
        :raises MissingDocumentError: if the document is not found
        """
        collection = await self._get_async_collection()
        documents = await self._find_documents_async(collection, {"_id": {"$in": [document_id]}})
        return self._single_document(documents, document_id)

    def search(self, query_embedding: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Perform a search for a list of queries.

        :param query_embedding: a list of query embeddings.
        :param top_k: the number of results to return.
        :param filters: filters to apply during search.
        :returns: matching documents.
        """
        converted_filters = _convert_filters(filters)
        return self._find_documents(self._get_collection(), converted_filters, vector=query_embedding, limit=top_k)

    async def search_async(
        self, query_embedding: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[Document]:
        """
        Search using AstraPy's native async API.

        The collection connection is initialized lazily and reused across searches on the same event loop.
        Call `close_async()` when finished, including after failures or cancellation, before closing the loop.

        :param query_embedding: A list of query embeddings.
        :param top_k: The number of results to return.
        :param filters: Filters to apply during search.
        :returns: Matching documents, including embeddings, metadata and similarity scores.
        """
        converted_filters = _convert_filters(filters)
        collection = await self._get_async_collection()
        return await self._find_documents_async(collection, converted_filters, vector=query_embedding, limit=top_k)

    @staticmethod
    def _check_deleted(document_ids: list[str], deletion_counter: int) -> None:
        logger.info("{count} documents deleted", count=deletion_counter)
        if document_ids and deletion_counter == 0:
            msg = f"Document {document_ids} does not exist"
            raise MissingDocumentError(msg)

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents from the document store.

        :param document_ids: IDs of the documents to delete.
        :raises MissingDocumentError: if no document was deleted but document IDs were provided.
        """
        collection = self._get_collection()
        if collection.find_one({}, projection={"_id": True}) is None:
            logger.info("No documents in document store")
            return
        deletion_counter = 0
        for batch in _batches(document_ids, MAX_BATCH_SIZE):
            deletion_counter += collection.delete_many({"_id": {"$in": batch}}).deleted_count
        self._check_deleted(document_ids, deletion_counter)

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Asynchronously deletes documents from the document store.

        :param document_ids: IDs of the documents to delete.
        :raises MissingDocumentError: if no document was deleted but document IDs were provided.
        """
        collection = await self._get_async_collection()
        if await collection.find_one({}, projection={"_id": True}) is None:
            logger.info("No documents in document store")
            return
        deletion_counter = 0
        for batch in _batches(document_ids, MAX_BATCH_SIZE):
            deletion_counter += (await collection.delete_many({"_id": {"$in": batch}})).deleted_count
        self._check_deleted(document_ids, deletion_counter)

    @staticmethod
    def _log_delete_all(deletion_counter: int) -> None:
        # The Data API reports -1 when all documents of a collection are deleted.
        if deletion_counter == -1:
            logger.info("All documents deleted")
        else:
            logger.error("Could not delete all documents")

    def delete_all_documents(self) -> None:
        """
        Deletes all documents from the document store.

        :raises DocumentStoreError: if the documents could not be deleted.
        """
        try:
            deletion_counter = self._get_collection().delete_many({}).deleted_count
        except Exception as e:
            msg = f"Failed to delete all documents from Astra: {e!s}"
            raise DocumentStoreError(msg) from e
        self._log_delete_all(deletion_counter)

    async def delete_all_documents_async(self) -> None:
        """
        Asynchronously deletes all documents from the document store.

        :raises DocumentStoreError: if the documents could not be deleted.
        """
        try:
            collection = await self._get_async_collection()
            deletion_counter = (await collection.delete_many({})).deleted_count
        except Exception as e:
            msg = f"Failed to delete all documents from Astra: {e!s}"
            raise DocumentStoreError(msg) from e
        self._log_delete_all(deletion_counter)

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes documents that match the provided filters.

        :param filters: The filters to apply to find documents to delete.
        :returns: The number of documents deleted.
        :raises AstraDocumentStoreFilterError: if the filter is invalid or not supported.
        """
        converted_filters = _convert_filters(self._normalize_new_filter_input(filters))
        deletion_count = self._get_collection().delete_many(converted_filters or {}).deleted_count
        logger.info("{count} documents deleted by filter", count=deletion_count)
        return deletion_count

    async def delete_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously deletes documents that match the provided filters.

        :param filters: The filters to apply to find documents to delete.
        :returns: The number of documents deleted.
        :raises AstraDocumentStoreFilterError: if the filter is invalid or not supported.
        """
        converted_filters = _convert_filters(self._normalize_new_filter_input(filters))
        collection = await self._get_async_collection()
        deletion_count = (await collection.delete_many(converted_filters or {})).deleted_count
        logger.info("{count} documents deleted by filter", count=deletion_count)
        return deletion_count

    @staticmethod
    def _update_by_filter_operation(
        filters: dict[str, Any], meta: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized_filters = AstraDocumentStore._normalize_new_filter_input(filters)
        if not isinstance(meta, dict):
            msg = "Meta must be a dictionary"
            raise AstraDocumentStoreFilterError(msg)
        # use dot notation to update nested fields in the meta-object - ensures fields are created if they don't exist
        update_fields = {f"meta.{key}": value for key, value in meta.items()}
        return _convert_filters(normalized_filters) or {}, {"$set": update_fields}

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates documents that match the provided filters with the given metadata.

        :param filters: The filters to apply to find documents to update.
        :param meta: The metadata fields to update. This will be merged with existing metadata.

        :returns:
            The number of documents updated.

        :raises AstraDocumentStoreFilterError: if the filter is invalid or not supported.
        """
        filter_, update = self._update_by_filter_operation(filters, meta)
        update_count = self._get_collection().update_many(filter_, update).update_info["nModified"]
        logger.info("{count} documents updated by filter", count=update_count)
        return update_count

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Asynchronously updates documents that match the provided filters with the given metadata.

        :param filters: The filters to apply to find documents to update.
        :param meta: The metadata fields to update. This will be merged with existing metadata.

        :returns:
            The number of documents updated.

        :raises AstraDocumentStoreFilterError: if the filter is invalid or not supported.
        """
        filter_, update = self._update_by_filter_operation(filters, meta)
        collection = await self._get_async_collection()
        update_count = (await collection.update_many(filter_, update)).update_info["nModified"]
        logger.info("{count} documents updated by filter", count=update_count)
        return update_count

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Applies a filter and counts the documents that matched it.

        :param filters: The filters to apply to the document list.
        :returns: The number of documents that match the filter.
        """
        converted_filters = _convert_filters(self._normalize_new_filter_input(filters))
        return self._get_collection().count_documents(converted_filters or {}, upper_bound=1_000_000_000)

    async def count_documents_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously applies a filter and counts the documents that matched it.

        :param filters: The filters to apply to the document list.
        :returns: The number of documents that match the filter.
        """
        converted_filters = _convert_filters(self._normalize_new_filter_input(filters))
        collection = await self._get_async_collection()
        return await collection.count_documents(converted_filters or {}, upper_bound=1_000_000_000)

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]:
        """
        Applies a filter selecting documents and counts the unique values for each meta field of the matched documents.

        :param filters: The filters to apply to the document list.
        :param metadata_fields: The metadata fields to count unique values for.
        :returns: A dictionary where the keys are the metadata field names and the values are the count of unique
            values.
        """
        converted_filters = _convert_filters(self._normalize_new_filter_input(filters))
        collection = self._get_collection()
        return {
            field: len(self._normalize_distinct_values(collection.distinct(f"meta.{field}", filter=converted_filters)))
            for field in metadata_fields
        }

    async def count_unique_metadata_by_filter_async(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """
        Asynchronously counts the unique values of each meta field across the documents matching a filter.

        :param filters: The filters to apply to the document list.
        :param metadata_fields: The metadata fields to count unique values for.
        :returns: A dictionary where the keys are the metadata field names and the values are the count of unique
            values.
        """
        converted_filters = _convert_filters(self._normalize_new_filter_input(filters))
        collection = await self._get_async_collection()
        return {
            field: len(
                self._normalize_distinct_values(await collection.distinct(f"meta.{field}", filter=converted_filters))
            )
            for field in metadata_fields
        }

    @staticmethod
    def _metadata_fields_info(documents: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
        if not documents:
            return {}

        fields_info: dict[str, dict[str, str]] = {}
        if any(document.get("content") is not None for document in documents):
            fields_info["content"] = {"type": "text"}

        field_values: dict[str, list[Any]] = {}
        for document in documents:
            for field, value in document.get("meta", {}).items():
                field_values.setdefault(field, []).append(value)

        for field, values in field_values.items():
            fields_info[field] = {"type": AstraDocumentStore._infer_metadata_field_type(values)}

        return fields_info

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns the metadata fields and the corresponding types.

        :returns: A dictionary mapping field names to dictionaries with a `type` key.
        """
        documents = list(self._get_collection().find(projection={"content": 1, "meta": 1}))
        return self._metadata_fields_info(documents)

    async def get_metadata_fields_info_async(self) -> dict[str, dict[str, str]]:
        """
        Asynchronously returns the metadata fields and the corresponding types.

        :returns: A dictionary mapping field names to dictionaries with a `type` key.
        """
        collection = await self._get_async_collection()
        documents = [document async for document in collection.find(projection={"content": 1, "meta": 1})]
        return self._metadata_fields_info(documents)

    @staticmethod
    def _min_max(distinct_values: list[Any]) -> dict[str, Any]:
        comparable_values = [value for value in distinct_values if isinstance(value, str | int | float | bool)]
        if not comparable_values:
            return {"min": None, "max": None}
        return {"min": min(comparable_values), "max": max(comparable_values)}

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, Any]:
        """
        For a given metadata field, find its max and min value.

        :param metadata_field: The metadata field to inspect.
        :returns: A dictionary with `min` and `max`.
        """
        field = metadata_field.removeprefix("meta.")
        return self._min_max(self._get_collection().distinct(f"meta.{field}"))

    async def get_metadata_field_min_max_async(self, metadata_field: str) -> dict[str, Any]:
        """
        Asynchronously, for a given metadata field, find its max and min value.

        :param metadata_field: The metadata field to inspect.
        :returns: A dictionary with `min` and `max`.
        """
        field = metadata_field.removeprefix("meta.")
        collection = await self._get_async_collection()
        return self._min_max(await collection.distinct(f"meta.{field}"))

    @staticmethod
    def _unique_values_query(metadata_field: str, filters: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
        field = metadata_field.removeprefix("meta.")
        converted_filters = None
        if filters:
            converted_filters = _convert_filters(AstraDocumentStore._normalize_new_filter_input(filters))
        return f"meta.{field}", converted_filters

    @staticmethod
    def _paginate_unique_values(
        distinct_values: list[Any], search_term: str | None, from_: int, size: int
    ) -> tuple[list[Any], int]:
        values = AstraDocumentStore._normalize_distinct_values(distinct_values)
        if search_term:
            search_term_lower = search_term.lower()
            values = [value for value in values if search_term_lower in str(value).lower()]
        return values[from_ : from_ + size], len(values)

    def get_metadata_field_unique_values(
        self,
        metadata_field: str,
        search_term: str | None = None,
        from_: int = 0,
        size: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[Any], int]:
        """
        Retrieves unique values for a field matching a search term or all possible values if no search term is given.

        **Note**: values of different types are kept distinct even when they compare equal in Python
        (e.g. the int `1`, the bool `True` and the str `"1"` are returned as three separate values), with
        one exception.
        AstraDB's Data API canonicalizes any whole-number float (e.g. `1.0`) to an int on storage, unconditionally so a
        whole-number float is always returned back as an int, never as a float.
        Example: 1.0 (float) is sent to storage and comes back a 1 (int)

        Exception are floats with a fractional part (e.g. `1.5`) are unaffected and round-trip normally.

        :param metadata_field: The metadata field to inspect.
        :param search_term: Optional case-insensitive substring search term.
        :param from_: The starting index for pagination.
        :param size: The number of values to return.
        :param filters: Optional filters to restrict the documents considered.
        :returns: A tuple containing the paginated values (in their original type) and the total count.
        """
        key, converted_filters = self._unique_values_query(metadata_field, filters)
        distinct_values = self._get_collection().distinct(key, filter=converted_filters)
        return self._paginate_unique_values(distinct_values, search_term, from_, size)

    async def get_metadata_field_unique_values_async(
        self,
        metadata_field: str,
        search_term: str | None = None,
        from_: int = 0,
        size: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[Any], int]:
        """
        Asynchronously retrieves unique values for a field, optionally matching a search term.

        **Note**: values of different types are kept distinct even when they compare equal in Python
        (e.g. the int `1`, the bool `True` and the str `"1"` are returned as three separate values), with
        one exception.
        AstraDB's Data API canonicalizes any whole-number float (e.g. `1.0`) to an int on storage, unconditionally so a
        whole-number float is always returned back as an int, never as a float.
        Example: 1.0 (float) is sent to storage and comes back a 1 (int)

        Exception are floats with a fractional part (e.g. `1.5`) are unaffected and round-trip normally.

        :param metadata_field: The metadata field to inspect.
        :param search_term: Optional case-insensitive substring search term.
        :param from_: The starting index for pagination.
        :param size: The number of values to return.
        :param filters: Optional filters to restrict the documents considered.
        :returns: A tuple containing the paginated values (in their original type) and the total count.
        """
        key, converted_filters = self._unique_values_query(metadata_field, filters)
        collection = await self._get_async_collection()
        distinct_values = await collection.distinct(key, filter=converted_filters)
        return self._paginate_unique_values(distinct_values, search_term, from_, size)
