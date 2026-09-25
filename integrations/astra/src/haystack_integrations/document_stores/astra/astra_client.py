# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Generator
from typing import Any
from warnings import warn

from astrapy import DataAPIClient as AstraDBClient
from astrapy.api_options import APIOptions, SerdesOptions
from astrapy.constants import FilterType, ReturnDocument
from astrapy.info import CollectionDescriptor
from haystack import logging
from haystack.version import __version__ as integration_version
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)

NON_INDEXED_FIELDS = ["metadata._node_content", "content"]
CALLER_NAME = "haystack"
# Preserve embedding precision and plain Python types when reading with AstraPy 2.
_API_OPTIONS = APIOptions(serdes_options=SerdesOptions(binary_encode_vectors=False, custom_datatypes_in_reading=False))


def _collection_definition(embedding_dimension: int, similarity: str) -> dict[str, Any]:
    return {
        "vector": {"dimension": embedding_dimension, "metric": similarity},
        "indexing": {"deny": NON_INDEXED_FIELDS},
    }


def _find_collection(collection_name: str, collections: list[CollectionDescriptor]) -> CollectionDescriptor | None:
    return next((descriptor for descriptor in collections if descriptor.name == collection_name), None)


def _collection_indexing_warning(descriptor: CollectionDescriptor) -> str | None:
    indexing = descriptor.definition.indexing or {}
    if not indexing:
        return (
            f"Collection '{descriptor.name}' is detected as having indexing turned on for all fields "
            "(either created manually or by older versions of this plugin). This implies stricter "
            "limitations on the amount of text each entry can store. Consider indexing anew on a "
            "fresh collection to be able to store longer texts."
        )
    if indexing != {"deny": NON_INDEXED_FIELDS}:
        return (
            f"Collection '{descriptor.name}' has unexpected 'indexing' settings "
            f"(options.indexing = {json.dumps(indexing)}). This can result in odd behaviour when running "
            "metadata filtering and/or unwarranted limitations on storing long texts. "
            "Consider indexing anew on a fresh collection."
        )
    return None


def _vector_find_kwargs(vector: list[float], top_k: int | None, filters: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "filter": filters,
        "sort": {"$vector": vector},
        "limit": top_k,
        "include_similarity": True,
        "projection": {"*": 1},
    }


@dataclass
class Response:
    document_id: str
    text: str | None
    values: list | None
    metadata: dict | None
    score: float | None


@dataclass
class QueryResponse:
    matches: list[Response]

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Return the value for the given key."""
        return self.__dict__[key]


def _format_query_response(
    responses: list[dict[str, Any]] | None,
    *,
    include_metadata: bool | None,
    include_values: bool | None,
) -> QueryResponse:
    final_res = []
    for raw_response in responses or []:
        response = raw_response.copy()
        document_id = response.pop("_id")
        score = response.pop("$similarity", None)
        text = response.pop("content", None)
        values = response.pop("$vector", None) if include_values else []
        metadata = response if include_metadata else {}
        final_res.append(Response(document_id, text, values, metadata, score))
    return QueryResponse(final_res)


class AstraClient:
    """
    A client for interacting with an Astra index via JSON API
    """

    def __init__(
        self,
        api_endpoint: str,
        token: str,
        collection_name: str,
        embedding_dimension: int,
        similarity_function: str,
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
        :param similarity_function: the similarity function to use for the index.
        :param namespace: the namespace to use for the collection.
        """
        self.api_endpoint = api_endpoint
        self.token = token
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.similarity_function = similarity_function
        self.namespace = namespace

        # Get the keyspace from the collection name
        my_client = AstraDBClient(
            callers=[(CALLER_NAME, integration_version)],
            api_options=_API_OPTIONS,
        )

        # Get the database object
        self._astra_db = my_client.get_database(
            api_endpoint=api_endpoint,
            token=token,
            keyspace=namespace,
        )

        # AstraPy 2 no longer checks for existing collections before creation.
        descriptor = _find_collection(collection_name, self._astra_db.list_collections())
        if descriptor is not None:
            warning = _collection_indexing_warning(descriptor)
            if warning is not None:
                warn(warning, UserWarning, stacklevel=2)
            self._astra_db_collection = self._astra_db.get_collection(collection_name)
        else:
            # Listing and creation are not atomic; propagate concurrent configuration conflicts.
            self._astra_db_collection = self._astra_db.create_collection(
                name=collection_name,
                definition=_collection_definition(embedding_dimension, similarity_function),
            )

    def query(
        self,
        *,
        vector: list[float] | None = None,
        query_filter: dict[str, str | float | int | bool | list | dict] | None = None,
        top_k: int | None = None,
        include_metadata: bool | None = None,
        include_values: bool | None = None,
    ) -> QueryResponse:
        """
        Search the Astra index using a query vector.

        :param vector: the query vector. This should be the same length as the dimension of the index being queried.
            Each `query()` request can contain only one of the parameters `queries`, `id` or `vector`.
        :param query_filter: the filter to apply. You can use vector metadata to limit your search.
        :param top_k: the number of results to return for each query. Must be an integer greater than 1.
        :param include_metadata: indicates whether metadata is included in the response as well as the ids.
            If omitted the server will use the default value of `False`.
        :param include_values: indicates whether values/vector is included in the response as well as the ids.
            If omitted the server will use the default value of `False`.
        :returns: object which contains the list of the closest vectors as ScoredVector objects, and namespace name.
        """
        # get vector data and scores
        if vector is None:
            responses = self._query_without_vector(top_k, query_filter)
        else:
            responses = self._query(vector, top_k, query_filter)

        # include_metadata means return all columns in the table (including text that got embedded)
        # include_values means return the vector of the embedding for the searched items
        formatted_response = _format_query_response(
            responses, include_metadata=include_metadata, include_values=include_values
        )

        return formatted_response

    def _query_without_vector(self, top_k: int | None, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query = {"filter": filters, "limit": top_k}

        return self.find_documents(query)

    def _query(
        self, vector: list[float], top_k: int | None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        responses = list(self._astra_db_collection.find(**_vector_find_kwargs(vector, top_k, filters)))
        if not responses:
            logger.warning("No documents found.")
        return responses

    def find_documents(
        self, find_query: dict[str, Any], projection: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Find documents in the Astra index.

        :param find_query: a dictionary with the query options
        :param projection: optional projection for the returned documents
        :returns: the documents found in the index
        """
        find_cursor = self._astra_db_collection.find(
            filter=find_query.get("filter"),
            sort=find_query.get("sort"),
            limit=find_query.get("limit"),
            include_similarity=find_query.get("includeSimilarity"),
            projection=projection or {"*": 1},
        )

        find_results = []
        for result in find_cursor:
            find_results.append(result)

        if not find_results:
            logger.warning("No documents found.")

        return find_results

    def find_one_document(self, find_query: dict[str, Any]) -> dict[str, Any] | None:
        """
        Find one document in the Astra index.

        :param find_query: a dictionary with the query options
        :returns: the document found in the index
        """
        find_result = self._astra_db_collection.find_one(
            filter=find_query.get("filter"),
            projection={"*": 1},
        )

        if not find_result:
            logger.warning("No document found.")

        return find_result

    def get_documents(self, ids: list[str], batch_size: int = 20) -> QueryResponse:
        """
        Get documents from the Astra index by their ids.

        :param ids: a list of document ids
        :param batch_size: the batch size to use when querying the index
        :returns: the documents found in the index
        """
        document_batch = []

        def batch_generator(chunks: list[str], batch_size: int) -> Generator[list[str], None, None]:
            for i in range(0, len(chunks), batch_size):
                i_end = min(len(chunks), i + batch_size)
                batch = chunks[i:i_end]
                yield batch

        for id_batch in batch_generator(ids, batch_size):
            docs = self.find_documents({"filter": {"_id": {"$in": id_batch}}})
            if docs:
                document_batch.extend(docs)

        formatted_docs = _format_query_response(document_batch, include_metadata=True, include_values=True)

        return formatted_docs

    def insert(self, documents: list[dict]) -> list[str]:
        """
        Insert documents into the Astra index.

        :param documents: a list of documents to insert
        :returns: the IDs of the inserted documents
        """
        insert_result = self._astra_db_collection.insert_many(documents=documents)
        inserted_ids = [str(_id) for _id in insert_result.inserted_ids]

        return inserted_ids

    def update_document(self, document: dict, id_key: str) -> bool:
        """
        Update a document in the Astra index.

        :param document: the document to update
        :param id_key: the key to use as the document id
        :returns: whether the document was updated successfully
        """
        document_id = document.pop(id_key)

        update_result = self._astra_db_collection.find_one_and_update(
            filter={id_key: document_id},
            update={"$set": document},
            return_document=ReturnDocument.AFTER,
            projection={"*": 1},
        )

        document[id_key] = document_id

        if update_result is None:
            logger.warning(f"Documents {document_id} not updated in Astra DB.")

            return False

        return True

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        filters: dict[str, str | float | int | bool | list | dict] | None = None,
    ) -> int:
        """
        Delete documents from the Astra index.

        :param ids: the ids of the documents to delete
        :param filters: additional filters to apply when deleting documents
        :returns: the number of documents deleted
        """
        query: dict[str, dict[str, Any]] = {}

        if ids is not None:
            query = {"deleteMany": {"filter": {"_id": {"$in": ids}}}}
        if filters is not None:
            query = {"deleteMany": {"filter": filters}}

        filter_dict = {}
        filter_dict = query.get("deleteMany", {}).get("filter", {})
        delete_result = self._astra_db_collection.delete_many(filter=filter_dict)

        return delete_result.deleted_count

    def delete_all_documents(self) -> int:
        """
        Delete all documents from the Astra index.

        :returns: the number of documents deleted
        """
        delete_result = self._astra_db_collection.delete_many(filter={})

        return delete_result.deleted_count

    def count_documents(self, filters: FilterType | None = None, upper_bound: int = 10000) -> int:
        """
        Count the number of documents in the Astra index.

        :param filters: optional filter to restrict the counted documents
        :param upper_bound: maximum expected count, required by Astra's API
        :returns: the number of documents in the index
        """
        return self._astra_db_collection.count_documents(filters or {}, upper_bound=upper_bound)

    def distinct(self, key: str, filters: FilterType | None = None) -> list[Any]:
        """
        Return the distinct values for a field in the Astra index.

        :param key: field name
        :param filters: optional filter to restrict the matching documents
        :returns: distinct values for the field
        """
        return self._astra_db_collection.distinct(key, filter=filters)

    def update(
        self,
        *,
        filters: dict[str, str | float | int | bool | list | dict],
        update: dict[str, Any],
    ) -> int:
        """
        Update multiple documents in the Astra index that match the filter.

        :param filters: the filter to match documents to update
        :param update: the update operations to apply (e.g., {"$set": {...}})

        :returns:
            The number of documents updated
        """
        update_result = self._astra_db_collection.update_many(filter=filters, update=update, upsert=False)

        return update_result.update_info["nModified"]
