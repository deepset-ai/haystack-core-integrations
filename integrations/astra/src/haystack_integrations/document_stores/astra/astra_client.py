# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional, Union
from warnings import warn

from astrapy import DataAPIClient as AstraDBClient
from astrapy.constants import ReturnDocument
from astrapy.exceptions import CollectionAlreadyExistsException
from haystack import logging
from haystack.version import __version__ as integration_version
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)

NON_INDEXED_FIELDS = ["metadata._node_content", "content"]
CALLER_NAME = "haystack"


@dataclass
class Response:
    document_id: str
    text: Optional[str]
    values: Optional[list]
    metadata: Optional[dict]
    score: Optional[float]


@dataclass
class QueryResponse:
    matches: list[Response]

    def get(self, key):
        return self.__dict__[key]


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
        namespace: Optional[str] = None,
    ):
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
        )

        # Get the database object
        self._astra_db = my_client.get_database(
            api_endpoint=api_endpoint,
            token=token,
            keyspace=namespace,
        )

        indexing_options = {"deny": NON_INDEXED_FIELDS}
        try:
            # Create and connect to the newly created collection
            self._astra_db_collection = self._astra_db.create_collection(
                name=collection_name,
                dimension=embedding_dimension,
                indexing=indexing_options,
            )
        except CollectionAlreadyExistsException as _:
            # possibly the collection is preexisting and has legacy
            # indexing settings: verify
            preexisting = [
                coll_descriptor
                for coll_descriptor in self._astra_db.list_collections()
                if coll_descriptor.name == collection_name
            ]

            if preexisting:
                # if it has no "indexing", it is a legacy collection;
                # otherwise it's unexpected: warn and proceed at user's risk
                pre_col_idx_opts = preexisting[0].options.indexing or {}
                if not pre_col_idx_opts:
                    warn(
                        (
                            f"Collection '{collection_name}' is detected as "
                            "having indexing turned on for all fields "
                            "(either created manually or by older versions "
                            "of this plugin). This implies stricter "
                            "limitations on the amount of text"
                            " each entry can store. Consider indexing anew on a"
                            " fresh collection to be able to store longer texts."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    self._astra_db_collection = self._astra_db.get_collection(
                        collection_name,
                    )
                # check if the indexing options match entirely
                elif pre_col_idx_opts == indexing_options:
                    self._astra_db_collection = self._astra_db.get_collection(
                        collection_name,
                    )
                else:
                    options_json = json.dumps(pre_col_idx_opts)
                    warn(
                        (
                            f"Collection '{collection_name}' has unexpected 'indexing'"
                            f" settings (options.indexing = {options_json})."
                            " This can result in odd behaviour when running "
                            " metadata filtering and/or unwarranted limitations"
                            " on storing long texts. Consider indexing anew on a"
                            " fresh collection."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    self._collection = self._astra_db.get_collection(
                        collection_name,
                    )
            else:
                # other exception
                raise

    def query(
        self,
        *,
        vector: Optional[list[float]] = None,
        query_filter: Optional[dict[str, Union[str, float, int, bool, list, dict]]] = None,
        top_k: Optional[int] = None,
        include_metadata: Optional[bool] = None,
        include_values: Optional[bool] = None,
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
        formatted_response = self._format_query_response(responses, include_metadata, include_values)

        return formatted_response

    def _query_without_vector(self, top_k, filters=None):
        query = {"filter": filters, "limit": top_k}

        return self.find_documents(query)

    @staticmethod
    def _format_query_response(responses, include_metadata, include_values):
        final_res = []

        if responses is None:
            return QueryResponse(matches=[])

        for response in responses:
            _id = response.pop("_id")
            score = response.pop("$similarity", None)
            text = response.pop("content", None)
            values = response.pop("$vector", None) if include_values else []

            metadata = response if include_metadata else {}  # Add all remaining fields to the metadata

            rsp = Response(_id, text, values, metadata, score)

            final_res.append(rsp)

        return QueryResponse(final_res)

    def _query(self, vector, top_k, filters=None):
        query = {"sort": {"$vector": vector}, "limit": top_k, "includeSimilarity": True}

        if filters is not None:
            query["filter"] = filters

        result = self.find_documents(query)

        return result

    def find_documents(self, find_query):
        """
        Find documents in the Astra index.

        :param find_query: a dictionary with the query options
        :returns: the documents found in the index
        """
        find_cursor = self._astra_db_collection.find(
            filter=find_query.get("filter"),
            sort=find_query.get("sort"),
            limit=find_query.get("limit"),
            include_similarity=find_query.get("includeSimilarity"),
            projection={"*": 1},
        )

        find_results = []
        for result in find_cursor:
            find_results.append(result)

        if not find_results:
            logger.warning("No documents found.")

        return find_results

    def find_one_document(self, find_query):
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

        def batch_generator(chunks, batch_size):
            for i in range(0, len(chunks), batch_size):
                i_end = min(len(chunks), i + batch_size)
                batch = chunks[i:i_end]
                yield batch

        for id_batch in batch_generator(ids, batch_size):
            docs = self.find_documents({"filter": {"_id": {"$in": id_batch}}})
            if docs:
                document_batch.extend(docs)

        formatted_docs = self._format_query_response(document_batch, include_metadata=True, include_values=True)

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
        ids: Optional[list[str]] = None,
        filters: Optional[dict[str, Union[str, float, int, bool, list, dict]]] = None,
    ) -> int:
        """Delete documents from the Astra index.

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

    def count_documents(self, upper_bound: int = 10000) -> int:
        """
        Count the number of documents in the Astra index.
        :returns: the number of documents in the index
        """
        return self._astra_db_collection.count_documents({}, upper_bound=upper_bound)

    def update(
        self,
        *,
        filters: dict[str, Union[str, float, int, bool, list, dict]],
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
