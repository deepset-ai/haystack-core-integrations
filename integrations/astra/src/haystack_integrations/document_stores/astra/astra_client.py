import json
import logging
from typing import Dict, List, Optional, Union
from warnings import warn

from astrapy.api import APIRequestError
from astrapy.db import AstraDB
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
    matches: List[Response]

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

        # Build the Astra DB object
        self._astra_db = AstraDB(
            api_endpoint=api_endpoint,
            token=token,
            namespace=namespace,
            caller_name=CALLER_NAME,
            caller_version=integration_version,
        )

        indexing_options = {"indexing": {"deny": NON_INDEXED_FIELDS}}
        try:
            # Create and connect to the newly created collection
            self._astra_db_collection = self._astra_db.create_collection(
                collection_name=collection_name,
                dimension=embedding_dimension,
                options=indexing_options,
            )
        except APIRequestError:
            # possibly the collection is preexisting and has legacy
            # indexing settings: verify
            get_coll_response = self._astra_db.get_collections(options={"explain": True})

            collections = (get_coll_response["status"] or {}).get("collections") or []

            preexisting = [collection for collection in collections if collection["name"] == collection_name]

            if preexisting:
                pre_collection = preexisting[0]
                # if it has no "indexing", it is a legacy collection;
                # otherwise it's unexpected warn and proceed at user's risk
                pre_col_options = pre_collection.get("options") or {}
                if "indexing" not in pre_col_options:
                    warn(
                        (
                            f"Astra DB collection '{collection_name}' is "
                            "detected as having indexing turned on for all "
                            "fields (either created manually or by older "
                            "versions of this plugin). This implies stricter "
                            "limitations on the amount of text each string in a "
                            "document can store. Consider indexing anew on a "
                            "fresh collection to be able to store longer texts. "
                            "See https://github.com/deepset-ai/haystack-core-"
                            "integrations/blob/main/integrations/astra/README"
                            ".md#warnings-about-indexing for more details."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    self._astra_db_collection = self._astra_db.collection(
                        collection_name=collection_name,
                    )
                elif pre_col_options["indexing"] != indexing_options["indexing"]:
                    detected_options_json = json.dumps(pre_col_options["indexing"])
                    indexing_options_json = json.dumps(indexing_options["indexing"])
                    warn(
                        (
                            f"Astra DB collection '{collection_name}' is "
                            "detected as having the following indexing policy: "
                            f"{detected_options_json}. This does not match the requested "
                            f"indexing policy for this object: {indexing_options_json}. "
                            "In particular, there may be stricter "
                            "limitations on the amount of text each string in a "
                            "document can store. Consider indexing anew on a "
                            "fresh collection to be able to store longer texts. "
                            "See https://github.com/deepset-ai/haystack-core-"
                            "integrations/blob/main/integrations/astra/README"
                            ".md#warnings-about-indexing for more details."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    self._astra_db_collection = self._astra_db.collection(
                        collection_name=collection_name,
                    )
                else:
                    # the collection mismatch lies elsewhere than the indexing
                    raise
            else:
                # other exception
                raise

    def query(
        self,
        vector: Optional[List[float]] = None,
        query_filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
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
        query = {"filter": filters, "options": {"limit": top_k}}

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
        query = {"sort": {"$vector": vector}, "options": {"limit": top_k, "includeSimilarity": True}}

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
        response_dict = self._astra_db_collection.find(
            filter=find_query.get("filter"),
            sort=find_query.get("sort"),
            options=find_query.get("options"),
            projection={"*": 1},
        )

        if "data" in response_dict and "documents" in response_dict["data"]:
            return response_dict["data"]["documents"]
        else:
            logger.warning(f"No documents found: {response_dict}")

    def get_documents(self, ids: List[str], batch_size: int = 20) -> QueryResponse:
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

    def insert(self, documents: List[Dict]):
        """
        Insert documents into the Astra index.

        :param documents: a list of documents to insert
        :returns: the IDs of the inserted documents
        """
        response_dict = self._astra_db_collection.insert_many(documents=documents)

        inserted_ids = (
            response_dict["status"]["insertedIds"]
            if "status" in response_dict and "insertedIds" in response_dict["status"]
            else []
        )
        if "errors" in response_dict:
            logger.error(response_dict["errors"])

        return inserted_ids

    def update_document(self, document: Dict, id_key: str):
        """
        Update a document in the Astra index.

        :param document: the document to update
        :param id_key: the key to use as the document id
        :returns: whether the document was updated successfully
        """
        document_id = document.pop(id_key)

        response_dict = self._astra_db_collection.find_one_and_update(
            filter={id_key: document_id},
            update={"$set": document},
            options={"returnDocument": "after"},
            projection={"*": 1},
        )

        document[id_key] = document_id

        if "status" in response_dict and "errors" not in response_dict:
            if "matchedCount" in response_dict["status"] and "modifiedCount" in response_dict["status"]:
                if response_dict["status"]["matchedCount"] == 1 and response_dict["status"]["modifiedCount"] == 1:
                    return True

        logger.warning(f"Documents {document_id} not updated in Astra DB.")

        return False

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        filters: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
    ) -> int:
        """Delete documents from the Astra index.

        :param ids: the ids of the documents to delete
        :param delete_all: if `True`, delete all documents from the index
        :param filters: additional filters to apply when deleting documents
        :returns: the number of documents deleted
        """
        if delete_all:
            query = {"deleteMany": {}}  # type: dict
        if ids is not None:
            query = {"deleteMany": {"filter": {"_id": {"$in": ids}}}}
        if filters is not None:
            query = {"deleteMany": {"filter": filters}}

        filter_dict = {}
        if "filter" in query["deleteMany"]:
            filter_dict = query["deleteMany"]["filter"]

        deletion_counter = 0
        moredata = True
        while moredata:
            response_dict = self._astra_db_collection.delete_many(filter=filter_dict)

            if "moreData" not in response_dict.get("status", {}):
                moredata = False

            deletion_counter += int(response_dict["status"].get("deletedCount", 0))

        return deletion_counter

    def count_documents(self) -> int:
        """
        Count the number of documents in the Astra index.
        :returns: the number of documents in the index
        """
        documents_count = self._astra_db_collection.count_documents()

        return documents_count["status"]["count"]
