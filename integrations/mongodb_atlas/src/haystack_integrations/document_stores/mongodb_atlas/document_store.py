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
from pymongo import InsertOne, MongoClient, ReplaceOne, UpdateOne
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
        self._connection: Optional[MongoClient] = None
        self._collection: Optional[Collection] = None

    @property
    def connection(self) -> MongoClient:
        if self._connection is None:
            self._connection = MongoClient(
                self.mongo_connection_string.resolve_value(), driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
            )

        return self._connection

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            database = self.connection[self.database_name]

            if self.collection_name not in database.list_collection_names():
                msg = f"Collection '{self.collection_name}' does not exist in database '{self.database_name}'."
                raise ValueError(msg)
            self._collection = database[self.collection_name]
        return self._collection

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
        return self.collection.count_documents({})

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the Haystack [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: The filters to apply. It returns only the documents that match the filters.
        :returns: A list of Documents that match the given filters.
        """
        filters = _normalize_filters(filters) if filters else None
        documents = list(self.collection.find(filters))
        for doc in documents:
            doc.pop("_id", None)  # MongoDB's internal id doesn't belong into a Haystack document, so we remove it.
        return [Document.from_dict(doc) for doc in documents]

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

        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        mongo_documents = []
        for doc in documents:
            doc_dict = doc.to_dict(flatten=False)
            if "sparse_embedding" in doc_dict:
                sparse_embedding = doc_dict.pop("sparse_embedding", None)
                if sparse_embedding:
                    logger.warning(
                        "Document {id} has the `sparse_embedding` field set,"
                        "but storing sparse embeddings in MongoDB Atlas is not currently supported."
                        "The `sparse_embedding` field will be ignored.",
                        id=doc.id,
                    )
            mongo_documents.append(doc_dict)
        operations: List[Union[UpdateOne, InsertOne, ReplaceOne]]
        written_docs = len(documents)

        if policy == DuplicatePolicy.SKIP:
            operations = [UpdateOne({"id": doc["id"]}, {"$setOnInsert": doc}, upsert=True) for doc in mongo_documents]
            existing_documents = self.collection.count_documents({"id": {"$in": [doc.id for doc in documents]}})
            written_docs -= existing_documents
        elif policy == DuplicatePolicy.FAIL:
            operations = [InsertOne(doc) for doc in mongo_documents]
        else:
            operations = [ReplaceOne({"id": doc["id"]}, upsert=True, replacement=doc) for doc in mongo_documents]

        try:
            self.collection.bulk_write(operations)
        except BulkWriteError as e:
            msg = f"Duplicate documents found: {e.details['writeErrors']}"
            raise DuplicateDocumentError(msg) from e

        return written_docs

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return
        self.collection.delete_many(filter={"id": {"$in": document_ids}})

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
        if not query_embedding:
            msg = "Query embedding must not be empty"
            raise ValueError(msg)

        filters = _normalize_filters(filters) if filters else {}

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_search_index,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k,
                    "filter": filters,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "content": 1,
                    "blob": 1,
                    "meta": 1,
                    "embedding": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        try:
            documents = list(self.collection.aggregate(pipeline))
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
        text_search: Dict[str, Any] = {"path": "content", "query": query}
        if match_criteria:
            text_search["matchCriteria"] = match_criteria
        if synonyms:
            text_search["synonyms"] = synonyms
        if fuzzy:
            text_search["fuzzy"] = fuzzy
        if score:
            text_search["score"] = score

        # Define the pipeline for MongoDB aggregation
        pipeline = [
            {
                "$search": {
                    "index": self.full_text_search_index,
                    "compound": {"must": [{"text": text_search}]},
                }
            },
            # TODO: Use compound filter. See: (https://www.mongodb.com/docs/atlas/atlas-search/performance/query-performance/#avoid--match-after--search)
            {"$match": filters},
            {"$limit": top_k},
            {
                "$project": {
                    "_id": 0,
                    "content": 1,
                    "blob": 1,
                    "meta": 1,
                    "embedding": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]

        try:
            documents = list(self.collection.aggregate(pipeline))
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
        mongo_doc.pop("_id", None)
        return Document.from_dict(mongo_doc)
