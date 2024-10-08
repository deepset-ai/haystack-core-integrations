import logging
import re
from typing import Any, Dict, List, Optional, Union

from haystack import default_from_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from pymongo import InsertOne, MongoClient, ReplaceOne, UpdateOne
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo
from pymongo.errors import BulkWriteError

from integrations.azure_cosmos_db_mongo_vcore.src.haystack_integrations.document_stores.mongodb_vcore.filters import _normalize_filters

logger = logging.getLogger(__name__)


class AzureCosmosDBMongoVCoreDocumentStore:
    """
    AzureCosmosDBMongoVCoreDocumentStore is a DocumentStore implementation that uses
    [Azure CosmosDB Mongo vCore](https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search#filtered-vector-search-preview)
    service.

    To connect to Azure CosmosDB Mongo vCore, you need to provide a connection string in the format:
    `"mongodb+srv://{mongo_vcore_username}:{mongo_vcore_password}@{mongo_vcore_host}/?{mongo_vcore_params_string}"`.

    This connection string can be obtained from the Azure portal cosmos db account keys tab. The connection string
    can be provided as an environment variable `AZURE_COSMOS_MONGO_CONNECTION_STRING` or directly as a parameter to the
    `AzureCosmosDBMongoVCoreDocumentStore` constructor.

    After providing the connection string, you'll need to specify the `database_name` and `collection_name` to use.
    AzureCosmosDBMongoVCoreDocumentStore has an implementation for creating a collection if one is not created already.

    You need to provide a `vector_search_index_name` for the name of the vector which would be created when the colleciton
    is being created.

    The last parameter users need to provide is a `vector_search_kwargs` - used for configs for vector search in mongo vCore.
    {
        "dimensions": 1536,
        "num_lists": 1,
        "similarity": "COS",
        "kind": "vector-hnsw",
        "m": 2,
        "ef_construction": 64,
        "ef_search": 40
    }
    """
    def __init__(
        self,
        *,
        mongo_connection_string: Secret = Secret.from_env_var("AZURE_COSMOS_MONGO_CONNECTION_STRING"),
        database_name: str,
        collection_name: str,
        vector_search_index_name: str,
        vector_search_kwargs: dict[str, Any],
    ):
        """
        Creates a new MongoDBAtlasDocumentStore instance.

        :param mongo_connection_string: MongoDB Atlas connection string in the format:
            `"mongodb+srv://{mongo_vcore_username}:{mongo_vcore_password}@{mongo_vcore_host}/?{mongo_vcore_params_string}"`.
        :param database_name: Name of the database to use.
        :param collection_name: Name of the collection to use. To use this document store for embedding retrieval,
            this collection needs to have a vector search index set up on the `embedding` field.
        :param vector_search_index_name: The name of the vector search index to use for vector search operations.
        :param vector_search_kwargs: Configs for vector search in mongo vCore

        :raises ValueError: If the collection name contains invalid characters.
        """
        if collection_name and not bool(re.match(r"^[a-zA-Z0-9\-_]+$", collection_name)):
            msg = f'Invalid collection name: "{collection_name}". It can only contain letters, numbers, -, or _.'
            raise ValueError(msg)

        self.mongo_connection_string = mongo_connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.vector_search_index_name = vector_search_index_name
        self.vector_search_kwargs = vector_search_kwargs

        self._mongo_client = MongoClient(
            self.mongo_connection_string.resolve_value(),
            appname="HayStack-CDBMongoVCore-DocumentStore-Python",
            driver=DriverInfo(name="AzureCosmosDBMongoVCoreHayStackIntegration")
        )

        self._collection = self._create_collection_and_index()

    def _create_collection_and_index(self) -> Collection:
        database = self._mongo_client[self.database_name]
        if self.collection_name not in database.list_collection_names():
            # check the kind of vector search to be performed
            # prepare the command accordingly
            create_index_commands = {}
            if self.vector_search_kwargs.get("kind") == "vector-ivf":
                create_index_commands = self._get_vector_index_ivf()
            elif self.vector_search_kwargs.get("kind") == "vector-hnsw":
                create_index_commands = self._get_vector_index_hnsw()
            database.command(create_index_commands)
        return database[self.collection_name]

    def create_filter_index(
            self, property_to_filter: str,
    ) -> dict[str, Any]:
        command = {
            "createIndexes": self.collection_name,
            "indexes": [
                {
                    "key": {property_to_filter: 1},
                    "name": self.vector_search_index_name,
                }
            ],
        }

        create_index_response: dict[str, Any] = self.mongo_client[self.database_name].command(command)
        return create_index_response

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureCosmosDBMongoVCoreDocumentStore":
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
        return self._collection.count_documents({})

    def delete_documents(self, document_ids: Optional[List[str]] = None, delete_all: Optional[bool] = None) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        :param delete_all: if `True`, delete all documents.
        """
        if document_ids is not None:
            self._collection.delete_many(filter={"id": {"$in": document_ids}})
        elif delete_all:
            self._collection.delete_many({})

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
                        "Document %s has the `sparse_embedding` field set,"
                        "but storing sparse embeddings in MongoDB Atlas is not currently supported."
                        "The `sparse_embedding` field will be ignored.",
                        doc.id,
                    )
            mongo_documents.append(doc_dict)
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

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the Haystack [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: The filters to apply. It returns only the documents that match the filters.
        :returns: A list of Documents that match the given filters.
        """
        filters = _normalize_filters(filters) if filters else None
        documents = list(self._collection.find(filters))
        for doc in documents:
            doc.pop("_id", None)  # MongoDB's internal id doesn't belong into a Haystack document, so we remove it.
        return [Document.from_dict(doc) for doc in documents]

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
        pipeline: list[dict[str, Any]] = []
        if self.vector_search_kwargs.get("kind") == "vector-ivf":
            pipeline = self._get_pipeline_vector_ivf(query_embedding, top_k, filters)
        elif self.vector_search_kwargs.get("kind") == "vector-hnsw":
            pipeline = self._get_pipeline_vector_hnsw(query_embedding, top_k, filters)

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

    def _get_vector_index_ivf(self) -> dict[str, Any]:
        return {
            "createIndexes": self.collection_name,
            "indexes": [
                {
                    "name": self.vector_search_index_name,
                    "key": {"embedding": "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": self.vector_search_kwargs.get("kind"),
                        "numLists": self.vector_search_kwargs.get("num_lists"),
                        "similarity": self.vector_search_kwargs.get("similarity"),
                        "dimensions": self.vector_search_kwargs.get("dimensions"),
                    },
                }
            ],
        }

    def _get_vector_index_hnsw(self) -> dict[str, Any]:
        return {
            "createIndexes": self.collection_name,
            "indexes": [
                {
                    "name": self.vector_search_index_name,
                    "key": {"embedding": "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": self.vector_search_kwargs.get("kind"),
                        "m": self.vector_search_kwargs.get("m"),
                        "efConstruction": self.vector_search_kwargs.get("ef_construction"),
                        "similarity": self.vector_search_kwargs.get("similarity"),
                        "dimensions": self.vector_search_kwargs.get("dimensions"),
                    },
                }
            ],
        }

    def _get_pipeline_vector_ivf(
            self, embeddings: list[float], top_k: int, filters: Optional[dict]
    ) -> list[dict[str, Any]]:
        params = {
            "vector": embeddings,
            "path": "embedding",
            "k": top_k,
        }
        if filters:
            params["filter"] = filters

        pipeline: list[dict[str, Any]] = [
            {
                "$search": {
                    "cosmosSearch": params,
                    "returnStoredSource": True,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            },
        ]
        return pipeline

    def _get_pipeline_vector_hnsw(
            self, embeddings: list[float], top_k: int, filters: Optional[dict]
    ) -> list[dict[str, Any]]:
        params = {
            "vector": embeddings,
            "path": "embedding",
            "k": top_k,
            "efSearch": self.vector_search_kwargs.get("ef_search"),
        }
        if filters:
            params["filter"] = filters

        pipeline: list[dict[str, Any]] = [
            {
                "$search": {
                    "cosmosSearch": params,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            },
        ]
        return pipeline

    def _mongo_doc_to_haystack_doc(self, mongo_doc: Dict[str, Any]) -> Document:
        """
        Converts the dictionary coming out of MongoDB into a Haystack document

        :param mongo_doc: A dictionary representing a document as stored in MongoDB
        :returns: A Haystack Document object
        """
        mongo_doc.pop("_id", None)
        return Document.from_dict(mongo_doc)
