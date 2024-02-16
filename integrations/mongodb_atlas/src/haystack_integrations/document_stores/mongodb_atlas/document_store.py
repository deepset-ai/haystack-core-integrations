# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack_integrations.document_stores.mongodb_atlas.filters import haystack_filters_to_mongo
from pymongo import InsertOne, MongoClient, ReplaceOne, UpdateOne  # type: ignore
from pymongo.driver_info import DriverInfo  # type: ignore
from pymongo.errors import BulkWriteError  # type: ignore

logger = logging.getLogger(__name__)


METRIC_TYPES = ["euclidean", "cosine", "dotProduct"]


class MongoDBAtlasDocumentStore:
    def __init__(
        self,
        *,
        mongo_connection_string: Secret = Secret.from_env_var("MONGO_CONNECTION_STRING"),  # noqa: B008
        database_name: str,
        collection_name: str,
        vector_search_index: str,
        recreate_collection: bool = False,
    ):
        """
        Creates a new MongoDBAtlasDocumentStore instance.

        This Document Store uses MongoDB Atlas as a backend (https://www.mongodb.com/docs/atlas/getting-started/).

        :param mongo_connection_string: MongoDB Atlas connection string in the format:
            "mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}".
            This can be obtained on the MongoDB Atlas Dashboard by clicking on the `CONNECT` button.
            This value will be read automatically from the env var "MONGO_CONNECTION_STRING".
        :param database_name: Name of the database to use.
        :param collection_name: Name of the collection to use.
        :param vector_search_index: The name of the vector search index to use for vector search operations.
            Create a vector_search_index in the Atlas web UI and specify the init params of MongoDBAtlasDocumentStore. \
            See https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#std-label-avs-create-index
        :param recreate_collection: Whether to recreate the collection when initializing the document store. Defaults
            to False.
        """
        if collection_name and not bool(re.match(r"^[a-zA-Z0-9\-_]+$", collection_name)):
            msg = f'Invalid collection name: "{collection_name}". It can only contain letters, numbers, -, or _.'
            raise ValueError(msg)

        resolved_connection_string = mongo_connection_string.resolve_value()
        self.mongo_connection_string = mongo_connection_string

        self.database_name = database_name
        self.collection_name = collection_name
        self.vector_search_index = vector_search_index
        self.recreate_collection = recreate_collection

        self.connection: MongoClient = MongoClient(
            resolved_connection_string, driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
        )
        database = self.connection[self.database_name]

        if self.recreate_collection and self.collection_name in database.list_collection_names():
            database[self.collection_name].drop()

        # Implicitly create the collection if it doesn't exist
        if collection_name not in database.list_collection_names():
            database.create_collection(self.collection_name)
            database[self.collection_name].create_index("id", unique=True)

        self.collection = database[self.collection_name]

    def to_dict(self) -> Dict[str, Any]:
        """
        Utility function that serializes this Document Store's configuration into a dictionary.
        """
        return default_to_dict(
            self,
            mongo_connection_string=self.mongo_connection_string.to_dict(),
            database_name=self.database_name,
            collection_name=self.collection_name,
            vector_search_index=self.vector_search_index,
            recreate_collection=self.recreate_collection,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasDocumentStore":
        """
        Utility function that deserializes this Document Store's configuration from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["mongo_connection_string"])
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self.collection.count_documents({})

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: The filters to apply. It returns only the documents that match the filters.
        :return: A list of Documents that match the given filters.
        """
        mongo_filters = haystack_filters_to_mongo(filters)
        documents = list(self.collection.find(mongo_filters))
        for doc in documents:
            doc.pop("_id", None)  # MongoDB's internal id doesn't belong into a Haystack document, so we remove it.
        return [Document.from_dict(doc) for doc in documents]

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents into to PgvectorDocumentStore.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to DuplicatePolicy.FAIL (or not specified).
        :return: The number of documents written to the document store.
        """

        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        mongo_documents = [doc.to_dict() for doc in documents]
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

    def embedding_retrieval(
        self,
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        similarity: str = "cosine",
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Find the documents that are most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query
        :param filters: optional filters (see get_all_documents for description).
        :param top_k: How many documents to return.
        :param similarity: The similarity function to use. Currently supported: `dotProduct`, `cosine` and `euclidean`.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a 
                            different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if similarity not in METRIC_TYPES:
            raise ValueError(
                "MongoDB Atlas currently supports 'dotProduct', 'cosine' and 'euclidean' similarity metrics. \
                Please set 'similarity' to one of the above."
            )
        
        query_embedding = np.array(query_embedding).astype(np.float32)

        if similarity == "cosine":
            self.normalize_embedding(query_embedding)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_search_index,
                    "queryVector": query_embedding.tolist(),
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": top_k,
                }
            }
        ]

        filters = haystack_filters_to_mongo(filters)
        if filters is not None:
            pipeline.append({"$match": filters})
            
        pipeline.append({"$set": {"score": {"$meta": "vectorSearchScore"}}})
        documents = list(self.collection.aggregate(pipeline))

        if scale_score:
            for doc in documents:
                doc["score"] = self.scale_to_unit_interval(doc["score"], similarity)

        documents = [self.mongo_doc_to_haystack_doc(doc) for doc in documents]
        return documents
    
    def mongo_doc_to_haystack_doc(mongo_doc: Dict[str, Any]) -> Document:
        """
        Converts the dictionary coming out of MongoDB into a Haystack document

        :param mongo_doc: A dictionary representing a document as stored in MongoDB
        :return: A Haystack Document object
        """
        mongo_doc.pop("_id", None)
        return Document.from_dict(mongo_doc)
    
    def normalize_embedding(self, emb: np.ndarray) -> None:
        """
        Performs L2 normalization of a 1D embeddings vector **inplace**. 
        """
        norm = np.sqrt(emb.dot(emb))  # faster than np.linalg.norm()
        if norm != 0.0:
            emb /= norm

    def scale_to_unit_interval(self, score: float, similarity: Optional[str]) -> float:
        if similarity == "cosine":
            return (score + 1) / 2
        return float(1 / (1 + np.exp(-score / 100)))
