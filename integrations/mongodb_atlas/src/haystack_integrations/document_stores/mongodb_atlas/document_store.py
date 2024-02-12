# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union

import re
import logging

import pymongo
from pymongo import InsertOne, ReplaceOne, UpdateOne
from pymongo.driver_info import DriverInfo
from haystack import default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.filters import convert

from haystack_integrations.document_stores.mongodb_atlas.filters import haystack_filters_to_mongo


logger = logging.getLogger(__name__)


METRIC_TYPES = ["euclidean", "cosine", "dotProduct"]


class MongoDBAtlasDocumentStore:
    def __init__(
        self,
        mongo_connection_string: str,
        database_name: str,
        collection_name: str,
        vector_search_index: Optional[str] = None,
        embedding_dim: int = 768,
        similarity: str = "cosine",
        embedding_field: str = "embedding",
        recreate_index: bool = False,
    ):
        """
        Creates a new MongoDBAtlasDocumentStore instance.

        This Document Store uses MongoDB Atlas as a backend (https://www.mongodb.com/docs/atlas/getting-started/).

        :param mongo_connection_string: MongoDB Atlas connection string in the format: 
            "mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}".
            This can be obtained on the MongoDB Atlas Dashboard by clicking on the `CONNECT` button.
        :param database_name: Name of the database to use.
        :param collection_name: Name of the collection to use.
        :param vector_search_index: The name of the index to use for vector search. To use the search index it must have been created in the Atlas web UI before. None by default.
        :param embedding_dim: Dimensionality of embeddings, 768 by default.
        :param similarity: The similarity function to use for the embeddings. One of "euclidean", "cosine" or "dotProduct". "cosine" is the default.
        :param embedding_field: The name of the field in the document that contains the embedding.
        :param recreate_index: Whether to recreate the index when initializing the document store.
        """
        if similarity not in METRIC_TYPES:
            raise ValueError(
                "MongoDB Atlas currently supports dotProduct, cosine and euclidean metrics. Please set similarity to one of the above."
            )
        if collection_name and not bool(re.match(r"^[a-zA-Z0-9\-_]+$", collection_name)):
            raise ValueError(
                f'Invalid collection name: "{collection_name}". Index name can only contain letters, numbers, hyphens, or underscores.'
            )
        
        self.mongo_connection_string = mongo_connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.connection: pymongo.MongoClient = pymongo.MongoClient(
            self.mongo_connection_string, driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
        )
        self.database = self.connection[self.database_name]
        
        self.similarity = similarity
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.index = collection_name
        self.recreate_index = recreate_index
        self.vector_search_index = vector_search_index

        if self.recreate_index:
            self.delete_index()

        # Implicitly create the collection if it doesn't exist
        if collection_name not in self.database.list_collection_names():
            self.database.create_collection(self.collection_name)
            self._get_collection().create_index("id", unique=True)

    def _create_document_field_map(self) -> Dict:
        return {self.embedding_field: "embedding"}

    def _get_collection(self, index=None) -> pymongo.collection.Collection:
        """
        Returns the collection named by index or returns the collection specified when the
        driver was initialized.
        """
        _validate_index_name(index)
        if index is not None:
            return self.database[index]
        else:
            return self.database[self.collection_name]

    def to_dict(self) -> Dict[str, Any]:
        """
        Utility function that serializes this Document Store's configuration into a dictionary.
        """
        return default_to_dict(
            self,
            mongo_connection_string=self.mongo_connection_string,
            database_name=self.database_name,
            collection_name=self.collection_name,
            vector_search_index=self.vector_search_index,
            embedding_dim=self.embedding_dim,
            similarity=self.similarity,
            embedding_field=self.embedding_field,
            recreate_index=self.recreate_index,
        )

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        collection = self._get_collection(self.index)
        return collection.count_documents()
    
    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :return: A list of Documents that match the given filters.
        """
        mongo_filters = haystack_filters_to_mongo(filters)
        collection = self._get_collection(self.index)
        documents = collection.find(mongo_filters)
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

        collection = self._get_collection(self.index)
        field_map = self._create_document_field_map()
        mongo_documents = [doc.to_dict() for doc in documents]
        operations: List[Union[UpdateOne, InsertOne, ReplaceOne]]
        written_docs = len(documents)

        if policy == DuplicatePolicy.SKIP:
            operations = [UpdateOne({"id": doc["id"]}, {"$setOnInsert": doc}, upsert=True) for doc in mongo_documents]
            existing_documents = collection.count_documents({"id": {"$in": [doc.id for doc in documents]}})
            written_docs -= existing_documents
        elif policy == DuplicatePolicy.FAIL:
            operations = [InsertOne(doc) for doc in mongo_documents]
        else:
            operations = [ReplaceOne({"id": doc["id"]}, upsert=True, replacement=doc) for doc in mongo_documents]

        collection.bulk_write(operations)

        return written_docs

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return
        collection = self._get_collection(self.index)
        collection.delete_many(filter={"id": {"$in": document_ids}})




