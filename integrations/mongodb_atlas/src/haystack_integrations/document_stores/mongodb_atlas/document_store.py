# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import re
from typing import Any, Dict, List, Optional, Union

from haystack import default_to_dict, default_from_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.mongodb_atlas.filters import haystack_filters_to_mongo
from pymongo import InsertOne, MongoClient, ReplaceOne, UpdateOne  # type: ignore
from pymongo.driver_info import DriverInfo  # type: ignore
from pymongo.errors import BulkWriteError  # type: ignore

logger = logging.getLogger(__name__)


class MongoDBAtlasDocumentStore:
    def __init__(
        self,
        *,
        mongo_connection_string: Secret = Secret.from_env_var("MONGO_CONNECTION_STRING"),  # noqa: B008
        database_name: str,
        collection_name: str,
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
        :param recreate_collection: Whether to recreate the collection when initializing the document store.
        """
        if collection_name and not bool(re.match(r"^[a-zA-Z0-9\-_]+$", collection_name)):
            msg = f'Invalid collection name: "{collection_name}". It can only contain letters, numbers, -, or _.'
            raise ValueError(msg)
        
        resolved_connection_string = mongo_connection_string.resolve_value()
        if resolved_connection_string is None:
            msg = (
                "MongoDBAtlasDocumentStore expects an API key. "
                "Set the MONGO_CONNECTION_STRING environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)
        self.mongo_connection_string = mongo_connection_string

        self.database_name = database_name
        self.collection_name = collection_name
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
            recreate_collection=self.recreate_collection,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasDocumentStore":
        """
        Utility function that deserializes this Document Store's configuration from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["mongo_connection_string"])
        return default_from_dict(cls, data)

    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Returns how many documents are present in the document store.

        :param filters: The filters to apply. It counts only the documents that match the filters.
        """
        return self.collection.count_documents(filters or {})

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
