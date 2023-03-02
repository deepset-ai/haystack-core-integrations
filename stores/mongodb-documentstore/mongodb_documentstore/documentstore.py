from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Generator

import time
import logging
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from pymongo import MongoClient
from bson.objectid import ObjectId

from haystack.schema import Document, Label
from haystack.errors import DuplicateDocumentError, DocumentStoreError
from haystack.document_stores import BaseDocumentStore
from haystack.document_stores.base import get_batches_from_generator
from haystack.modeling.utils import initialize_device_settings
from haystack.document_stores.filter_utils import LogicalFilterClause

if TYPE_CHECKING:
    from haystack.nodes.retriever import BaseRetriever

logger = logging.getLogger(__name__)


class MongoDBDocumentStore(BaseDocumentStore):
    """
    MongoDB document store
    """

    def __init__(
        self,
        database_name: str = "haystack",
        documents_collection: str = "documents",
        labels_collection: str = "labels",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
    ):
        super().__init__()

        self.client = MongoClient()
        self.db = self.client[database_name]

        self.documents_collection: str = documents_collection
        self.labels_collection: str = labels_collection
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

    def _create_document_field_map(self) -> Dict:
        pass

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection_name = index or self.documents_collection
        collection = self.db[collection_name]

        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        # convert to Documents if needed
        documents_objects = [
            Document.from_dict(d) if isinstance(d, dict) else d for d in documents
        ]

        # Drop duplicates from the input
        documents_objects = self._drop_duplicate_documents(
            documents=documents_objects, index=collection_name
        )

        # Apply duplicate_documents strategy to documents present in the collection
        for document in documents_objects:
            if collection.find_one({"id": document.id}):
                if duplicate_documents == "fail":
                    raise DuplicateDocumentError(
                        f"Document with id '{document.id} already exists in collection {collection.name}"
                    )
                if duplicate_documents == "skip":
                    logger.warning(
                        f"Document with id '{document.id} already exists in collection {collection.name}"
                    )
                    continue
            collection.insert_one(document.to_dict())

    def write_labels(
        self,
        labels: Union[List[dict], List[Label]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Write annotation labels into document store.
        """
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection = self.db[index or self.labels_collection]

        # Convert to Labels if needed
        label_objects = [
            Label.from_dict(l) if isinstance(l, dict) else l for l in labels
        ]

        # Look for duplicates
        duplicate_ids: list = [
            label.id
            for label in self._get_duplicate_labels(
                label_objects, index=self.labels_collection
            )
        ]
        if len(duplicate_ids) > 0:
            logger.warning(
                f"Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                f" This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                f" the answer annotation and not the question."
                f" Problematic ids: {','.join(duplicate_ids)}"
            )

        for label in label_objects:
            # create timestamps if not available yet
            if not label.created_at:
                label.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if not label.updated_at:
                label.updated_at = label.created_at

            collection.insert_one(label.to_dict())

    def get_document_by_id(
        self,
        id: str,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Document]:
        """
        Fetch a document by specifying its text id string.
        """
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection = self.db[index or self.documents_collection]
        return collection.find_one({"_id": ObjectId(id)})

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None) -> List[Document]:  # type: ignore
        """
        Fetch documents by specifying a list of text id strings.
        """
        collection = self.db[index or self.documents_collection]
        documents = []
        for id in ids:
            doc = collection.find_one({"_id": ObjectId(id)})
            if doc:
                documents.append(doc)

        return documents

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        return []

    def get_document_count(
        self,
        filters: Optional[Any] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the number of documents in the document store.

        FIXME: FILTERS ARE NOT SUPPORTED!
        """
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection = self.db[index or self.documents_collection]
        return collection.count_documents({})

    def update_document_meta(self, id: str, meta: Dict[str, Any], index: str = None):
        """
        Update the metadata dictionary of a document by specifying its string id.

        :param id: The ID of the Document whose metadata is being updated.
        :param meta: A dictionary with key-value pairs that should be added / changed for the provided Document ID.
        :param index: Name of the index the Document is located at.
        """
        collection = self.db[index or self.documents_collection]
        doc = collection.find_one({"_id": ObjectId(id)})
        for key, value in meta.items():
            doc.meta[key] = value

        collection.update_one({"_id": ObjectId(id)}, {"$set": doc}, upsert=False)

    def get_label_count(
        self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Return the number of labels in the document store.
        """
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection = self.db[index or self.labels_collection]
        return collection.count_documents({})

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Any] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        result = self.get_all_documents_generator(
            index=index,
            filters=filters,
            return_embedding=return_embedding,
            batch_size=batch_size,
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Any] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection = self.db[index or self.documents_collection]
        for doc in collection.find():
            yield Document.from_dict(doc)

    def get_all_labels(
        self,
        index: str = None,
        filters: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        """
        Return all labels in the document store.
        """
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection = self.db[index or self.labels_collection]
        return [Label.from_dict(d) for d in collection.find()]

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        logger.warning(
            """DEPRECATION WARNINGS:
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, filters, headers)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection = self.db[index or self.documents_collection]

        if not filters and not ids:
            collection.drop()
            return

        docs_to_delete = self.get_all_documents(index=index, filters=filters)
        if ids:
            docs_to_delete = [doc for doc in docs_to_delete if doc.id in ids]

        for doc in docs_to_delete:
            collection.delete_one({"_id": ObjectId(doc.id)})

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        self.db[index or self.documents_collection].drop()

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers:
            raise NotImplementedError(f"{self.__class__} does not support headers.")

        collection = self.db[index or self.labels_collection]
        if not filters and not ids:
            collection.drop()
            return

        labels_to_delete = self.get_all_labels(index=index, filters=filters)
        if ids:
            labels_to_delete = [label for label in labels_to_delete if label.id in ids]

        for label in labels_to_delete:
            collection.delete_one({"_id": ObjectId(label.id)})
