# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from haystack.dataclasses import Document
from haystack.document_stores import (
    DuplicateDocumentError,
    DuplicatePolicy,
    MissingDocumentError,
)
from pydantic import validate_arguments

from astra_store.astra_client import AstraClient
from astra_store.errors import AstraDocumentStoreFilterError
from astra_store.filters import _convert_filters

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _batches(input_list, batch_size):
    input_length = len(input_list)
    for ndx in range(0, input_length, batch_size):
        yield input_list[ndx : min(ndx + batch_size, input_length)]


class AstraDocumentStore:
    """
    An AstraDocumentStore document store for Haystack.
    """

    @validate_arguments
    def __init__(
        self,
        astra_id: str,
        astra_region: str,
        astra_application_token: str,
        astra_keyspace: str,
        astra_collection: str,
        embedding_dim: int,
        duplicates_policy: Optional[DuplicatePolicy] = None,
        similarity: str = "cosine",
    ):
        """
        The connection to Astra DB is established and managed through the JSON API.
        The required credentials (database ID, region, and application token) can be generated
        through the UI by clicking and the connect tab, and then selecting JSON API and
        Generate Configuration.

        :param astra_id: id of the Astra DB instance.
        :param astra_region: Region of cloud servers (can be found when generating the token).
        :param astra_application_token: the connection token for Astra.
        :param astra_keyspace: The keyspace for the current Astra DB.
        :param astra_collection: The current collection in the keyspace in the current Astra DB.
        :param embedding_dim: Dimension of embedding vector.
        :param similarity: The similarity function used to compare document vectors.
        :param duplicates_policy: Handle duplicate documents based on DuplicatePolicy parameter options.
                                  Parameter options : (SKIP,OVERWRITE,FAIL)
                                  skip: Ignore the duplicates documents
                                  overwrite: Update any existing documents with the same ID when adding documents.
                                  fail: an error is raised if the document ID of the document being added already
                                  exists.
        """

        self.duplicates_policy = duplicates_policy
        self.astra_id = astra_id
        self.astra_region = astra_region
        self.astra_application_token = astra_application_token
        self.astra_keyspace = astra_keyspace
        self.astra_collection = astra_collection
        self.embedding_dim = embedding_dim
        self.similarity = similarity

        self.index = AstraClient(
            astra_id=self.astra_id,
            astra_region=self.astra_region,
            astra_application_token=self.astra_application_token,
            keyspace_name=self.astra_keyspace,
            collection_name=self.astra_collection,
            embedding_dim=self.embedding_dim,
            similarity_function=self.similarity,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AstraDocumentStore":
        return AstraDocumentStore(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duplicates_policy": self.duplicates_policy,
            "astra_id": self.astra_id,
            "astra_region": self.astra_region,
            "astra_application_token": self.astra_application_token,
            "astra_keyspace": self.astra_keyspace,
            "astra_collection": self.astra_collection,
            "embedding_dim": self.embedding_dim,
            "similarity": self.similarity,
        }

    def write_documents(
        self,
        documents: List[Document],
        index: Optional[str] = None,
        batch_size: Optional[int] = 20,
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ):
        """
        Indexes documents for later queries.

        :param documents: a list of Haystack Document objects.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param policy:  Handle duplicate documents based on DuplicatePolicy parameter options.
                        Parameter options : (SKIP,OVERWRITE,FAIL)
                        skip: Ignore the duplicates documents
                        overwrite: Update any existing documents with the same ID when adding documents.
                        fail: an error is raised if the document ID of the document being added already
                        exists.
        :return: int
        """

        if index is None:
            index = self.index

        if policy is None or policy == DuplicatePolicy.NONE:
            if self.duplicates_policy is not None and self.duplicates_policy != DuplicatePolicy.NONE:
                policy = self.duplicates_policy
            else:
                policy = DuplicatePolicy.SKIP

        if batch_size > 20:
            logger.warning(
                f"batch_size set to {batch_size}, "
                f"but maximum batch_size for Astra when using the JSON API is 20. batch_size set to 20."
            )
            batch_size = 20

        def _convert_input_document(document: Union[dict, Document]):
            if isinstance(document, Document):
                document_dict = asdict(document)
            elif isinstance(document, dict):
                document_dict = document
            else:
                raise ValueError(f"Unsupported type for documents, documents is of type {type(document)}.")

            if "id" in document_dict:
                if "_id" not in document_dict:
                    document_dict["_id"] = document_dict.pop("id")
                elif "_id" in document_dict:
                    raise Exception(
                        f"Duplicate id definitions, both 'id' and '_id' present in document {document_dict}"
                    )
            if "_id" in document_dict:
                if not isinstance(document_dict["_id"], str):
                    raise Exception(
                        f"Document id {document_dict['_id']} is not a string, "
                        f"but is of type {type(document_dict['_id'])}"
                    )

            if "dataframe" in document_dict and document_dict["dataframe"] is not None:
                document_dict["dataframe"] = document_dict.pop("dataframe").to_json()
            document_dict["$vector"] = document_dict.pop("embedding", None)

            return document_dict

        documents_to_write = [_convert_input_document(doc) for doc in documents]

        duplicate_documents = []
        new_documents = []
        i = 0
        while i < len(documents_to_write):
            doc = documents_to_write[i]
            response = self.index.find_documents({"filter": {"_id": doc["_id"]}})
            if response:
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"ID '{doc['_id']}' already exists.")
                duplicate_documents.append(doc)
            else:
                new_documents.append(doc)
            i = i + 1

        insertion_counter = 0
        if policy == DuplicatePolicy.SKIP:
            if len(new_documents) > 0:
                for batch in _batches(new_documents, batch_size):
                    inserted_ids = index.insert(batch)
                    insertion_counter += len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to SKIP")

        elif policy == DuplicatePolicy.OVERWRITE:
            if len(new_documents) > 0:
                for batch in _batches(new_documents, batch_size):
                    inserted_ids = index.insert(batch)
                    insertion_counter += len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to OVERWRITE")

            if len(duplicate_documents) > 0:
                updated_ids = []
                for duplicate_doc in duplicate_documents:
                    updated = index.update_document(duplicate_doc, "_id")
                    if updated:
                        updated_ids.append(duplicate_doc["_id"])
                insertion_counter = insertion_counter + len(updated_ids)
                logger.info(f"write_documents updated documents with id {updated_ids}")
            else:
                logger.info("No documents updated. Argument policy set to OVERWRITE")

        elif policy == DuplicatePolicy.FAIL:
            if len(new_documents) > 0:
                for batch in _batches(new_documents, batch_size):
                    inserted_ids = index.insert(batch)
                    insertion_counter = insertion_counter + len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to FAIL")

        return insertion_counter

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self.index.count_documents()

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Returns at most 1000 documents that match the filter

        Args:
            filters (Optional[Dict[str, Any]], optional): Filters to apply. Defaults to None.

        Raises:
            AstraDocumentStoreFilterError: If the filter is invalid or not supported by this class.

        Returns:
            List[Document]: A list of matching documents.
        """
        if not isinstance(filters, dict) and filters is not None:
            msg = "Filters must be a dictionary or None"
            raise AstraDocumentStoreFilterError(msg)

        if filters is not None:
            if "id" in filters:
                filters["_id"] = filters.pop("id")
        vector = None
        if filters is not None and "embedding" in filters.keys():
            if "$in" in filters["embedding"]:
                embeds = filters.pop("embedding")
                vectors = embeds["$in"]
            else:
                filters["$vector"] = filters.pop("embedding")
                vectors = [filters.pop("$vector")]
            documents = []
            for vector in vectors:
                converted_filters = _convert_filters(filters)
                results = self.index.query(
                    vector=vector, filter=converted_filters, top_k=1000, include_values=True, include_metadata=True
                )
                documents.extend(self._get_result_to_documents(results))
        else:
            converted_filters = _convert_filters(filters)
            results = self.index.query(
                vector=vector, filter=converted_filters, top_k=1000, include_values=True, include_metadata=True
            )
            documents = self._get_result_to_documents(results)
        return documents

    @staticmethod
    def _get_result_to_documents(results) -> List[Document]:
        documents = []
        for match in results.matches:
            dataframe = match.metadata.pop("dataframe", None)
            if dataframe is not None:
                df = pd.DataFrame.from_dict(json.loads(dataframe))
            else:
                df = None
            document = Document(
                content=match.text,
                id=match.id,
                embedding=match.values,
                dataframe=df,
                blob=match.metadata.pop("blob", None),
                meta=match.metadata.pop("meta", None),
                score=match.score,
            )
            documents.append(document)
        return documents

    def get_documents_by_id(self, ids: List[str]) -> List[Document]:
        """
        Returns documents with given ids.
        """
        results = self.index.get_documents(ids=ids)
        ret = self._get_result_to_documents(results)
        return ret

    def get_document_by_id(self, document_id: str) -> Document:
        """
        :param document_id: id of the document to retrieve
        Returns documents with given ids.
        Raises MissingDocumentError when document_id does not exist in document store
        """
        document = self.index.get_documents(ids=[document_id])
        ret = self._get_result_to_documents(document)
        if not ret:
            raise MissingDocumentError(f"Document {document_id} does not exist")
        return ret[0]

    def search(
        self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform a search for a list of queries.

        Args:
            queries (List[Union[str, Dict[str, float]]]): A list of queries.
            top_k (int): The number of results to return.
            filters (Optional[Dict[str, Any]], optional): Filters to apply during search. Defaults to None.

        Returns:
            List[List[Document]]: A list of matching documents for each query.
        """
        results = []
        converted_filters = _convert_filters(filters)

        result = self._get_result_to_documents(
            self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=converted_filters,
                include_metadata=True,
                include_values=True,
            )
        )
        results.append(result)
        logger.debug(f"Raw responses: {result}")  # leaving for debugging

        return results

    def delete_documents(self, document_ids: List[str] = None, delete_all: Optional[bool] = None) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param document_ids: the document_ids to delete.
        :param delete_all: delete all documents.
        """

        deletion_counter = 0
        if self.index.count_documents() > 0:
            if document_ids is not None:
                for batch in _batches(document_ids, 20):
                    deletion_counter += self.index.delete(ids=batch)
            else:
                deletion_counter = self.index.delete(delete_all=delete_all)
            logger.info(f"{deletion_counter} documents deleted")

            if document_ids is not None and deletion_counter == 0:
                raise MissingDocumentError(f"Document {document_ids} does not exist")
        else:
            logger.info("No documents in document store")
