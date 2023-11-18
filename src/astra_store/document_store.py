# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional, Literal
from pydantic import validate_arguments

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores.decorator import document_store
from haystack.preview.document_stores.errors import DuplicateDocumentError, MissingDocumentError
from haystack.preview.document_stores.protocols import DuplicatePolicy

from astra_client import AstraClient

from errors import AstraDocumentStoreFilterError

import json
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@document_store
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
        duplicate_documents: Literal["skip", "overwrite", "fail"],
        similarity: str = "cosine",
    ):
        """
        The connection to Astra DB is established and managed through the JSON API.
        The required credentials (databse ID, region, and application token) can be generated
        through the UI by clicking and the connect tab, and then selecting JSON API and
        Generate Configuration.

        :param astra_id: id of the Astra DB instance.
        :param astra_region: Region of cloud servers (can be found when generating the token).
        :param astra_application_token: the connection token for Astra.
        :param astra_keyspace: The keyspace for the current Astra DB.
        :param astra_collection: The current collection in the keyspace in the current Astra DB.
        :param embedding_dim: Dimension of embedding vector.
        :param similarity: The similarity function used to compare document vectors.
        :param duplicate_documents: Handle duplicate documents based on parameter options.\
            Parameter options:
                - `"skip"`: Ignore the duplicate documents.
                - `"overwrite"`: Update any existing documents with the same ID when adding documents.
                - `"fail"`: An error is raised if the document ID of the document being added already exists.
        """

        self.astra_id = astra_id
        self.astra_region = astra_region
        self.astra_application_token = astra_application_token
        self.astra_keyspace = astra_keyspace
        self.astra_collection = astra_collection
        self.embedding_dim = embedding_dim
        self.similarity = similarity
        self.duplicate_documents = duplicate_documents

        self.index = AstraClient(
            astra_id = self.astra_id,
            astra_region = self.astra_region,
            astra_application_token = self.astra_application_token,
            keyspace_name = self.astra_keyspace,
            collection_name = self.astra_collection,
            embedding_dim = self.embedding_dim,
            similarity_function = self.similarity
        )
    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        count = requests.request(
            "POST",
            self.index.request_url,
            headers=self.index.request_header,
            data=json.dumps({"countDocuments":{}}),
        ).json()["status"]["count"]
        return count

        return self._index.describe_index_stats()["total_document_count"]

    # def count_vectors(self) -> int:
    #     """
    #     Returns how many vectors are present in the document store.
    #     """
    #     return self._index.describe_index_stats()["total_vector_count"]

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

        filter_string = self._convert_filters(filters)
        results = self._index.query(filter=filter_string, top_k=1000, include_values=True, include_metadata=True)
        documents = self._get_documents(results)
        return documents

    def _get_result_to_documents(self, results, return_embedding) -> List[Document]:
        documents = []
        for res in results:
            id = res.pop("_id")
            metadata = res.pop("$vector")
            val = res
            if return_embedding:
                document = Document(
                    id=id,
                    text=val,
                    metadata=metadata,
                    mime_type="",
                    score=1,
                )
            else:
                document = Document(
                    id=id,
                    text=val,
                    mime_type="",
                    score=1,
                )
            documents.append(document)
        return documents

    def get_documents(self, ids: List[str]) -> List[Document]:
        query = {"find": {"filter": {"_id": ""}}}
        query["find"]["filter"] = {"_id": {"$in":ids}}
        documents = requests.request(
            "POST",
            self.index.request_url,
            headers=self.index.request_header,
            data=json.dumps(query),
        ).json()["data"]["documents"]
        return documents

    def get_documents_by_id(self, ids: List[str], return_embedding: Optional[bool] = None) -> List[Document]:
        """
            Returns documents with given ids.
        """
        results = self.get_documents(ids=ids)
        ret = self._get_result_to_documents(results, return_embedding)
        return ret

    def get_document_by_id(self, id: str, return_embedding: Optional[bool] = None) -> Document:
        """
            Returns documents with given ids.
        """
        document = self.get_documents(ids=[id])
        ret = self._get_result_to_documents(document, return_embedding)
        return ret

    # def search(
    #         self, queries: List[Union[str, Dict[str, float]]], top_k: int, filters: Optional[Dict[str, Any]] = None
    # ) -> List[List[Document]]:
    #     """Perform a search for a list of queries.
    #
    #     Args:
    #         queries (List[Union[str, Dict[str, float]]]): A list of queries.
    #         top_k (int): The number of results to return.
    #         filters (Optional[Dict[str, Any]], optional): Filters to apply during search. Defaults to None.
    #
    #     Returns:
    #         List[List[Document]]: A list of matching documents for each query.
    #     """
    #     results = []
    #     for query in queries:
    #         result = self._index.search(q=query, limit=top_k, filter_string=self._convert_filters(filters))
    #         results.append(result)
    #
    #     return self._query_result_to_documents(results)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param document_ids: the document_ids to delete
        """
        self._index.delete(ids=document_ids)
