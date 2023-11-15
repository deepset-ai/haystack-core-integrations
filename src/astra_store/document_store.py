# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores.decorator import document_store
from haystack.preview.document_stores.errors import DuplicateDocumentError, MissingDocumentError
from haystack.preview.document_stores.protocols import DuplicatePolicy
from astra_client import AstraClient
# from astra_store.astra_client import AstraClient
# from astra_store.errors import AstraDocumentStoreFilterError
from errors import AstraDocumentStoreFilterError

logger = logging.getLogger(__name__)


@document_store
class AstraDocumentStore:  # FIXME
    """
        An AstraDocumentStore document store for Haystack.
    """

    def __init__(
            self,
            collection_name: str = "documents",
            keyspace_name: str = "haystack",
            application_token: Optional[str] = None,
            region_id: Optional[str] = None,
            astra_id: Optional[str] = None,
            client_batch_size: int = 4,
    ):
        self._index = AstraClient(astra_id=astra_id, region=region_id, token=application_token, keyspace_name=keyspace_name, collection_name=collection_name)
        self._collection = collection_name
        self.client_batch_size = client_batch_size

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self._index.describe_index_stats()["total_document_count"]

    def count_vectors(self) -> int:
        """
        Returns how many vectors are present in the document store.
        """
        return self._index.describe_index_stats()["total_vector_count"]

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

    def _get_documents(self, results) -> List[Document]:
        documents = []
        for res in results.matches:
            document = Document(
                id=res.id,
                text=res.metadata.text,  ## Needs to agree on whether to put this inside metadata or outside
                metadata=res.metadata if not res.metadata else {},
                mime_type="",
                score=res.score,
            )
            documents.append(document)
        return documents


    def get_documents_by_id(self, ids: List[str]) -> List[Document]:
        """
            Returns documents with given ids.
        """
        results = self._index.get_documents(document_ids=ids)["results"] ## todo get_documents
        results = [r for r in results if r["_found"]]
        return self._get_result_to_documents(results)

    def search(
            self, queries: List[Union[str, Dict[str, float]]], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Document]]:
        """Perform a search for a list of queries.

        Args:
            queries (List[Union[str, Dict[str, float]]]): A list of queries.
            top_k (int): The number of results to return.
            filters (Optional[Dict[str, Any]], optional): Filters to apply during search. Defaults to None.

        Returns:
            List[List[Document]]: A list of matching documents for each query.
        """
        results = []
        for query in queries:
            result = self._index.search(q=query, limit=top_k, filter_string=self._convert_filters(filters))
            results.append(result)

        return self._query_result_to_documents(results)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param document_ids: the document_ids to delete
        """
        self._index.delete(ids=document_ids)
