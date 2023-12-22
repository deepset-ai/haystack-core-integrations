# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Document, component

from astra_store.document_store import AstraDocumentStore


@component
class AstraRetriever:
    """
    A component for retrieving documents from an AstraDocumentStore.
    """

    def __init__(self, document_store: AstraDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        Create an AstraRetriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        """
        self.filters = filters
        self.top_k = top_k
        self.document_store = document_store

        if not isinstance(document_store, AstraDocumentStore):
            raise Exception("document_store must be an instance of AstraDocumentStore")

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """Run the retriever on the given list of queries.

        Args:
            queries (List[str]): An input list of queries
            filters (Optional[Dict[str, Any]], optional): A dictionary with filters to narrow down the search space. Defaults to None.
            top_k (Optional[int], optional): The maximum number of documents to retrieve. Defaults to None.
        """

        if not top_k:
            top_k = self.top_k

        if not filters:
            filters = self.filters

        return {"documents": self.document_store.search(query_embedding, top_k, filters=filters)}
