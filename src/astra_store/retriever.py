# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack.preview import Document, component

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

    @component.output_types(documents=List[List[Document]])
    def run(self, queries: List[str], filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
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

        return {"documents": self.document_store.search(queries, top_k, filters=filters)}


@component
class AstraSingleRetriever(AstraRetriever):
    def __init__(self, *args, **kwargs):
        super(AstraSingleRetriever, self).__init__(*args, **kwargs)

    @component.output_types(documents=List[Document])
    def run(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """Run the retriever on a single query.

        Args:
            query (str): An input query
            filters (Optional[Dict[str, Any]], optional): A dictionary with filters to narrow down the search space. Defaults to None.
            top_k (Optional[int], optional): The maximum number of documents to retrieve. Defaults to None.
        """
        return {"documents": super(AstraSingleRetriever, self).run([query], filters, top_k)["documents"][0]}
