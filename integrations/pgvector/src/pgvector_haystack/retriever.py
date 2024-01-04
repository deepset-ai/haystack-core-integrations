# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from haystack import component

from pgvector_haystack import pgvectorDocumentStore


@component
class pgvectorQueryRetriever:
    """
    A component for retrieving documents from an pgvectorDocumentStore.
    """

    def __init__(
        self, 
        *,
        document_store: pgvectorDocumentStore, 
        filters: Optional[Dict[str, Any]] = None, 
        top_k: int = 10,
    ):
        """
        Create an pgvectorRetriever component.

        :param document_store: An instance of pgvectorDocumentStore
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve default is 10.

        :raises ValueError: If the specified top_k is not > 0.
        """
        if top_k <= 0:
            msg = "top_k must be greater than zero"
            raise ValueError(msg)
        self.filters = filters
        self.top_k = top_k
        self.document_store = document_store

    def run(self, _):
        """
        Run the Retriever on the given input data.

        :param data: The input data for the retriever. In this case, a list of queries.
        :return: The retrieved documents.
        """
        return []  # FIXME