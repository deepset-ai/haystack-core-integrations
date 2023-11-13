# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from haystack.preview import component

from example_store import ExampleDocumentStore


@component
class ExampleRetriever:
    """
    A component for retrieving documents from an ExampleDocumentStore.
    """

    def __init__(self, document_store: ExampleDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        Create an ExampleRetriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param document_store: A Document Store object used to retrieve documents
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).

        :raises ValueError: If the specified top_k is not > 0.
        """
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
