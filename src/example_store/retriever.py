# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Any, Optional

from haystack.preview import component, Document
from haystack.preview.document_stores import StoreAwareMixin

from example_store import ExampleDocumentStore


@component
class ExampleRetriever(StoreAwareMixin):
    """
    A component for retrieving documents from an ExampleDocumentStore.
    """

    supported_stores = [ExampleDocumentStore]

    def __init__(self, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        Create an ExampleRetriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).

        :raises ValueError: If the specified top_k is not > 0.
        """
        # FIXME
        self.filters = filters
        self.top_k = top_k

    def run(self, data):
        """
        Run the MemoryRetriever on the given input data.

        :param data: The input data for the retriever. In this case, a list of queries.
        :return: The retrieved documents.

        :raises ValueError: If the specified document store is not found or is not a MemoryDocumentStore instance.
        """
        if not self.store:
            # `self.store` is normally populated when adding this component to a pipeline.
            # If you want to use this component standalone, you must create an instance
            # of the right document store and assign it to `self.store` before invoking `run()`
            raise ValueError("ExampleRetriever needs a store to run!")
        return []  # FIXME
