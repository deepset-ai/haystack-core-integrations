# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar, Dict, List, Optional

from haystack.preview import Document, component
from haystack.preview.document_stores import DocumentStore, DocumentStoreAwareMixin

from chroma_haystack import ChromaDocumentStore


@component
class ChromaDenseRetriever(DocumentStoreAwareMixin):
    """
    A component for retrieving documents from an ChromaDocumentStore.
    """

    supported_document_stores: ClassVar[DocumentStore] = [ChromaDocumentStore]

    def __init__(self, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        Create an ExampleRetriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        """
        self.filters = filters
        self.top_k = top_k

    @component.output_types(documents=List[List[Document]])
    def run(
        self,
        queries: List[str],
        _: Optional[Dict[str, Any]] = None,  # filters not yet supported
        top_k: Optional[int] = None,
    ):
        """
        Run the retriever on the given input data.

        :param queries: The input data for the retriever. In this case, a list of queries.
        :return: The retrieved documents.

        :raises ValueError: If the specified document store is not found or is not a MemoryDocumentStore instance.
        """
        if not self.document_store:
            # `self.store` is normally populated when adding this component to a pipeline.
            # If you want to use this component standalone, you must create an instance
            # of the right document store and assign it to `self.store` before invoking `run()`
            msg = "ExampleRetriever needs a store to run!"
            raise ValueError(msg)

        return self.document_store.search(queries, top_k)
