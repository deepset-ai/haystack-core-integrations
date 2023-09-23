# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack.preview import Document, component, default_from_dict, default_to_dict

from chroma_haystack import ChromaDocumentStore


@component
class ChromaDenseRetriever:
    """
    A component for retrieving documents from an ChromaDocumentStore.
    """

    def __init__(self, document_store: ChromaDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        Create an ExampleRetriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        """
        self.filters = filters
        self.top_k = top_k
        self.document_store = document_store

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
        if not top_k:
            top_k = 3
        return {"documents": self.document_store.search(queries, top_k)}

    def to_dict(self) -> Dict[str, Any]:
        """
        Override the default serializer in order to manage the Chroma client string representation
        """
        d = default_to_dict(self, filters=self.filters, top_k=self.top_k)
        d["init_parameters"]["document_store"] = self.document_store.to_dict()

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChromaDenseRetriever":
        document_store = ChromaDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        return default_from_dict(cls, data)
