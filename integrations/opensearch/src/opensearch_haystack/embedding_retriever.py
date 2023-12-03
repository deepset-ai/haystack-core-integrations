# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from opensearch_haystack.document_store import OpenSearchDocumentStore


@component
class OpenSearchEmbeddingRetriever:
    """
    Uses a vector similarity metric to retrieve documents from the OpenSearchDocumentStore.

    Needs to be connected to the OpenSearchDocumentStore to run.
    """

    def __init__(
        self,
        *,
        document_store: OpenSearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ):
        """
        Create the OpenSearchEmbeddingRetriever component.

        :param document_store: An instance of OpenSearchDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
            Filters are applied during the approximate kNN search to ensure that top_k matching documents are returned.
        :param top_k: Maximum number of Documents to return, defaults to 10
        :raises ValueError: If `document_store` is not an instance of OpenSearchDocumentStore.
        """
        if not isinstance(document_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenSearchEmbeddingRetriever":
        data["init_parameters"]["document_store"] = OpenSearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """
        Retrieve documents using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: Optional filters to narrow down the search space.
        :param top_k: Maximum number of Documents to return.
        :return: List of Document similar to `query_embedding`.
        """
        if filters is None:
            filters = self._filters
        if top_k is None:
            top_k = self._top_k

        docs = self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
