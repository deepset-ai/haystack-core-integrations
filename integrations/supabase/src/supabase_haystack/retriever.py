# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component
from haystack.dataclasses import Document

from src.supabase_haystack.document_store import SupabaseDocumentStore

@component
class SupabaseEmbeddingRetriever:
    """
    Retrieves documents from the SupabaseDocumentStore, based on their dense embeddings.

    Needs to be connected to the SupabaseDocumentStore.
    """

    def __init__(
        self,
        *,
        document_store: SupabaseDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10):
        """
        Create an SupabaseEmbeddingRetriever component.

        :param document_store: An instance of SupabaseDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
        :param top_k: Maximum number of Documents to return, defaults to 10.

        :raises ValueError: If `document_store` is not an instance of SupabaseDocumentStore.
        """
        if not isinstance(document_store, SupabaseDocumentStore):
            msg = "document_store must be an instance of SupabaseDocumentStore"
            raise ValueError(msg)


        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float]):
        """
        Retrieve documents from the SupabaseDocumentStore, based on their dense embeddings.

        :param query_embedding: Embedding of the query.
        :return: List of Document similar to `query_embedding`.
        """
        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=self.filters,
            top_k=self.top_k,
        )
        return {"documents": docs}
