# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_integrations.document_stores.arangodb.document_store import ArangoDocumentStore


@component
class ArangoEmbeddingRetriever:
    """
    Retrieves documents from an `ArangoDocumentStore` using cosine similarity on embeddings.

    Example usage:

    ```python
    from haystack_integrations.document_stores.arangodb import ArangoDocumentStore
    from haystack_integrations.components.retrievers.arangodb import ArangoEmbeddingRetriever

    store = ArangoDocumentStore(host="http://localhost:8529", database="haystack",
                                username="root", collection_name="docs", embedding_dimension=768)
    retriever = ArangoEmbeddingRetriever(document_store=store, top_k=5)
    result = retriever.run(query_embedding=[0.1, 0.2, ...])
    ```
    """

    def __init__(
        self,
        *,
        document_store: ArangoDocumentStore,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a new ArangoEmbeddingRetriever.

        :param document_store: The `ArangoDocumentStore` to retrieve documents from.
        :param top_k: Maximum number of documents to return.
        :param filters: Optional Haystack metadata filters applied at retrieval time.
        """
        self.document_store = document_store
        self.top_k = top_k
        self.filters = filters

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents most similar to `query_embedding`.

        :param query_embedding: The query vector.
        :param top_k: Overrides the instance-level `top_k` for this call.
        :param filters: Overrides the instance-level `filters` for this call.
        :returns: A dictionary with `documents` — a list of `Document` objects sorted by score.
        """
        top_k = top_k if top_k is not None else self.top_k
        filters = filters if filters is not None else self.filters
        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )
        return {"documents": docs}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            top_k=self.top_k,
            filters=self.filters,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArangoEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        data["init_parameters"]["document_store"] = ArangoDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)
