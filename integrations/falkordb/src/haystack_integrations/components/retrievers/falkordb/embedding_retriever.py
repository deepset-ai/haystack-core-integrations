# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy, apply_filter_policy

from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore


@component
class FalkorDBEmbeddingRetriever:
    """
    A component for retrieving documents from a FalkorDBDocumentStore using vector similarity.

    The retriever uses FalkorDB's native vector search index to find documents whose embeddings
    are most similar to the provided query embedding.

    Usage example:
    ```python
    from haystack.dataclasses import Document
    from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore
    from haystack_integrations.components.retrievers.falkordb import FalkorDBEmbeddingRetriever

    store = FalkorDBDocumentStore(host="localhost", port=6379)
    store.write_documents([
        Document(content="GraphRAG is powerful.", embedding=[0.1, 0.2, 0.3]),
        Document(content="FalkorDB is fast.", embedding=[0.8, 0.9, 0.1]),
    ])

    retriever = FalkorDBEmbeddingRetriever(document_store=store)
    res = retriever.run(query_embedding=[0.1, 0.2, 0.3])
    print(res["documents"][0].content)  # "GraphRAG is powerful."
    ```
    """

    def __init__(
        self,
        document_store: FalkorDBDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create a new FalkorDBEmbeddingRetriever.

        :param document_store: The FalkorDBDocumentStore instance.
        :param filters: Optional Haystack filters to narrow down the search space.
        :param top_k: Maximum number of documents to retrieve.
        :param filter_policy: Policy to determine how runtime filters are combined with
            initialization filters.
        :raises ValueError: If the provided `document_store` is not a `FalkorDBDocumentStore`.
        """
        if not isinstance(document_store, FalkorDBDocumentStore):
            msg = "document_store must be an instance of FalkorDBDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.filter_policy = FilterPolicy(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the retriever to a dictionary.

        :returns: Dictionary representation of the retriever.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FalkorDBEmbeddingRetriever":
        """
        Deserialise a `FalkorDBEmbeddingRetriever` produced by `to_dict`.

        :param data: Serialised retriever dictionary.
        :returns: Reconstructed `FalkorDBEmbeddingRetriever` instance.
        """
        init_params = data["init_parameters"]
        init_params["document_store"] = FalkorDBDocumentStore.from_dict(init_params["document_store"])
        if "filter_policy" in init_params:
            init_params["filter_policy"] = FilterPolicy(init_params["filter_policy"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents by vector similarity.

        :param query_embedding: Query embedding vector.
        :param filters: Optional Haystack filters to be combined with the init filters based
            on the configured filter policy.
        :param top_k: Maximum number of documents to return. If not provided, the default
            top_k from initialization is used.
        :returns: Dictionary containing a `"documents"` key with the retrieved documents.
        """
        final_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        final_top_k = top_k if top_k is not None else self.top_k

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            top_k=final_top_k,
            filters=final_filters,
        )

        return {"documents": docs}
