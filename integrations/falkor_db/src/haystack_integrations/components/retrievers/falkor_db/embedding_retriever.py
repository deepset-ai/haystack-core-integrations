# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types.filter_policy import FilterPolicy, apply_filter_policy

from haystack_integrations.document_stores.falkor_db import FalkorDBDocumentStore


@component
class FalkorDBEmbeddingRetriever:
    """
    A component for retrieving documents from a FalkorDBDocumentStore using vector similarity.

    The retriever uses FalkorDB's native vector search index to find documents whose embeddings
    are most similar to the provided query embedding.

    Usage example:
    ```python
    from haystack.dataclasses import Document
    from haystack_integrations.document_stores.falkor_db import FalkorDBDocumentStore
    from haystack_integrations.components.retrievers.falkor_db import FalkorDBEmbeddingRetriever

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
        :param top_k: Maximum number of documents to retrieve. Defaults to ``10``.
        :param filter_policy: Policy to determine how runtime filters are combined with
            initialization filters. Defaults to ``FilterPolicy.REPLACE``.
        :raises ValueError: If the provided `document_store` is not a `FalkorDBDocumentStore`.
        """
        if not isinstance(document_store, FalkorDBDocumentStore):
            msg = "document_store must be an instance of FalkorDBDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters
        self._top_k = top_k
        self._filter_policy = FilterPolicy(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise this component to a dictionary.

        :returns: Dictionary representation of this retriever's configuration.
        """
        data = default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            filter_policy=self._filter_policy.value,
        )
        data["init_parameters"]["document_store"] = self._document_store.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FalkorDBEmbeddingRetriever":
        """
        Deserialise this component from a dictionary.

        :param data: Dictionary previously produced by :meth:`to_dict`.
        :returns: A new :class:`FalkorDBEmbeddingRetriever` instance.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" in init_params:
            init_params["document_store"] = FalkorDBDocumentStore.from_dict(init_params["document_store"])
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
        final_filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        final_top_k = top_k if top_k is not None else self._top_k

        docs = self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            top_k=final_top_k,
            filters=final_filters,
        )

        return {"documents": docs}
