# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.faiss import FAISSDocumentStore


@component
class FAISSEmbeddingRetriever:
    """
    Retrieves documents from the `FAISSDocumentStore`, based on their dense embeddings.

    Example usage:
    ```python
    from haystack import Document, Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
    from haystack.document_stores.types import DuplicatePolicy

    from haystack_integrations.document_stores.faiss import FAISSDocumentStore
    from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever

    document_store = FAISSDocumentStore(embedding_dim=768)

    documents = [
        Document(content="There are over 7,000 languages spoken around the world today."),
        Document(content="Elephants have been observed to behave in a way that indicates a high level of intelligence."),
        Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
    ]

    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)["documents"]

    document_store.write_documents(documents_with_embeddings, policy=DuplicatePolicy.OVERWRITE)

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", FAISSEmbeddingRetriever(document_store=document_store))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    query = "How many languages are there?"
    res = query_pipeline.run({"text_embedder": {"text": query}})

    assert res["retriever"]["documents"][0].content == "There are over 7,000 languages spoken around the world today."
    ```
    """ # noqa: E501

    def __init__(
        self,
        *,
        document_store: FAISSDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ):
        """
        :param document_store: An instance of `FAISSDocumentStore`.
        :param filters: Filters applied to the retrieved Documents at initialisation time. At runtime, these are merged
            with any runtime filters according to the `filter_policy`.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how init-time and runtime filters are combined.
            See `FilterPolicy` for details. Defaults to `FilterPolicy.REPLACE`.
        :raises ValueError: If `document_store` is not an instance of `FAISSDocumentStore`.
        """
        if not isinstance(document_store, FAISSDocumentStore):
            msg = "document_store must be an instance of FAISSDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FAISSEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = FAISSDocumentStore.from_dict(doc_store_params)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the `FAISSDocumentStore`, based on their embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of Documents to return. Overrides the value set at initialization.
        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s that are similar to `query_embedding`.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k
        docs = self.document_store.search(query_embedding=query_embedding, top_k=top_k, filters=filters)
        return {"documents": docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve documents from the `FAISSDocumentStore`, based on their embeddings.

        Since FAISS search is CPU-bound and fully in-memory, this delegates directly to the synchronous
        `run()` method. No I/O or network calls are involved.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of Documents to return. Overrides the value set at initialization.
        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s that are similar to `query_embedding`.
        """
        return self.run(query_embedding=query_embedding, filters=filters, top_k=top_k)
