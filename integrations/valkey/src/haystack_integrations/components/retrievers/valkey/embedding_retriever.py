from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.valkey import ValkeyDocumentStore


@component
class ValkeyEmbeddingRetriever:
    """
    A component for retrieving documents from a ValkeyDocumentStore using vector similarity search.

    This retriever uses dense embeddings to find semantically similar documents. It supports
    filtering by metadata fields and configurable similarity thresholds.

    Key features:
    - Vector similarity search using HNSW algorithm
    - Metadata filtering with tag and numeric field support
    - Configurable top-k results
    - Filter policy management for runtime filter application

    Usage example:
    ```python
    from haystack.document_stores.types import DuplicatePolicy
    from haystack import Document
    from haystack import Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
    from haystack_integrations.components.retrievers.valkey import ValkeyEmbeddingRetriever
    from haystack_integrations.document_stores.valkey import ValkeyDocumentStore

    document_store = ValkeyDocumentStore(index_name="my_index", embedding_dim=768)

    documents = [Document(content="There are over 7,000 languages spoken around the world today."),
                 Document(content="Elephants have been observed to behave in a way that indicates..."),
                 Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)

    document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", ValkeyEmbeddingRetriever(document_store=document_store))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    query = "How many languages are there?"

    res = query_pipeline.run({"text_embedder": {"text": query}})
    assert res['retriever']['documents'][0].content == "There are over 7,000 languages spoken around the world today."
    ```
    """

    def __init__(
        self,
        *,
        document_store: ValkeyDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ):
        """
        :param document_store: The Valkey Document Store.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.

        :raises ValueError: If `document_store` is not an instance of `ValkeyDocumentStore`.
        """
        if not isinstance(document_store, ValkeyDocumentStore):
            msg = "document_store must be an instance of ValkeyDocumentStore"
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
        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValkeyEmbeddingRetriever:
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = ValkeyDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the `ValkeyDocumentStore`, based on their dense embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of `Document`s to return.

        :returns: List of Document similar to `query_embedding`.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)

        top_k = top_k or self.top_k

        docs = self.document_store.search(
            embedding=query_embedding,
            filters=filters,
            limit=top_k,
        )
        return {"documents": docs}
