# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@component
class PineconeEmbeddingRetriever:
    """
    Retrieves documents from the `PineconeDocumentStore`, based on their dense embeddings.

    Usage example:
    ```python
    import os
    from haystack.document_stores.types import DuplicatePolicy
    from haystack import Document
    from haystack import Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
    from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

    os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"
    document_store = PineconeDocumentStore(index="my_index", namespace="my_namespace", dimension=768)

    documents = [Document(content="There are over 7,000 languages spoken around the world today."),
                 Document(content="Elephants have been observed to behave in a way that indicates..."),
                 Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

    document_embedder = SentenceTransformersDocumentEmbedder()
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)

    document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=document_store))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    query = "How many languages are there?"

    res = query_pipeline.run({"text_embedder": {"text": query}})
    assert res['retriever']['documents'][0].content == "There are over 7,000 languages spoken around the world today."
    ```
    """

    def __init__(
        self,
        *,
        document_store: PineconeDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        :param document_store: The Pinecone Document Store.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.

        :raises ValueError: If `document_store` is not an instance of `PineconeDocumentStore`.
        """
        if not isinstance(document_store, PineconeDocumentStore):
            msg = "document_store must be an instance of PineconeDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "PineconeEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = PineconeDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents from the `PineconeDocumentStore`, based on their dense embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of `Document`s to return.

        :returns: List of Document similar to `query_embedding`.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)

        top_k = top_k or self.top_k

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Asynchronously retrieve documents from the `PineconeDocumentStore`, based on their dense embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of `Document`s to return.

        :returns: List of Document similar to `query_embedding`.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)

        top_k = top_k or self.top_k

        docs = await self.document_store._embedding_retrieval_async(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
