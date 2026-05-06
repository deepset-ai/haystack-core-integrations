# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.elasticsearch.document_store import ElasticsearchDocumentStore


@component
class ElasticsearchSparseEmbeddingRetriever:
    """
    ElasticsearchSparseEmbeddingRetriever retrieves documents using sparse vector similarity.

    Usage example:
    ```python
    from haystack import Document
    from haystack.dataclasses.sparse_embedding import SparseEmbedding
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchSparseEmbeddingRetriever

    document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200", sparse_vector_field="sparse_vec")
    retriever = ElasticsearchSparseEmbeddingRetriever(document_store=document_store)

    documents = [
        Document(
            content="My name is Carla and I live in Berlin",
            sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.8, 0.4]),
        ),
        Document(
            content="My name is Paul and I live in New York",
            sparse_embedding=SparseEmbedding(indices=[2, 3], values=[0.7, 0.6]),
        ),
    ]
    document_store.write_documents(documents)

    query_sparse_embedding = SparseEmbedding(indices=[0, 1], values=[1.0, 1.0])
    result = retriever.run(query_sparse_embedding=query_sparse_embedding)
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(
        self,
        *,
        document_store: ElasticsearchDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create the ElasticsearchSparseEmbeddingRetriever component.

        :param document_store: An instance of ElasticsearchDocumentStore.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an instance of ElasticsearchDocumentStore.
        """
        if not isinstance(document_store, ElasticsearchDocumentStore):
            msg = "document_store must be an instance of ElasticsearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            filter_policy=self._filter_policy.value,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchSparseEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = ElasticsearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_sparse_embedding: SparseEmbedding,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents using sparse vector similarity.

        :param query_sparse_embedding: Sparse embedding of the query.
        :param filters: Filters applied when fetching documents from the Document Store.
            The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
        :param top_k: Maximum number of documents to return.
        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s most similar to the given `query_sparse_embedding`
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        docs = self._document_store._sparse_vector_retrieval(
            query_sparse_embedding=query_sparse_embedding,
            filters=filters,
            top_k=top_k or self._top_k,
        )
        return {"documents": docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_sparse_embedding: SparseEmbedding,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve documents using sparse vector similarity.

        :param query_sparse_embedding: Sparse embedding of the query.
        :param filters: Filters applied when fetching documents from the Document Store.
            The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
        :param top_k: Maximum number of documents to return.
        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s most similar to the given `query_sparse_embedding`
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        docs = await self._document_store._sparse_vector_retrieval_async(
            query_sparse_embedding=query_sparse_embedding,
            filters=filters,
            top_k=top_k or self._top_k,
        )
        return {"documents": docs}
