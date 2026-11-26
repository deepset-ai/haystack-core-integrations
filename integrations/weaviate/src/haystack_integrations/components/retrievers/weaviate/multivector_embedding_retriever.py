# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@component
class WeaviateMultivectorEmbeddingRetriever:
    """
    Retrieve documents using Weaviate's multi-vector (MaxSim) search.

    The collection must have a multi-vector HNSW index named `multi_vector`. See the
    [Weaviate multivector guide](https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings).
    Returned documents include their stored embeddings. Scores are negative Weaviate distances, so higher is better;
    cosine certainty is not supported for MaxSim.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
    from haystack_integrations.components.retrievers.weaviate import WeaviateMultivectorEmbeddingRetriever

    store = WeaviateDocumentStore(
        url="http://localhost:8080",
        collection_settings={
            "class": "MultivectorDocuments",
            "vectorConfig": {
                "multi_vector": {
                    "vectorizer": {"none": {}},
                    "vectorIndexType": "hnsw",
                    "vectorIndexConfig": {"multivector": {"enabled": True}},
                }
            },
        },
    )
    store.write_documents([
        Document(id="1", content="A document", multivector_embedding=[[1.0, 0.0], [0.0, 1.0]])
    ])
    retriever = WeaviateMultivectorEmbeddingRetriever(document_store=store)
    result = retriever.run(query_multivector_embedding=[[1.0, 0.0]])
    print([doc.content for doc in result["documents"]])  # ["A document"]
    retriever.close()
    ```
    """

    def __init__(
        self,
        *,
        document_store: WeaviateDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        distance: float | None = None,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create a multivector embedding retriever.

        :param document_store:
            Weaviate document store with a `multi_vector` index.
        :param filters:
            Filters applied to retrieved documents.
        :param top_k:
            Maximum number of documents to return.
        :param distance:
            Maximum allowed Weaviate MaxSim distance, which can be negative and is not normalized.
            `None` applies no distance cutoff.
        :param filter_policy:
            Policy for replacing or merging initialization filters with runtime filters.
        """
        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._distance = distance
        self._filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component.

        :returns:
            Dictionary containing the component type and initialization parameters.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            filters=self._filters,
            top_k=self._top_k,
            distance=self._distance,
            filter_policy=self._filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeaviateMultivectorEmbeddingRetriever":
        """
        Deserialize this component.

        :param data:
            Serialized component dictionary.
        :returns:
            The restored retriever.
        """
        data["init_parameters"]["document_store"] = WeaviateDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    def close(self) -> None:
        """
        Release the synchronous resources of the underlying document store.
        """
        self._document_store.close()

    async def close_async(self) -> None:
        """
        Release the asynchronous resources of the underlying document store.
        """
        await self._document_store.close_async()

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_multivector_embedding: list[list[float]],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        distance: float | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents using a matrix of query token vectors.

        :param query_multivector_embedding:
            Nonempty matrix of token vectors with the same dimension as the indexed document vectors.
        :param filters:
            Runtime filters, applied according to `filter_policy`.
        :param top_k:
            Maximum number of documents to return. `None` uses the initialization value.
        :param distance:
            Maximum allowed MaxSim distance. `None` uses the initialization value.
        :returns:
            A dictionary with `documents`: a list of retrieved `Document` objects, ordered by similarity.
        """
        documents = self._document_store._multivector_embedding_retrieval(
            query_multivector_embedding=query_multivector_embedding,
            filters=apply_filter_policy(self._filter_policy, self._filters, filters),
            top_k=self._top_k if top_k is None else top_k,
            distance=self._distance if distance is None else distance,
        )
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_multivector_embedding: list[list[float]],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        distance: float | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve documents using a matrix of query token vectors.

        :param query_multivector_embedding:
            Nonempty matrix of token vectors with the same dimension as the indexed document vectors.
        :param filters:
            Runtime filters, applied according to `filter_policy`.
        :param top_k:
            Maximum number of documents to return. `None` uses the initialization value.
        :param distance:
            Maximum allowed MaxSim distance. `None` uses the initialization value.
        :returns:
            A dictionary with `documents`: a list of retrieved `Document` objects, ordered by similarity.
        """
        documents = await self._document_store._multivector_embedding_retrieval_async(
            query_multivector_embedding=query_multivector_embedding,
            filters=apply_filter_policy(self._filter_policy, self._filters, filters),
            top_k=self._top_k if top_k is None else top_k,
            distance=self._distance if distance is None else distance,
        )
        return {"documents": documents}
