from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@component
class WeaviateHybridRetriever:
    """
    A retriever that uses Weaviate's hybrid search to find similar documents based on the embeddings of the query.
    """

    def __init__(
        self,
        *,
        document_store: WeaviateDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        alpha: Optional[float] = None,
        max_vector_distance: Optional[float] = None,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        Creates a new instance of WeaviateHybridRetriever.

        :param document_store:
            Instance of WeaviateDocumentStore that will be used from this retriever.
        :param filters:
            Custom filters applied when running the retriever.
        :param top_k:
            Maximum number of documents to return.
        :param alpha:
            The alpha value for hybrid retrieval.
        :param max_vector_distance:
            The maximum vector distance for vector part of hybrid retrieval.
        :param filter_policy:
            Policy to determine how filters are applied.
        """

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._alpha = alpha
        self._max_vector_distance = max_vector_distance
        self._filter_policy = (
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
            filters=self._filters,
            top_k=self._top_k,
            alpha=self._alpha,
            max_vector_distance=self._max_vector_distance,
            filter_policy=self._filter_policy.value,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaviateHybridRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = WeaviateDocumentStore.from_dict(
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
        query: str,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        max_vector_distance: Optional[float] = None,
    ) -> Dict[str, List[Document]]:
        """
        Retrieves documents from Weaviate using the hybrid search.

        :param query:
            The query text.
        :param query_embedding:
            Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k:
            The maximum number of documents to return.
        :param alpha:
            The alpha value for hybrid retrieval.
        :param max_vector_distance:
            The maximum vector distance for vector part of hybrid retrieval.
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        top_k = self._top_k if top_k is None else top_k
        alpha = self._alpha if alpha is None else alpha
        max_vector_distance = self._max_vector_distance if max_vector_distance is None else max_vector_distance

        documents = self._document_store._hybrid_retrieval(
            query=query,
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            alpha=alpha,
            max_vector_distance=max_vector_distance,
        )
        return {"documents": documents}
