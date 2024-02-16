from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@component
class WeaviateEmbeddingRetriever:
    """
    A retriever that uses Weaviate's vector search to find similar documents based on the embeddings of the query.
    """

    def __init__(
        self,
        *,
        document_store: WeaviateDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        distance: Optional[float] = None,
        certainty: Optional[float] = None,
    ):
        """
        Create a new instance of WeaviateEmbeddingRetriever.
        Raises ValueError if both `distance` and `certainty` are provided.
        See the official Weaviate documentation to learn more about the `distance` and `certainty` parameters:
        https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables

        :param document_store: Instance of WeaviateDocumentStore that will be associated with this retriever.
        :param filters: Custom filters applied when running the retriever, defaults to None
        :param top_k: Maximum number of documents to return, defaults to 10
        :param distance: The maximum allowed distance between Documents' embeddings, defaults to None
        :param certainty: Normalized distance between the result item and the search vector, defaults to None
        """
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._distance = distance
        self._certainty = certainty

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            distance=self._distance,
            certainty=self._certainty,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaviateEmbeddingRetriever":
        data["init_parameters"]["document_store"] = WeaviateDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        distance: Optional[float] = None,
        certainty: Optional[float] = None,
    ):
        filters = filters or self._filters
        top_k = top_k or self._top_k
        distance = distance or self._distance
        certainty = certainty or self._certainty
        return self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            distance=distance,
            certainty=certainty,
        )
