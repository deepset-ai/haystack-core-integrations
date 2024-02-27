from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@component
class WeaviateBM25Retriever:
    """
    Retriever that uses BM25 to find the most promising documents for a given query.
    """

    def __init__(
        self,
        *,
        document_store: WeaviateDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ):
        """
        Create a new instance of WeaviateBM25Retriever.

        :param document_store: Instance of WeaviateDocumentStore that will be associated with this retriever.
        :param filters: Custom filters applied when running the retriever, defaults to None
        :param top_k: Maximum number of documents to return, defaults to 10
        """
        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaviateBM25Retriever":
        data["init_parameters"]["document_store"] = WeaviateDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        filters = filters or self._filters
        top_k = top_k or self._top_k
        documents = self._document_store._bm25_retrieval(query=query, filters=filters, top_k=top_k)
        return {"documents": documents}
