from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@component
class WeaviateBM25Retriever:
    """
    A component for retrieving documents from Weaviate using the BM25 algorithm.

    Example usage:
    ```python
    from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
    from haystack_integrations.components.retrievers.weaviate.bm25_retriever import WeaviateBM25Retriever

    document_store = WeaviateDocumentStore(url="http://localhost:8080")
    retriever = WeaviateBM25Retriever(document_store=document_store)
    retriever.run(query="How to make a pizza", top_k=3)
    ```
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

        :param document_store:
            Instance of WeaviateDocumentStore that will be used from this retriever.
        :param filters:
            Custom filters applied when running the retriever
        :param top_k:
            Maximum number of documents to return
        """
        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k

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
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaviateBM25Retriever":
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
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """
        Retrieves documents from Weaviate using the BM25 algorithm.

        :param query:
            The query text.
        :param filters:
            Filters to use when running the retriever.
        :param top_k:
            The maximum number of documents to return.
        """
        filters = filters or self._filters
        top_k = top_k or self._top_k
        documents = self._document_store._bm25_retrieval(query=query, filters=filters, top_k=top_k)
        return {"documents": documents}
