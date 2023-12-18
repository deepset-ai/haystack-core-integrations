from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict

from qdrant_haystack import QdrantDocumentStore


@component
class QdrantRetriever:
    """
    A component for retrieving documents from an QdrantDocumentStore.
    """

    def __init__(
        self,
        document_store: QdrantDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,  # noqa: FBT001, FBT002
        return_embedding: bool = False,  # noqa: FBT001, FBT002
    ):
        """
        Create a QdrantRetriever component.

        :param document_store: An instance of QdrantDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space. Default is None.
        :param top_k: The maximum number of documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents or not. Default is True.
        :param return_embedding: Whether to return the embedding of the retrieved Documents. Default is False.

        :raises ValueError: If 'document_store' is not an instance of QdrantDocumentStore.
        """

        if not isinstance(document_store, QdrantDocumentStore):
            msg = "document_store must be an instance of QdrantDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store

        self._filters = filters
        self._top_k = top_k
        self._scale_score = scale_score
        self._return_embedding = return_embedding

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        d = default_to_dict(
            self,
            document_store=self._document_store,
            filters=self._filters,
            top_k=self._top_k,
            scale_score=self._scale_score,
            return_embedding=self._return_embedding,
        )
        d["init_parameters"]["document_store"] = self._document_store.to_dict()

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdrantRetriever":
        """
        Deserialize this component from a dictionary.
        """
        document_store = QdrantDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ):
        """
        Run the Embedding Retriever on the given input data.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the scores of the retrieved documents or not.
        :param return_embedding: Whether to return the embedding of the retrieved Documents.
        :return: The retrieved documents.

        """
        docs = self._document_store.query_by_embedding(
            query_embedding=query_embedding,
            filters=filters or self._filters,
            top_k=top_k or self._top_k,
            scale_score=scale_score or self._scale_score,
            return_embedding=return_embedding or self._return_embedding,
        )

        return {"documents": docs}
