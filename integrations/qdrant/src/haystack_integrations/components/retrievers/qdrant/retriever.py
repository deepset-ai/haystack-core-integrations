from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


@component
class QdrantEmbeddingRetriever:
    """
    A component for retrieving documents from an QdrantDocumentStore using dense vectors.

    Usage example:
    ```python
    from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    document_store = QdrantDocumentStore(
        ":memory:",
        recreate_index=True,
        return_embedding=True,
        wait_result_from_api=True,
    )
    retriever = QdrantEmbeddingRetriever(document_store=document_store)

    # using a fake vector to keep the example simple
    retriever.run(query_embedding=[0.1]*768)
    ```
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
        Create a QdrantEmbeddingRetriever component.

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
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
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
    def from_dict(cls, data: Dict[str, Any]) -> "QdrantEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
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
        :returns:
            The retrieved documents.

        """
        docs = self._document_store.query_by_embedding(
            query_embedding=query_embedding,
            filters=filters or self._filters,
            top_k=top_k or self._top_k,
            scale_score=scale_score or self._scale_score,
            return_embedding=return_embedding or self._return_embedding,
        )

        return {"documents": docs}


@component
class QdrantSparseRetriever:
    """
    A component for retrieving documents from an QdrantDocumentStore using sparse vectors.

    Usage example:
    ```python
    from haystack_integrations.components.retrievers.qdrant import QdrantSparseRetriever
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    document_store = QdrantDocumentStore(
        ":memory:",
        recreate_index=True,
        return_sparse_embedding=True,
        wait_result_from_api=True,
    )
    retriever = QdrantSparseRetriever(document_store=document_store)

    # using a fake sparse vector to keep the example simple
    retriever.run(query_sparse_embedding={"indices":[0, 1, 2, 3], "values":[0.1, 0.8, 0.05, 0.33]})
    ```
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
        Create a QdrantSparseRetriever component.

        :param document_store: An instance of QdrantDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space. Default is None.
        :param top_k: The maximum number of documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents or not. Default is True.
        :param return_embedding: Whether to return the sparse embedding of the retrieved Documents. Default is False.

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
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
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
    def from_dict(cls, data: Dict[str, Any]) -> "QdrantEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        document_store = QdrantDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_sparse_embedding: Dict[str, Union[List[int], List[float]]],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ):
        """
        Run the Sparse Embedding Retriever on the given input data.

        :param query_sparse_embedding: Sparse Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the scores of the retrieved documents or not.
        :param return_embedding: Whether to return the embedding of the retrieved Documents.
        :returns:
            The retrieved documents.

        """
        docs = self._document_store.query_by_sparse(
            query_sparse_embedding=query_sparse_embedding,
            filters=filters or self._filters,
            top_k=top_k or self._top_k,
            scale_score=scale_score or self._scale_score,
            return_embedding=return_embedding or self._return_embedding,
        )

        return {"documents": docs}

@component
class QdrantHybridRetriever:
    """
    A component for retrieving documents from an QdrantDocumentStore using hybrid search (dense+sparse).

    Usage example:
    ```python
    from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    document_store = QdrantDocumentStore(
        ":memory:",
        recreate_index=True,
        return_sparse_embedding=True,
        wait_result_from_api=True,
    )
    retriever = QdrantHybridRetriever(document_store=document_store)

    # using a fake sparse vector to keep the example simple
    retriever.run(query_embedding=[0.1]*768, query_sparse_embedding={"indices":[0, 1, 2, 3], "values":[0.1, 0.8, 0.05, 0.33]})
    ```
    """

    def __init__(
        self,
        document_store: QdrantDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        scale_score: bool = True,  # noqa: FBT001, FBT002
        return_embedding: bool = False,  # noqa: FBT001, FBT002
    ):
        """
        Create a QdrantSparseRetriever component.

        :param document_store: An instance of QdrantDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space. Default is None.
        :param top_k_dense: The maximum number of documents to retrieve by dense retriever. Default is 10.
        :param top_k_sparse: The maximum number of documents to retrieve by sparse retriever. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents or not. Default is True.
        :param return_embedding: Whether to return the sparse embedding of the retrieved Documents. Default is False.

        :raises ValueError: If 'document_store' is not an instance of QdrantDocumentStore.
        """

        if not isinstance(document_store, QdrantDocumentStore):
            msg = "document_store must be an instance of QdrantDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters
        self._top_k_dense = top_k_sparse
        self._top_k_sparse = top_k_sparse
        self._scale_score = scale_score
        self._return_embedding = return_embedding

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        d = default_to_dict(
            self,
            document_store=self._document_store,
            filters=self._filters,
            top_k_dense=self._top_k_dense,
            top_k_sparse=self._top_k_sparse,
            scale_score=self._scale_score,
            return_embedding=self._return_embedding,
        )
        d["init_parameters"]["document_store"] = self._document_store.to_dict()

        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdrantEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        document_store = QdrantDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_sparse_embedding: Dict[str, Union[List[int], List[float]]],
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ):
        """
        Run the Sparse Embedding Retriever on the given input data.

        :param query_sparse_embedding: Sparse Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the scores of the retrieved documents or not.
        :param return_embedding: Whether to return the embedding of the retrieved Documents.
        :returns:
            The retrieved documents.

        """
        docs = self._document_store.query_hybrid(
            query_sparse_embedding=query_sparse_embedding,
            query_embedding=query_embedding,
            filters=filters or self._filters,
            top_k_dense=top_k_dense or self._top_k_dense,
            top_k_sparse=top_k_sparse or self._top_k_sparse,
            scale_score=scale_score or self._scale_score,
            return_embedding=return_embedding or self._return_embedding,
        )

        return {"documents": docs}
