import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from azure.search.documents.models import VectorizedQuery
from haystack import Document, component
from haystack.document_stores.types import FilterPolicy
from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore

# from haystack.components.embedders import AzureOpenAIDocumentEmbedder, AzureOpenAITextEmbedder
# from .vectorizer import create_vectorizer, get_document_emebeddings, get_text_embeddings

logger = logging.getLogger(__name__)


@component
class AzureAISearchEmbeddingRetriever:
    """
    Retrieves documents from the AzureAISearchDocumentStore using a vector similarity metric.

     Must be connected to the AzureAISearchDocumentStore to run.
    """

    def __init__(
        self,
        *,
        document_store: AzureAISearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        Create the AzureAISearchEmbeddingRetriever component.

        :param document_store: An instance of AzureAISearchDocumentStore to use with the Retriever.
        :param filters: Filters applied when fetching documents from the Document Store.
            Filters are applied during the approximate kNN search to ensure the Retriever returns
              `top_k` matching documents.
        :param top_k: Maximum number of documents to return.

        """
        self.filters = filters or {}
        self.top_k = top_k
        self.document_store = document_store
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

        if not isinstance(document_store, AzureAISearchDocumentStore):
            message = "document_store must be an instance of AstraDocumentStore"
            raise Exception(message)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """Retrieve documents from the AzureAISearchDocumentStore.

        :param query_embedding: floats representing the query embedding
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: the maximum number of documents to retrieve.
        :returns: a dictionary with the following keys:
            - `documents`: A list of documents retrieved from the AzureAISearchDocumentStore.
        """
        # filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k

        return {"documents": self._vector_search(query_embedding, top_k, filters=filters)}

    def _vector_search(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 10,
        fields: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.
        It uses the vector configuration of the document store. By default it uses the HNSW algorithm with cosine similarity.

        This method is not meant to be part of the public interface of
        `AzureAISearchDocumentStore` nor called directly.
        `AzureAISearchEmbeddingRetriever` uses this method directly and is the public interface for it.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
            Filters are applied during the approximate kNN search to ensure that top_k matching documents are returned.
        :param top_k: Maximum number of Documents to return, defaults to 10

        :raises ValueError: If `query_embedding` is an empty list
        :returns: List of Document that are most similar to `query_embedding`
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        # embedding = get_embeddings(input=query, model=embedding_model_name, dimensions=self._embedding_dimension)

        vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=3, fields="embeddings")

        results = self.client.search(search_text=None, vector_queries=[vector_query], select=fields)

        return results
