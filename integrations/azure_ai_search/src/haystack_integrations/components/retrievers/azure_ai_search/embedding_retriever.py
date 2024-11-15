import logging
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore, _normalize_filters

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
        **kwargs,
    ):
        """
        Create the AzureAISearchEmbeddingRetriever component.

        :param document_store: An instance of AzureAISearchDocumentStore to use with the Retriever.
        :param filters: Filters applied when fetching documents from the Document Store.
        :param top_k: Maximum number of documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        :param kwargs: Additional keyword arguments to pass to the Azure AI's search endpoint.
            Some of the supported parameters:
                - `query_type`: A string indicating the type of query to perform. Possible values are
                'simple','full' and 'semantic'.
                - `semantic_configuration_name`: The name of semantic configuration to be used when
                processing semantic queries.
            For more information on parameters, see the
            [official Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/).

        """
        self._filters = filters or {}
        self._top_k = top_k
        self._document_store = document_store
        self._filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )
        self._kwargs = kwargs

        if not isinstance(document_store, AzureAISearchDocumentStore):
            message = "document_store must be an instance of AzureAISearchDocumentStore"
            raise Exception(message)

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
            filter_policy=self._filter_policy.value,
            **self._kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureAISearchEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = AzureAISearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )

        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if "filter_policy" in data["init_parameters"]:
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(data["init_parameters"]["filter_policy"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """Retrieve documents from the AzureAISearchDocumentStore.

        :param query_embedding: A list of floats representing the query embedding.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                the `filter_policy` chosen at retriever initialization. See `__init__` method docstring for more
                details.
        :param top_k: The maximum number of documents to retrieve.
        :returns: Dictionary with the following keys:
                - `documents`: A list of documents retrieved from the AzureAISearchDocumentStore.
        """

        top_k = top_k or self._top_k
        if filters is not None:
            applied_filters = apply_filter_policy(self._filter_policy, self._filters, filters)
            normalized_filters = _normalize_filters(applied_filters)
        else:
            normalized_filters = ""

        try:
            docs = self._document_store._embedding_retrieval(
                query_embedding=query_embedding, filters=normalized_filters, top_k=top_k, **self._kwargs
            )
        except Exception as e:
            msg = (
                "An error occurred during the embedding retrieval process from the AzureAISearchDocumentStore. "
                "Ensure that the query embedding is valid and the document store is correctly configured."
            )
            raise RuntimeError(msg) from e

        return {"documents": docs}
