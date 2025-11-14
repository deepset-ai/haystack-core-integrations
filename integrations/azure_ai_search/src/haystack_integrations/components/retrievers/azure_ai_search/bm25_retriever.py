from typing import Any, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore, _normalize_filters

logger = logging.getLogger(__name__)


@component
class AzureAISearchBM25Retriever:
    """
    Retrieves documents from the AzureAISearchDocumentStore using BM25 retrieval.
    Must be connected to the AzureAISearchDocumentStore to run.

    """

    def __init__(
        self,
        *,
        document_store: AzureAISearchDocumentStore,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
        **kwargs: Any,
    ):
        """
        Create the AzureAISearchBM25Retriever component.

        :param document_store: An instance of AzureAISearchDocumentStore to use with the Retriever.
        :param filters: Filters applied when fetching documents from the Document Store.
            Filters are applied during the BM25 search to ensure the Retriever returns
              `top_k` matching documents.
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
        :raises TypeError: If the document store is not an instance of AzureAISearchDocumentStore.
        :raises RuntimeError: If the query is not valid, or if the document store is not correctly configured.

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
            raise TypeError(message)

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
            document_store=self._document_store.to_dict(),
            filter_policy=self._filter_policy.value,
            **self._kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AzureAISearchBM25Retriever":
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

    @component.output_types(documents=list[Document])
    def run(
        self, query: str, filters: Optional[dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> dict[str, list[Document]]:
        """Retrieve documents from the AzureAISearchDocumentStore.

        :param query: Text of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: the maximum number of documents to retrieve.
        :raises RuntimeError: If an error occurs during the BM25 retrieval process.
        :returns: a dictionary with the following keys:
            - `documents`: A list of documents retrieved from the AzureAISearchDocumentStore.
        """

        top_k = top_k or self._top_k
        filters = filters or self._filters

        normalized_filters = ""
        applied_filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        if applied_filters:
            normalized_filters = _normalize_filters(applied_filters)

        try:
            docs = self._document_store._bm25_retrieval(
                query=query,
                filters=normalized_filters,
                top_k=top_k,
                **self._kwargs,
            )
        except Exception as e:
            msg = (
                "An error occurred during the bm25 retrieval process from the AzureAISearchDocumentStore. "
                "Ensure that the query is valid and the document store is correctly configured."
            )
            raise RuntimeError(msg) from e

        return {"documents": docs}
