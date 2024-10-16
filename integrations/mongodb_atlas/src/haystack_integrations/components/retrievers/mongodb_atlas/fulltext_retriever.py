from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@component
class MongoDBAtlasFullTextRetriever:

    def __init__(
        self,
        *,
        document_store: MongoDBAtlasDocumentStore,
        search_path: Union[str, List[str]] = "content",
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        Create the MongoDBAtlasFullTextRetriever component.

        :param document_store: An instance of MongoDBAtlasDocumentStore.
        :param search_path: Field(s) to search within, e.g., "content" or ["content", "title"].
        :param filters: Filters applied to the retrieved Documents. Make sure that the fields used in the filters are
            included in the configuration of the `vector_search_index`. The configuration must be done manually
            in the Web UI of MongoDB Atlas.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an instance of `MongoDBAtlasDocumentStore`.
        """

        if not isinstance(document_store, MongoDBAtlasDocumentStore):
            msg = "document_store must be an instance of MongoDBAtlasDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.search_path = search_path
        self.filter_policy = (
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
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasFullTextRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
              Deserialized component.
        """
        data["init_parameters"]["document_store"] = MongoDBAtlasDocumentStore.from_dict(
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
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents from the MongoDBAtlasDocumentStore, based on the provided query.

        :param query: Text query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of Documents to return. Overrides the value specified at initialization.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given `query`
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k

        docs = self.document_store._fulltext_retrieval(
            query=query, filters=filters, top_k=top_k, search_path=self.search_path
        )
        return {"documents": docs}
