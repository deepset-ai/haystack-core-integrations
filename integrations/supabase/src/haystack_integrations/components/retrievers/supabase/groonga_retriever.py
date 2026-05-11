# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore


@component
class SupabaseGroongaRetriever:
    """
        Retrieves documents from SupabaseGroongaDocumentStore using PGroonga full-text search.

        This retriever works without embeddings — it searches documents using plain text queries.
        It can be used alongside SupabasePgvectorEmbeddingRetriever in hybrid search pipelines.

        Example usage:

    ```python
        from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore
        from haystack_integrations.components.retrievers.supabase import SupabaseGroongaRetriever
        from haystack.utils import Secret

        document_store = SupabaseGroongaDocumentStore(
            supabase_url="https://<project>.supabase.co",
            supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
            table_name="haystack_fts_documents",
        )

        retriever = SupabaseGroongaRetriever(document_store=document_store, top_k=10)
        result = retriever.run(query="python programming")
        print(result["documents"])
    ```
    """

    def __init__(
        self,
        *,
        document_store: SupabaseGroongaDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize the SupabaseGroongaRetriever.

        :param document_store: An instance of SupabaseGroongaDocumentStore.
        :param filters: Optional filters applied to retrieved Documents.
        :param top_k: Maximum number of Documents to return. Defaults to 10.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If document_store is not an instance of SupabaseGroongaDocumentStore.
        """
        if not isinstance(document_store, SupabaseGroongaDocumentStore):
            msg = "document_store must be an instance of SupabaseGroongaDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Runs the retriever on the given query.

        :param query: The text query to search for.
        :param filters: Optional runtime filters. Merged or replaced based on filter_policy.
        :param top_k: Optional override for maximum number of documents to return.
        :returns: Dictionary with key "documents" containing list of matching Documents.
        """
        if not query:
            return {"documents": []}

        # Handle filter policy
        if filters is not None:
            if self.filter_policy == FilterPolicy.MERGE:
                merged_filters = {**self.filters, **filters}
            else:
                merged_filters = filters
        else:
            merged_filters = self.filters

        effective_top_k = top_k if top_k is not None else self.top_k

        documents = self.document_store._groonga_retrieval(
            query=query,
            top_k=effective_top_k,
            filters=merged_filters,
        )

        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupabaseGroongaRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        data = copy.deepcopy(data)
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = SupabaseGroongaDocumentStore.from_dict(doc_store_params)
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)
