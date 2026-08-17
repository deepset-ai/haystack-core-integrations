# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore


@component
class MariaDBKeywordRetriever:
    """
    Retrieves documents from `MariaDBDocumentStore` using full-text keyword search.

    Uses MariaDB's `MATCH ... AGAINST` full-text search in natural language mode,
    backed by a FULLTEXT index on the `content` column.

    ### Usage example

    ```python
    from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore
    from haystack_integrations.components.retrievers.mariadb import MariaDBKeywordRetriever

    store = MariaDBDocumentStore(host="localhost", database="haystack", embedding_dimension=768)
    retriever = MariaDBKeywordRetriever(document_store=store, top_k=5)
    result = retriever.run(query="climate change")
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        *,
        document_store: MariaDBDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize the MariaDBKeywordRetriever.

        :param document_store: A `MariaDBDocumentStore` instance.
        :param filters: Default Haystack metadata filters.
        :param top_k: Maximum number of documents to return.
        :param filter_policy: How runtime filters interact with init-time filters.
        :raises ValueError: If `document_store` is not a `MariaDBDocumentStore`.
        """
        if not isinstance(document_store, MariaDBDocumentStore):
            msg = "document_store must be an instance of MariaDBDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MariaDBKeywordRetriever":
        """Deserialize the component from a dictionary."""
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = MariaDBDocumentStore.from_dict(doc_store_params)
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents matching the query via full-text search.

        :param query: The keyword query string.
        :param filters: Runtime filters merged with init-time filters per `filter_policy`.
        :param top_k: Override the retriever's `top_k`.
        :returns: Dictionary with `"documents"` key containing results ranked by relevance.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        return {
            "documents": self.document_store._keyword_retrieval(
                query=query,
                filters=filters,
                top_k=top_k or self.top_k,
            )
        }
