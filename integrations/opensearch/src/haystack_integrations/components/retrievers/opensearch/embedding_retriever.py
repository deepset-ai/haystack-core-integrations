# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


@component
class OpenSearchEmbeddingRetriever:
    """
    Uses a vector similarity metric to retrieve documents from the OpenSearchDocumentStore.

    Needs to be connected to the OpenSearchDocumentStore to run.
    """

    def __init__(
        self,
        *,
        document_store: OpenSearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        custom_query: Optional[Dict[str, Any]] = None,
    ):
        """
        Create the OpenSearchEmbeddingRetriever component.

        :param document_store: An instance of OpenSearchDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
            Filters are applied during the approximate kNN search to ensure that top_k matching documents are returned.
        :param top_k: Maximum number of Documents to return, defaults to 10
        :param custom_query: The query containing a mandatory `$query_embedding` and an optional `$filters` placeholder

            **An example custom_query:**

            ```python
            {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": "$query_embedding",   // mandatory query placeholder
                                        "k": 10000,
                                    }
                                }
                            }
                        ],
                        "filter": "$filters"                            // optional filter placeholder
                    }
                }
            }
            ```

        **For this custom_query, a sample `run()` could be:**

        ```python
        retriever.run(query_embedding=embedding,
                        filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
        ```

        :raises ValueError: If `document_store` is not an instance of OpenSearchDocumentStore.
        """
        if not isinstance(document_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._custom_query = custom_query

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
    def from_dict(cls, data: Dict[str, Any]) -> "OpenSearchEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = OpenSearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        custom_query: Optional[Dict[str, Any]] = None,
    ):
        """
        Retrieve documents using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: Optional filters to narrow down the search space.
        :param top_k: Maximum number of Documents to return.
        :param custom_query: The query containing a mandatory `$query_embedding` and an optional `$filters` placeholder

            **An example custom_query:**

            ```python
            {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": "$query_embedding",   // mandatory query placeholder
                                        "k": 10000,
                                    }
                                }
                            }
                        ],
                        "filter": "$filters"                            // optional filter placeholder
                    }
                }
            }
            ```

        **For this custom_query, a sample `run()` could be:**

        ```python
        retriever.run(query_embedding=embedding,
                        filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
        ```

        :returns:
            Dictionary with key "documents" containing the retrieved Documents.
            - documents: List of Document similar to `query_embedding`.
        """
        if filters is None:
            filters = self._filters
        if top_k is None:
            top_k = self._top_k
        if custom_query is None:
            custom_query = self._custom_query

        docs = self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            custom_query=custom_query,
        )
        return {"documents": docs}
