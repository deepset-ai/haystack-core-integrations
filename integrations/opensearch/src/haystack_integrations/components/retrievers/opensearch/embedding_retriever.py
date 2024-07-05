# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

logger = logging.getLogger(__name__)


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
        filter_policy: Optional[Union[str, FilterPolicy]] = FilterPolicy.REPLACE,
        custom_query: Optional[Dict[str, Any]] = None,
        raise_on_failure: bool = True,
    ):
        """
        Create the OpenSearchEmbeddingRetriever component.

        :param document_store: An instance of OpenSearchDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
            Filters are applied during the approximate kNN search to ensure that top_k matching documents are returned.
        :param top_k: Maximum number of Documents to return, defaults to 10
        :param filter_policy: Policy to determine how filters are applied.
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
        :param raise_on_failure:
            Whether to raise an exception if the API call fails. Otherwise log a warning and return an empty list.

        :raises ValueError: If `document_store` is not an instance of OpenSearchDocumentStore.
        """
        if not isinstance(document_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )
        self._custom_query = custom_query
        self._raise_on_failure = raise_on_failure

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
            filter_policy=self._filter_policy.value if self._filter_policy else None,
            custom_query=self._custom_query,
            raise_on_failure=self._raise_on_failure,
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
        if "filter_policy" in data["init_parameters"]:
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(data["init_parameters"]["filter_policy"])
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
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at document store initialization. See init method docstring for more
                        details.
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
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        top_k = top_k or self._top_k
        if filters is None:
            filters = self._filters
        if top_k is None:
            top_k = self._top_k
        if custom_query is None:
            custom_query = self._custom_query

        docs: List[Document] = []

        try:
            docs = self._document_store._embedding_retrieval(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                custom_query=custom_query,
            )
        except Exception as e:
            if self._raise_on_failure:
                raise e
            else:
                logger.warning(
                    "An error during embedding retrieval occurred and will be ignored by returning empty results: %s",
                    str(e),
                    exc_info=True,
                )

        return {"documents": docs}
