# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: FBT001  Boolean-typed positional argument in function definition

from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

logger = logging.getLogger(__name__)


@component
class OpenSearchEmbeddingRetriever:
    """
    Retrieves documents from the OpenSearchDocumentStore using a vector similarity metric.

     Must be connected to the OpenSearchDocumentStore to run.
    """

    def __init__(
        self,
        *,
        document_store: OpenSearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
        custom_query: Optional[Dict[str, Any]] = None,
        raise_on_failure: bool = True,
        efficient_filtering: bool = False,
    ):
        """
        Create the OpenSearchEmbeddingRetriever component.

        :param document_store: An instance of OpenSearchDocumentStore to use with the Retriever.
        :param filters: Filters applied when fetching documents from the Document Store.
            Filters are applied during the approximate kNN search to ensure the Retriever returns
              `top_k` matching documents.
        :param top_k: Maximum number of documents to return.
        :param filter_policy: Policy to determine how filters are applied. Possible options:
        - `merge`: Runtime filters are merged with initialization filters.
        - `replace`: Runtime filters replace initialization filters. Use this policy to change the filtering scope.
        :param custom_query: The custom OpenSearch query containing a mandatory `$query_embedding` and
          an optional `$filters` placeholder.

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

        For this `custom_query`, an example `run()` could be:

        ```python
        retriever.run(
            query_embedding=embedding,
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.years", "operator": "==", "value": "2019"},
                    {"field": "meta.quarters", "operator": "in", "value": ["Q1", "Q2"]},
                ],
            },
        )
        ```
        :param raise_on_failure:
            If `True`, raises an exception if the API call fails.
            If `False`, logs a warning and returns an empty list.
        :param efficient_filtering: If `True`, the filter will be applied during the approximate kNN search.
            This is only supported for knn engines "faiss" and "lucene" and does not work with the default "nmslib".

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
        self._efficient_filtering = efficient_filtering

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
            custom_query=self._custom_query,
            raise_on_failure=self._raise_on_failure,
            efficient_filtering=self._efficient_filtering,
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

        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
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
        efficient_filtering: Optional[bool] = None,
        document_store: Optional[OpenSearchDocumentStore] = None,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied when fetching documents from the Document Store.
            Filters are applied during the approximate kNN search to ensure the Retriever returns `top_k` matching
            documents.
            The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
        :param top_k: Maximum number of documents to return.
        :param custom_query: A custom OpenSearch query containing a mandatory `$query_embedding` and an
          optional `$filters` placeholder.

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

        For this `custom_query`, an example `run()` could be:

        ```python
        retriever.run(
            query_embedding=embedding,
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.years", "operator": "==", "value": "2019"},
                    {"field": "meta.quarters", "operator": "in", "value": ["Q1", "Q2"]},
                ],
            },
        )
        ```

        :param efficient_filtering: If `True`, the filter will be applied during the approximate kNN search.
            This is only supported for knn engines "faiss" and "lucene" and does not work with the default "nmslib".
        :param document_store: Optional instance of OpenSearchDocumentStore to use with the Retriever.

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
        if efficient_filtering is None:
            efficient_filtering = self._efficient_filtering

        docs: List[Document] = []

        if document_store is not None:
            if not isinstance(document_store, OpenSearchDocumentStore):
                msg = "document_store must be an instance of OpenSearchDocumentStore"
                raise ValueError(msg)
            doc_store = document_store
        else:
            doc_store = self._document_store

        try:
            docs = doc_store._embedding_retrieval(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                custom_query=custom_query,
                efficient_filtering=efficient_filtering,
            )
        except Exception as e:
            if self._raise_on_failure:
                raise e
            else:
                logger.warning(
                    "An error during embedding retrieval occurred and will be ignored by returning empty results:"
                    "{error}",
                    error=str(e),
                    exc_info=True,
                )

        return {"documents": docs}

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        custom_query: Optional[Dict[str, Any]] = None,
        efficient_filtering: Optional[bool] = None,
        document_store: Optional[OpenSearchDocumentStore] = None,
    ) -> Dict[str, List[Document]]:
        """
        Asynchronously retrieve documents using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied when fetching documents from the Document Store.
            Filters are applied during the approximate kNN search to ensure the Retriever
              returns `top_k` matching documents.
            The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
        :param top_k: Maximum number of documents to return.
        :param custom_query: A custom OpenSearch query containing a mandatory `$query_embedding` and an
          optional `$filters` placeholder.

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

        For this `custom_query`, an example `run()` could be:

        ```python
        retriever.run(
            query_embedding=embedding,
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.years", "operator": "==", "value": "2019"},
                    {"field": "meta.quarters", "operator": "in", "value": ["Q1", "Q2"]},
                ],
            },
        )
        ```

        :param efficient_filtering: If `True`, the filter will be applied during the approximate kNN search.
            This is only supported for knn engines "faiss" and "lucene" and does not work with the default "nmslib".
        :param document_store: Optional instance of OpenSearchDocumentStore to use with the Retriever.

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
        if efficient_filtering is None:
            efficient_filtering = self._efficient_filtering

        docs: List[Document] = []

        if document_store is not None:
            if not isinstance(document_store, OpenSearchDocumentStore):
                msg = "document_store must be an instance of OpenSearchDocumentStore"
                raise ValueError(msg)
            doc_store = document_store
        else:
            doc_store = self._document_store

        try:
            docs = await doc_store._embedding_retrieval_async(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                custom_query=custom_query,
                efficient_filtering=efficient_filtering,
            )
        except Exception as e:
            if self._raise_on_failure:
                raise e
            logger.warning(
                "An error during embedding retrieval occurred and will be ignored by returning empty results: {error}",
                error=str(e),
                exc_info=True,
            )

        return {"documents": docs}
