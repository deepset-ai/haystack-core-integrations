# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

logger = logging.getLogger(__name__)


@component
class OpenSearchSQLRetriever:
    """
    Executes raw OpenSearch SQL queries against an OpenSearchDocumentStore.

    This component allows you to execute SQL queries directly against the OpenSearch index,
    which is useful for fetching metadata, aggregations, and other structured data at runtime.

    Returns the raw JSON response from the OpenSearch SQL API.
    """

    def __init__(
        self,
        *,
        document_store: OpenSearchDocumentStore,
        raise_on_failure: bool = True,
        fetch_size: int | None = None,
    ):
        """
        Creates the OpenSearchSQLRetriever component.

        :param document_store: An instance of OpenSearchDocumentStore to use with the Retriever.
        :param raise_on_failure:
            Whether to raise an exception if the API call fails. Otherwise, log a warning and return None.
        :param fetch_size: Optional number of results to fetch per page. If not provided, the default
            fetch size set in OpenSearch is used.

        :raises ValueError: If `document_store` is not an instance of OpenSearchDocumentStore.
        """
        if not isinstance(document_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._raise_on_failure = raise_on_failure
        self._fetch_size = fetch_size

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            raise_on_failure=self._raise_on_failure,
            fetch_size=self._fetch_size,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenSearchSQLRetriever":
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

    @component.output_types(result=dict[str, Any])
    def run(
        self,
        query: str,
        document_store: OpenSearchDocumentStore | None = None,
        fetch_size: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Execute a raw OpenSearch SQL query against the index.

        :param query: The OpenSearch SQL query to execute.
        :param document_store: Optionally, an instance of OpenSearchDocumentStore to use with the Retriever.
        :param fetch_size: Optional number of results to fetch per page. If not provided, uses the value
            specified during initialization, or the default fetch size set in OpenSearch.

        :returns:
            A dictionary containing the raw JSON response from OpenSearch SQL API:
            - result: The raw JSON response from OpenSearch (dict) or None on error.

        Example:
            ```python
            retriever = OpenSearchSQLRetriever(document_store=document_store)
            result = retriever.run(
                query="SELECT content, category FROM my_index WHERE category = 'A'"
            )
            # result["result"] contains the raw OpenSearch JSON response
            # For regular queries: result["result"]["hits"]["hits"] contains documents
            # For aggregate queries: result["result"]["aggregations"] contains aggregations
            ```
        """
        if document_store is not None:
            if not isinstance(document_store, OpenSearchDocumentStore):
                msg = "document_store must be an instance of OpenSearchDocumentStore"
                raise ValueError(msg)
            doc_store = document_store
        else:
            doc_store = self._document_store

        fetch_size = fetch_size if fetch_size is not None else self._fetch_size

        try:
            result = doc_store._query_sql(query=query, fetch_size=fetch_size)
        except Exception as e:
            if self._raise_on_failure:
                raise e
            else:
                logger.warning(
                    "An error during SQL query execution occurred and will be ignored by returning None: {error}",
                    error=str(e),
                    exc_info=True,
                )
                result = {}

        return {"result": result}

    @component.output_types(result=dict[str, Any])
    async def run_async(
        self,
        query: str,
        document_store: OpenSearchDocumentStore | None = None,
        fetch_size: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Asynchronously execute a raw OpenSearch SQL query against the index.

        :param query: The OpenSearch SQL query to execute.
        :param document_store: Optionally, an instance of OpenSearchDocumentStore to use with the Retriever.
        :param fetch_size: Optional number of results to fetch per page. If not provided, uses the value
            specified during initialization, or the default fetch size set in OpenSearch.

        :returns:
            A dictionary containing the raw JSON response from OpenSearch SQL API:
            - result: The raw JSON response from OpenSearch (dict) or None on error.

        Example:
            ```python
            retriever = OpenSearchSQLRetriever(document_store=document_store)
            result = await retriever.run_async(
                query="SELECT content, category FROM my_index WHERE category = 'A'"
            )
            # result["result"] contains the raw OpenSearch JSON response
            # For regular queries: result["result"]["hits"]["hits"] contains documents
            # For aggregate queries: result["result"]["aggregations"] contains aggregations
            ```
        """
        if document_store is not None:
            if not isinstance(document_store, OpenSearchDocumentStore):
                msg = "document_store must be an instance of OpenSearchDocumentStore"
                raise ValueError(msg)
            doc_store = document_store
        else:
            doc_store = self._document_store

        fetch_size = fetch_size if fetch_size is not None else self._fetch_size

        try:
            result = await doc_store._query_sql_async(query=query, fetch_size=fetch_size)
        except Exception as e:
            if self._raise_on_failure:
                raise e
            else:
                logger.warning(
                    "An error during SQL query execution occurred and will be ignored by returning None: {error}",
                    error=str(e),
                    exc_info=True,
                )
                result = {}

        return {"result": result}
