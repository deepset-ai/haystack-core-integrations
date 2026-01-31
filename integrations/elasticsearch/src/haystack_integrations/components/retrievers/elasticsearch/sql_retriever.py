# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

logger = logging.getLogger(__name__)


@component
class ElasticsearchSQLRetriever:
    """
    Executes raw Elasticsearch SQL queries against an ElasticsearchDocumentStore.

    This component allows you to execute SQL queries directly against the Elasticsearch index,
    which is useful for fetching metadata, aggregations, and other structured data at runtime.

    Returns the raw JSON response from the Elasticsearch SQL API.

    Usage example:
    ```python
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchSQLRetriever

    document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
    retriever = ElasticsearchSQLRetriever(document_store=document_store)

    result = retriever.run(
        query="SELECT content, category FROM \\"my_index\\" WHERE category = 'A'"
    )
    # result["result"] contains the raw Elasticsearch JSON response
    ```
    """

    def __init__(
        self,
        *,
        document_store: ElasticsearchDocumentStore,
        raise_on_failure: bool = True,
        fetch_size: int | None = None,
    ):
        """
        Creates the ElasticsearchSQLRetriever component.

        :param document_store: An instance of ElasticsearchDocumentStore to use with the Retriever.
        :param raise_on_failure:
            Whether to raise an exception if the API call fails. Otherwise, log a warning and return an empty dict.
        :param fetch_size: Optional number of results to fetch per page. If not provided, the default
            fetch size set in Elasticsearch is used.

        :raises ValueError: If `document_store` is not an instance of ElasticsearchDocumentStore.
        """
        if not isinstance(document_store, ElasticsearchDocumentStore):
            msg = "document_store must be an instance of ElasticsearchDocumentStore"
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
    def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchSQLRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = ElasticsearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(result=dict[str, Any])
    def run(
        self,
        query: str,
        document_store: ElasticsearchDocumentStore | None = None,
        fetch_size: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Execute a raw Elasticsearch SQL query against the index.

        :param query: The Elasticsearch SQL query to execute.
        :param document_store: Optionally, an instance of ElasticsearchDocumentStore to use with the Retriever.
        :param fetch_size: Optional number of results to fetch per page. If not provided, uses the value
            specified during initialization, or the default fetch size set in Elasticsearch.

        :returns:
            A dictionary containing the raw JSON response from Elasticsearch SQL API:
            - result: The raw JSON response from Elasticsearch (dict) or empty dict on error.

        Example:
            ```python
            retriever = ElasticsearchSQLRetriever(document_store=document_store)
            result = retriever.run(
                query="SELECT content, category FROM \\"my_index\\" WHERE category = 'A'"
            )
            # result["result"] contains the raw Elasticsearch JSON response
            # result["result"]["columns"] contains column metadata
            # result["result"]["rows"] contains the data rows
            ```
        """
        if document_store is not None:
            if not isinstance(document_store, ElasticsearchDocumentStore):
                msg = "document_store must be an instance of ElasticsearchDocumentStore"
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
                    "An error during SQL query execution occurred and will be ignored by returning empty dict: {error}",
                    error=str(e),
                    exc_info=True,
                )
                result = {}

        return {"result": result}

    @component.output_types(result=dict[str, Any])
    async def run_async(
        self,
        query: str,
        document_store: ElasticsearchDocumentStore | None = None,
        fetch_size: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Asynchronously execute a raw Elasticsearch SQL query against the index.

        :param query: The Elasticsearch SQL query to execute.
        :param document_store: Optionally, an instance of ElasticsearchDocumentStore to use with the Retriever.
        :param fetch_size: Optional number of results to fetch per page. If not provided, uses the value
            specified during initialization, or the default fetch size set in Elasticsearch.

        :returns:
            A dictionary containing the raw JSON response from Elasticsearch SQL API:
            - result: The raw JSON response from Elasticsearch (dict) or empty dict on error.

        Example:
            ```python
            retriever = ElasticsearchSQLRetriever(document_store=document_store)
            result = await retriever.run_async(
                query="SELECT content, category FROM \\"my_index\\" WHERE category = 'A'"
            )
            # result["result"] contains the raw Elasticsearch JSON response
            # result["result"]["columns"] contains column metadata
            # result["result"]["rows"] contains the data rows
            ```
        """
        if document_store is not None:
            if not isinstance(document_store, ElasticsearchDocumentStore):
                msg = "document_store must be an instance of ElasticsearchDocumentStore"
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
                    "An error during SQL query execution occurred and will be ignored by returning empty dict: {error}",
                    error=str(e),
                    exc_info=True,
                )
                result = {}

        return {"result": result}
