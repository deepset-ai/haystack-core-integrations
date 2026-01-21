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

    Returns query results as a list of dictionaries (the _source from each hit).
    """

    def __init__(
        self,
        *,
        document_store: OpenSearchDocumentStore,
        raise_on_failure: bool = True,
    ):
        """
        Creates the OpenSearchSQLRetriever component.

        :param document_store: An instance of OpenSearchDocumentStore to use with the Retriever.
        :param raise_on_failure:
            Whether to raise an exception if the API call fails. Otherwise, log a warning and return None.

        :raises ValueError: If `document_store` is not an instance of OpenSearchDocumentStore.
        """
        if not isinstance(document_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._raise_on_failure = raise_on_failure

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

    @component.output_types(result=Any)
    def run(
        self,
        query: str,
        document_store: OpenSearchDocumentStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute a raw OpenSearch SQL query against the index.

        :param query: The OpenSearch SQL query to execute.
        :param document_store: Optionally, an instance of OpenSearchDocumentStore to use with the Retriever.

        :returns:
            A dictionary containing the query results with the following structure:
            - result: The query results as a list of dictionaries (the _source from each hit).

        Example:
            ```python
            retriever = OpenSearchSQLRetriever(document_store=document_store)
            result = retriever.run(
                query="SELECT content, category FROM my_index WHERE category = 'A'"
            )
            # result["result"] contains a list of dictionaries with the query results
            ```
        """
        if document_store is not None:
            if not isinstance(document_store, OpenSearchDocumentStore):
                msg = "document_store must be an instance of OpenSearchDocumentStore"
                raise ValueError(msg)
            doc_store = document_store
        else:
            doc_store = self._document_store

        try:
            result = doc_store._query_sql(query=query)
        except Exception as e:
            if self._raise_on_failure:
                raise e
            else:
                logger.warning(
                    "An error during SQL query execution occurred and will be ignored by returning None: {error}",
                    error=str(e),
                    exc_info=True,
                )
                result = None

        return {"result": result}

    @component.output_types(result=Any)
    async def run_async(
        self,
        query: str,
        document_store: OpenSearchDocumentStore | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously execute a raw OpenSearch SQL query against the index.

        :param query: The OpenSearch SQL query to execute.
        :param document_store: Optionally, an instance of OpenSearchDocumentStore to use with the Retriever.

        :returns:
            A dictionary containing the query results with the following structure:
            - result: The query results as a list of dictionaries (the _source from each hit).

        Example:
            ```python
            retriever = OpenSearchSQLRetriever(document_store=document_store)
            result = await retriever.run_async(
                query="SELECT content, category FROM my_index WHERE category = 'A'"
            )
            # result["result"] contains a list of dictionaries with the query results
            ```
        """
        if document_store is not None:
            if not isinstance(document_store, OpenSearchDocumentStore):
                msg = "document_store must be an instance of OpenSearchDocumentStore"
                raise ValueError(msg)
            doc_store = document_store
        else:
            doc_store = self._document_store

        try:
            result = await doc_store._query_sql_async(query=query)
        except Exception as e:
            if self._raise_on_failure:
                raise e
            else:
                logger.warning(
                    "An error during SQL query execution occurred and will be ignored by returning None: {error}",
                    error=str(e),
                    exc_info=True,
                )
                result = None

        return {"result": result}
