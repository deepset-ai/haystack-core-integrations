# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore


@component
class FalkorDBCypherRetriever:
    """
    A power-user retriever for executing arbitrary OpenCypher queries against FalkorDB.

    This retriever allows you to leverage graph traversal and multi-hop queries in
    GraphRAG pipelines. The query must return nodes or dictionaries that can be
    mapped exactly to a Haystack `Document`.

    **Security Warning:** Raw Cypher queries must only come from trusted sources. Do
    not use un-sanitised user input directly in query strings. Use `parameters` instead.

    Usage example:
    ```python
    from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore
    from haystack_integrations.components.retrievers.falkordb import FalkorDBCypherRetriever

    store = FalkorDBDocumentStore(host="localhost", port=6379)
    retriever = FalkorDBCypherRetriever(
        document_store=store,
        custom_cypher_query="MATCH (d:Document)-[:RELATES_TO]->(:Concept {name: $concept}) RETURN d"
    )

    res = retriever.run(parameters={"concept": "GraphRAG"})
    print(res["documents"])
    ```
    """

    def __init__(
        self,
        document_store: FalkorDBDocumentStore,
        custom_cypher_query: str | None = None,
    ) -> None:
        """
        Create a new FalkorDBCypherRetriever.

        :param document_store: The FalkorDBDocumentStore instance.
        :param custom_cypher_query: A static OpenCypher query to execute. Can be
            overridden at runtime by passing `query` to `run()`.
        :raises ValueError: If the provided `document_store` is not a `FalkorDBDocumentStore`.
        """
        if not isinstance(document_store, FalkorDBDocumentStore):
            msg = "document_store must be an instance of FalkorDBDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.custom_cypher_query = custom_cypher_query

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the retriever to a dictionary.

        :returns: Dictionary representation of the retriever.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            custom_cypher_query=self.custom_cypher_query,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FalkorDBCypherRetriever":
        """
        Deserialise a `FalkorDBCypherRetriever` produced by `to_dict`.

        :param data: Serialised retriever dictionary.
        :returns: Reconstructed `FalkorDBCypherRetriever` instance.
        """
        init_params = data["init_parameters"]
        init_params["document_store"] = FalkorDBDocumentStore.from_dict(init_params["document_store"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents by executing an OpenCypher query.

        If a `query` is provided here, it overrides the `custom_cypher_query`
        set during initialisation.

        :param query: Optional OpenCypher query string.
        :param parameters: Optional dictionary of query parameters (referenced as
            `$param_name` in the Cypher string).
        :raises ValueError: If no query string is provided (both here and at init).
        :returns: Dictionary containing a `"documents"` key with the retrieved documents.
        """
        cypher = query or self.custom_cypher_query
        if not cypher:
            msg = "A Cypher query string must be provided either at init or at runtime."
            raise ValueError(msg)

        docs = self.document_store._cypher_retrieval(
            cypher_query=cypher,
            parameters=parameters,
        )

        return {"documents": docs}
