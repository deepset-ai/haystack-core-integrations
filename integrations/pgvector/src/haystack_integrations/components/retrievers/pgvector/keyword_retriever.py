# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@component
class PgvectorKeywordRetriever:
    """
    Retrieve documents from the `PgvectorDocumentStore`, based on keywords.

    To rank the documents, the `ts_rank_cd` function of PostgreSQL is used.
    It considers how often the query terms appear in the document, how close together the terms are in the document,
    and how important is the part of the document where they occur.
    For more details, see
    [Postgres documentation](https://www.postgresql.org/docs/current/textsearch-controls.html#TEXTSEARCH-RANKING).

    Usage example:
    ```python
    from haystack.document_stores import DuplicatePolicy
    from haystack import Document

    from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
    from haystack_integrations.components.retrievers.pgvector import PgvectorKeywordRetriever

    # Set an environment variable `PG_CONN_STR` with the connection string to your PostgreSQL database.
    # e.g., "postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"

    document_store = PgvectorDocumentStore(language="english", recreate_table=True)

    documents = [Document(content="There are over 7,000 languages spoken around the world today."),
        Document(content="Elephants have been observed to behave in a way that indicates..."),
        Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

    document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

    retriever = PgvectorKeywordRetriever(document_store=document_store)

    result = retriever.run(query="languages")

    assert res['retriever']['documents'][0].content == "There are over 7,000 languages spoken around the world today."
    """

    def __init__(
        self,
        *,
        document_store: PgvectorDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        :param document_store: An instance of `PgvectorDocumentStore`.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an instance of `PgvectorDocumentStore`.
        """
        if not isinstance(document_store, PgvectorDocumentStore):
            msg = "document_store must be an instance of PgvectorDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
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
    def from_dict(cls, data: Dict[str, Any]) -> "PgvectorKeywordRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = PgvectorDocumentStore.from_dict(doc_store_params)
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
    ):
        """
        Retrieve documents from the `PgvectorDocumentStore`, based on keywords.

        :param query: String to search in `Document`s' content.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of Documents to return.

        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s that match the query.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)

        top_k = top_k or self.top_k

        docs = self.document_store._keyword_retrieval(
            query=query,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
