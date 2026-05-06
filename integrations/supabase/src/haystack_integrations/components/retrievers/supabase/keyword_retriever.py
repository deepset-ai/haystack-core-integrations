# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.pgvector import PgvectorKeywordRetriever
from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore


@component
class SupabasePgvectorKeywordRetriever(PgvectorKeywordRetriever):
    """
    Retrieves documents from the `SupabasePgvectorDocumentStore`, based on keywords.

    This is a thin wrapper around `PgvectorKeywordRetriever`, adapted for use with
    `SupabasePgvectorDocumentStore`.

    To rank the documents, the `ts_rank_cd` function of PostgreSQL is used.
    It considers how often the query terms appear in the document, how close together the terms are in the document,
    and how important is the part of the document where they occur.

    Example usage:

    # Set an environment variable `SUPABASE_DB_URL` with the connection string to your Supabase database.
    ```bash
    export SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:5432/postgres
    ```


    ```python
    from haystack import Document, Pipeline
    from haystack.document_stores.types.policy import DuplicatePolicy

    from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore
    from haystack_integrations.components.retrievers.supabase import SupabasePgvectorKeywordRetriever

    document_store = SupabasePgvectorDocumentStore(
        embedding_dimension=768,
        recreate_table=True,
    )

    documents = [Document(content="There are over 7,000 languages spoken around the world today."),
                 Document(content="Elephants have been observed to behave in a way that indicates..."),
                 Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

    document_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    retriever = SupabasePgvectorKeywordRetriever(document_store=document_store)
    result = retriever.run(query="languages")

    print(result['documents'][0].content)
    # >> "There are over 7,000 languages spoken around the world today."
    ```
    """

    def __init__(
        self,
        *,
        document_store: SupabasePgvectorDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize the SupabasePgvectorKeywordRetriever.

        :param document_store: An instance of `SupabasePgvectorDocumentStore`.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an instance of `SupabasePgvectorDocumentStore`.
        """
        if not isinstance(document_store, SupabasePgvectorDocumentStore):
            msg = "document_store must be an instance of SupabasePgvectorDocumentStore"
            raise ValueError(msg)

        super(SupabasePgvectorKeywordRetriever, self).__init__(  # noqa: UP008
            document_store=document_store,
            filters=filters,
            top_k=top_k,
            filter_policy=filter_policy,
        )

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "SupabasePgvectorKeywordRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data = copy.deepcopy(data)
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = SupabasePgvectorDocumentStore.from_dict(doc_store_params)
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)
