# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from postgrest import CountMethod

from supabase import Client, create_client

logger = logging.getLogger(__name__)


class SupabaseGroongaDocumentStore(DocumentStore):
    """
    A Document Store for Supabase using PGroonga for full-text search.

    PGroonga is a PostgreSQL extension for fast, multilingual full-text search.
    Unlike vector search, this store works with plain text queries — no embeddings needed.

    Prerequisites:
    - A Supabase project with PGroonga extension enabled.
    - Enable PGroonga in your Supabase project by running:
      `CREATE EXTENSION IF NOT EXISTS pgroonga;`

    Example usage:

    ```python
    from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore
    from haystack.utils import Secret

    document_store = SupabaseGroongaDocumentStore(
        supabase_url="https://<project>.supabase.co",
        supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
        table_name="haystack_fts_documents",
    )
    document_store.warm_up()
    ```
    """

    def __init__(
        self,
        *,
        supabase_url: str,
        supabase_key: Secret = Secret.from_env_var("SUPABASE_SERVICE_KEY"),
        table_name: str = "haystack_groonga_documents",
        recreate_table: bool = False,
    ) -> None:
        """
        Creates a new SupabaseGroongaDocumentStore instance.

        Note: Call warm_up() before using the store to initialize the client and table.

        :param supabase_url: The URL of your Supabase project.
            Format: `https://<project-ref>.supabase.co`
        :param supabase_key: The service role key for your Supabase project.
            Defaults to reading from the `SUPABASE_SERVICE_KEY` environment variable.
        :param table_name: The name of the table to store documents in.
            Defaults to `haystack_groonga_documents`.
        :param recreate_table: Whether to drop and recreate the table on startup.
            Defaults to `False`.
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.table_name = table_name
        self.recreate_table = recreate_table

        # Client is initialized lazily in warm_up()
        self._client: Client | None = None

    def warm_up(self) -> None:
        """
        Initializes the Supabase client and sets up the table.

        Must be called before using the document store.
        """
        key = self.supabase_key.resolve_value() or ""
        self._client = create_client(self.supabase_url, key)
        self._setup_table()

    def _setup_table(self) -> None:
        """
        Creates the documents table with PGroonga index if it does not exist.

        If recreate_table is True, drops and recreates the table.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)

        if self.recreate_table:
            self._client.rpc("exec_sql", {"query": f"DROP TABLE IF EXISTS {self.table_name};"}).execute()

        # Create table if not exists
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                content TEXT,
                meta JSONB,
                score REAL
            );
        """
        self._client.rpc("exec_sql", {"query": create_table_sql}).execute()

        # Create PGroonga index on content column
        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS pgroonga_{self.table_name}_index
            ON {self.table_name}
            USING pgroonga (content);
        """
        self._client.rpc("exec_sql", {"query": create_index_sql}).execute()

    def count_documents(self) -> int:
        """
        Returns the number of documents in the store.

        :returns: Number of documents.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)
        result = self._client.table(self.table_name).select("*", count=CountMethod.exact).execute()
        return int(result.count) if result.count is not None else 0

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns documents matching the given filters.

        Supported filters: equality filters on `id`, `content`, and `meta` fields.

        :param filters: Optional dictionary of filters.
            Example: ``{"field": "meta.language", "operator": "==", "value": "en"}``
        :returns: List of matching Document objects.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)

        query = self._client.table(self.table_name).select("*")

        if filters:
            query = self._apply_filters(query, filters)

        result = query.execute()
        return [self._to_haystack_document(row) for row in result.data if isinstance(row, dict)]

    def _apply_filters(self, query: Any, filters: dict[str, Any]) -> Any:
        """
        Applies filters to a Supabase query.

        :param query: The Supabase query builder.
        :param filters: Dictionary of filters to apply.
        :returns: The query with filters applied.
        """
        conditions = filters.get("conditions", [])

        for condition in conditions:
            field = condition.get("field", "")
            op = condition.get("operator", "==")
            value = condition.get("value")

            # Handle nested meta fields e.g. "meta.language"
            if field.startswith("meta."):
                meta_key = field[len("meta.") :]
                if op == "==":
                    query = query.eq(f"meta->>'{meta_key}'", value)
                elif op == "!=":
                    query = query.neq(f"meta->>'{meta_key}'", value)
            elif op == "==":
                query = query.eq(field, value)
            elif op == "!=":
                query = query.neq(field, value)
            elif op == "in":
                query = query.in_(field, value)

        return query

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ) -> int:
        """
        Writes documents to the store.

        :param documents: List of Haystack Document objects to write.
        :param policy: How to handle duplicate documents. Defaults to DuplicatePolicy.FAIL.
        :returns: Number of documents written.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)

        if not documents:
            return 0

        written = 0
        for doc in documents:
            row = {
                "id": doc.id,
                "content": doc.content or "",
                "meta": doc.meta or {},
                "score": None,
            }
            if policy == DuplicatePolicy.OVERWRITE:
                self._client.table(self.table_name).upsert(row).execute()
                written += 1
            elif policy == DuplicatePolicy.SKIP:
                existing = self._client.table(self.table_name).select("id").eq("id", doc.id).execute()
                if not existing.data:
                    self._client.table(self.table_name).insert(row).execute()
                    written += 1
            elif policy == DuplicatePolicy.FAIL:
                existing = self._client.table(self.table_name).select("id").eq("id", doc.id).execute()
                if existing.data:
                    msg = f"Document with id {doc.id!r} already exists."
                    raise DuplicateDocumentError(msg)
                self._client.table(self.table_name).insert(row).execute()
                written += 1
            else:
                self._client.table(self.table_name).insert(row).execute()
                written += 1

        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents with the given IDs.

        :param document_ids: List of document IDs to delete.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)

        if not document_ids:
            return
        self._client.table(self.table_name).delete().in_("id", document_ids).execute()

    def _groonga_retrieval(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Searches documents using PGroonga full-text search.

        :param query: The text query to search for.
        :param top_k: Maximum number of results to return.
        :param filters: Optional filters to apply after retrieval.
        :returns: List of matching Document objects ranked by relevance.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)

        result = self._client.rpc(
            "groonga_search",
            {"query_text": query, "table": self.table_name, "top_k": top_k},
        ).execute()

        data = result.data if isinstance(result.data, list) else []
        documents = [self._to_haystack_document(row) for row in data if isinstance(row, dict)]

        # Apply filters post-retrieval if provided
        if filters:
            documents = self._filter_documents_in_memory(documents, filters)

        return documents

    def _filter_documents_in_memory(self, documents: list[Document], filters: dict[str, Any]) -> list[Document]:
        """
        Filters a list of documents in memory based on the given filters.

        :param documents: List of documents to filter.
        :param filters: Dictionary of filters to apply.
        :returns: Filtered list of documents.
        """
        conditions = filters.get("conditions", [])
        filtered = []

        for doc in documents:
            match = True
            for condition in conditions:
                field = condition.get("field", "")
                op = condition.get("operator", "==")
                value = condition.get("value")

                if field.startswith("meta."):
                    meta_key = field[len("meta.") :]
                    doc_value = doc.meta.get(meta_key)
                else:
                    doc_value = getattr(doc, field, None)

                if op == "==" and doc_value != value:
                    match = False
                    break
                elif op == "!=" and doc_value == value:
                    match = False
                    break
                elif op == "in" and doc_value not in value:
                    match = False
                    break

            if match:
                filtered.append(doc)

        return filtered

    def _to_haystack_document(self, row: dict[str, Any]) -> Document:
        """
        Converts a database row dictionary into a Haystack Document.

        :param row: Dictionary from database result.
        :returns: Haystack Document object.
        """
        return Document(
            id=row["id"],
            content=row.get("content"),
            meta=row.get("meta") or {},
            score=row.get("score"),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            supabase_url=self.supabase_url,
            supabase_key=self.supabase_key.to_dict(),
            table_name=self.table_name,
            recreate_table=self.recreate_table,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupabaseGroongaDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["supabase_key"])
        return default_from_dict(cls, data)
