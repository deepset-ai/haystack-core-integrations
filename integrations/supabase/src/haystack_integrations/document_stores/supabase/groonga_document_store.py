# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from supabase import Client, create_client

logger = logging.getLogger(__name__)


class SupabaseGroongaDocumentStore:
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

        # Connect to Supabase
        resolved_key = supabase_key.resolve_value()
        self._client: Client = create_client(supabase_url, resolved_key)

        # Set up the table
        self._setup_table()

    def _setup_table(self) -> None:
        """
        Creates the documents table with PGroonga index if it does not exist.
        If recreate_table is True, drops and recreates the table.
        """
        if self.recreate_table:
            self._client.rpc(
                "exec_sql",
                {"query": f"DROP TABLE IF EXISTS {self.table_name};"}
            ).execute()

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
        result = self._client.table(self.table_name).select("id", count="exact").execute()
        return result.count or 0

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns documents matching the given filters.

        :param filters: Optional dictionary of filters.
        :returns: List of matching Document objects.
        """
        query = self._client.table(self.table_name).select("*")
        result = query.execute()
        return [self._to_haystack_document(row) for row in result.data]

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Writes documents to the store.

        :param documents: List of Haystack Document objects to write.
        :param policy: How to handle duplicate documents.
        :returns: Number of documents written.
        """
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
                existing = (
                    self._client.table(self.table_name)
                    .select("id")
                    .eq("id", doc.id)
                    .execute()
                )
                if not existing.data:
                    self._client.table(self.table_name).insert(row).execute()
                    written += 1
            elif policy == DuplicatePolicy.FAIL:
                existing = (
                    self._client.table(self.table_name)
                    .select("id")
                    .eq("id", doc.id)
                    .execute()
                )
                if existing.data:
                    raise DuplicateDocumentError(
                        f"Document with id {doc.id!r} already exists."
                    )
                self._client.table(self.table_name).insert(row).execute()
                written += 1
            else:
                self._client.table(self.table_name).insert(row).execute()
                written += 1

        return written

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents with the given IDs.

        :param document_ids: List of document IDs to delete.
        """
        if not document_ids:
            return
        self._client.table(self.table_name).delete().in_("id", document_ids).execute()

    def _groonga_retrieval(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Searches documents using PGroonga full-text search.

        :param query: The text query to search for.
        :param top_k: Maximum number of results to return.
        :param filters: Optional filters to apply.
        :returns: List of matching Document objects ranked by relevance.
        """
        search_sql = f"""
            SELECT id, content, meta,
                   pgroonga_score(tableoid, ctid) AS score
            FROM {self.table_name}
            WHERE content &@~ %s
            ORDER BY score DESC
            LIMIT %s;
        """
        result = self._client.rpc(
            "groonga_search",
            {"query_text": query, "table": self.table_name, "top_k": top_k}
        ).execute()

        return [self._to_haystack_document(row) for row in result.data]

    def _to_haystack_document(self, row: Dict[str, Any]) -> Document:
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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "SupabaseGroongaDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["supabase_key"])
        return default_from_dict(cls, data)