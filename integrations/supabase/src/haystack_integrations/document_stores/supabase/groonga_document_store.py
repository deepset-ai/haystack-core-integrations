# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import datetime as _datetime
from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.errors import FilterError
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from postgrest import CountMethod

from supabase import AsyncClient, Client, acreate_client, create_client

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
        supabase_key: Secret = Secret.from_env_var("SUPABASE_SERVICE_KEY", strict=False),
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
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", table_name):
            msg = f"Invalid table_name {table_name!r}: must match [a-zA-Z_][a-zA-Z0-9_]*"
            raise ValueError(msg)

        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.table_name = table_name
        self.recreate_table = recreate_table

        # Sync client initialized in warm_up(); async client initialized lazily on first use
        self._client: Client | None = None
        self._async_client: AsyncClient | None = None

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

    async def _initialize_async_client(self) -> None:
        """Lazily creates the async Supabase client and sets up the table on first use."""
        if self._async_client is None:
            key = self.supabase_key.resolve_value() or ""
            self._async_client = await acreate_client(self.supabase_url, key)
            await self._setup_table_async()

    async def _setup_table_async(self) -> None:
        """Async counterpart of _setup_table; called by _initialize_async_client."""
        if self._async_client is None:
            msg = "Async client not initialized."
            raise RuntimeError(msg)

        if self.recreate_table:
            await self._async_client.rpc("exec_sql", {"query": f"DROP TABLE IF EXISTS {self.table_name};"}).execute()

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                content TEXT,
                meta JSONB,
                score REAL
            );
        """
        await self._async_client.rpc("exec_sql", {"query": create_table_sql}).execute()

        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS pgroonga_{self.table_name}_index
            ON {self.table_name}
            USING pgroonga (content);
        """
        await self._async_client.rpc("exec_sql", {"query": create_index_sql}).execute()

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

    async def count_documents_async(self) -> int:
        """
        Async version of count_documents.

        :returns: Number of documents.
        """
        await self._initialize_async_client()
        result = await self._async_client.table(self.table_name).select("*", count=CountMethod.exact).execute()  # type: ignore[union-attr]
        return int(result.count) if result.count is not None else 0

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns documents matching the given filters.

        Supports the standard Haystack filter syntax with the following operators:

        - Comparison: ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``, ``in``, ``not in``
        - Logical: ``AND``, ``OR``, ``NOT`` (``OR`` and ``NOT`` support simple conditions
          only — no nested logical operators inside them)

        **Known limitation:** For ``!=`` and ``not in`` on ``meta.*`` fields, documents
        where the field is absent are included in the result (matching Python ``None != value``
        semantics). For ``>`` / ``>=`` / ``<`` / ``<=``, documents where the field is absent
        are excluded (SQL ``NULL`` comparison semantics).

        :param filters: Optional Haystack filter dict.
            Simple comparison: ``{"field": "meta.language", "operator": "==", "value": "en"}``
            Logical: ``{"operator": "AND", "conditions": [...]}``
        :returns: List of matching Document objects.
        :raises FilterError: If the filter structure is malformed or uses an unsupported operator.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)

        query = self._client.table(self.table_name).select("*")

        if filters:
            query = SupabaseGroongaDocumentStore._apply_filters(query, filters)

        result = query.execute()
        return [self._to_haystack_document(row) for row in result.data if isinstance(row, dict)]

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Async version of filter_documents.

        :param filters: Optional Haystack filter dict.
        :returns: List of matching Document objects.
        :raises FilterError: If the filter structure is malformed or uses an unsupported operator.
        """
        await self._initialize_async_client()
        query = self._async_client.table(self.table_name).select("*")  # type: ignore[union-attr]

        if filters:
            query = SupabaseGroongaDocumentStore._apply_filters(query, filters)

        result = await query.execute()
        return [self._to_haystack_document(row) for row in result.data if isinstance(row, dict)]

    @staticmethod
    def _meta_col(field: str, value: Any) -> str:
        """
        Choose the PostgREST column expression for a meta field.

        Uses the JSONB accessor (->) for numeric values so that PostgREST performs
        correct numeric comparison. Uses the text accessor (->>) for strings, booleans,
        None, and mixed lists, which return the JSON value as text.
        """
        if not field.startswith("meta."):
            return field
        key = field[len("meta.") :]
        if isinstance(value, list):
            all_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value if v is not None)
            return f"meta->{key}" if (all_numeric and value) else f"meta->>{key}"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"meta->{key}"
        return f"meta->>{key}"

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Convert Python booleans to lowercase strings compatible with JSONB text accessor."""
        if isinstance(value, bool):
            return "true" if value else "false"
        return value

    @staticmethod
    def _apply_filters(query: Any, filters: dict[str, Any]) -> Any:
        """
        Applies Haystack filters to a PostgREST query builder.

        Supports AND, OR, NOT logical operators and all standard comparison operators.
        OR and NOT are supported for simple (non-nested) conditions only.

        :param query: The Supabase query builder.
        :param filters: Haystack filter dict.
        :returns: The query with filters applied.
        :raises FilterError: For unsupported operators, invalid value types, or malformed filters.
        """
        if not filters:
            return query

        if "field" in filters:
            return SupabaseGroongaDocumentStore._apply_condition(query, filters)

        if "operator" not in filters:
            msg = "Logical filter must include an 'operator' key ('AND', 'OR', 'NOT')."
            raise FilterError(msg)

        if "conditions" not in filters:
            msg = "Logical filter must include a 'conditions' key."
            raise FilterError(msg)

        op = filters["operator"]
        conditions = filters["conditions"]

        if op == "AND":
            for cond in conditions:
                query = SupabaseGroongaDocumentStore._apply_filters(query, cond)
            return query

        if op == "OR":
            pg_op_map = {"==": "eq", "!=": "neq", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}
            parts = []
            for cond in conditions:
                if "field" not in cond:
                    msg = "Nested logical operators inside OR are not supported."
                    raise FilterError(msg)
                cond_field = cond.get("field", "")
                cond_op = cond.get("operator", "")
                cond_value = cond.get("value")
                if cond_op not in pg_op_map:
                    msg = f"Operator '{cond_op}' inside OR filter is not supported."
                    raise FilterError(msg)
                # Use text accessor (->>): PostgREST OR strings don't support JSONB (->) expressions.
                col = f"meta->>{cond_field[len('meta.') :]}" if cond_field.startswith("meta.") else cond_field
                norm = SupabaseGroongaDocumentStore._normalize_value(cond_value)
                parts.append(f"{col}.{pg_op_map[cond_op]}.{norm}")
            return query.or_(",".join(parts))

        if op == "NOT":
            # NOT(A AND B) = NOT_A OR NOT_B, with null-inclusive semantics.
            # Use text accessor: PostgREST OR strings don't support JSONB (->) expressions.
            neg_map = {"==": "neq", "!=": "eq", ">": "lte", ">=": "lt", "<": "gte", "<=": "gt"}
            parts = []
            for cond in conditions:
                if "field" not in cond:
                    msg = "Nested logical operators inside NOT are not supported."
                    raise FilterError(msg)
                cond_field = cond.get("field", "")
                cond_op = cond.get("operator", "")
                cond_value = cond.get("value")
                if cond_op not in neg_map:
                    msg = f"Operator '{cond_op}' inside NOT filter is not supported."
                    raise FilterError(msg)
                col = f"meta->>{cond_field[len('meta.') :]}" if cond_field.startswith("meta.") else cond_field
                norm = SupabaseGroongaDocumentStore._normalize_value(cond_value)
                parts.append(f"{col}.{neg_map[cond_op]}.{norm}")
                if cond_op == "==" and cond_field.startswith("meta."):
                    # NOT(field==value) also covers docs where the field is absent (SQL NULL semantics)
                    parts.append(f"{col}.is.null")
            return query.or_(",".join(parts))

        msg = f"Filter operator '{op}' is not supported. Supported logical operators: AND, OR, NOT."
        raise FilterError(msg)

    @staticmethod
    def _apply_condition(query: Any, condition: dict[str, Any]) -> Any:
        field: str = condition.get("field", "")

        if "operator" not in condition:
            msg = "Comparison filter must include an 'operator' key."
            raise FilterError(msg)

        if "value" not in condition:
            msg = "Comparison filter must include a 'value' key."
            raise FilterError(msg)

        op: str = condition["operator"]
        value = condition["value"]

        col = SupabaseGroongaDocumentStore._meta_col(field, value)
        norm = SupabaseGroongaDocumentStore._normalize_value(value)

        if op == "==":
            return query.is_(col, "null") if norm is None else query.eq(col, norm)

        if op == "!=":
            if norm is None:
                return query.not_.is_(col, "null")
            if field.startswith("meta."):
                # SQL: NULL != value returns NULL (not TRUE), so include docs where the field is absent.
                key = field[len("meta.") :]
                return query.or_(f"{col}.neq.{norm},meta->>{key}.is.null")
            return query.neq(col, norm)

        if op in (">", ">=", "<", "<="):
            if isinstance(value, list):
                msg = f"Filter operator '{op}' does not support list values."
                raise FilterError(msg)
            if value is None:
                return query.eq("id", "")
            if isinstance(value, str):
                try:
                    _datetime.fromisoformat(value)
                except ValueError as err:
                    msg = f"Filter operator '{op}' does not support string values. Use a numeric or ISO date value."
                    raise FilterError(msg) from err
            if op == ">":
                return query.gt(col, norm)
            if op == ">=":
                return query.gte(col, norm)
            if op == "<":
                return query.lt(col, norm)
            return query.lte(col, norm)

        if op == "in":
            if not isinstance(value, list):
                msg = "Filter operator 'in' requires a list value."
                raise FilterError(msg)
            return query.in_(col, value)

        if op == "not in":
            if not isinstance(value, list):
                msg = "Filter operator 'not in' requires a list value."
                raise FilterError(msg)
            if field.startswith("meta."):
                # SQL: NULL NOT IN (...) returns NULL, so include docs where the field is absent.
                key = field[len("meta.") :]
                non_none = [v for v in value if v is not None]
                vals = ",".join(str(v) for v in non_none)
                return query.or_(f"{col}.not.in.({vals}),meta->>{key}.is.null")
            return query.not_.in_(col, value)

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
        if not isinstance(documents, list):
            msg = f"write_documents() expects a list of Document objects, got {type(documents).__name__}"
            raise ValueError(msg)
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"write_documents() expects Document objects, got {type(doc).__name__}"
                raise ValueError(msg)

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

    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ) -> int:
        """
        Async version of write_documents.

        :param documents: List of Haystack Document objects to write.
        :param policy: How to handle duplicate documents. Defaults to DuplicatePolicy.FAIL.
        :returns: Number of documents written.
        """
        if not isinstance(documents, list):
            msg = f"write_documents_async() expects a list of Document objects, got {type(documents).__name__}"
            raise ValueError(msg)
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"write_documents_async() expects Document objects, got {type(doc).__name__}"
                raise ValueError(msg)

        if not documents:
            return 0

        await self._initialize_async_client()

        written = 0
        for doc in documents:
            row = {
                "id": doc.id,
                "content": doc.content or "",
                "meta": doc.meta or {},
                "score": None,
            }
            if policy == DuplicatePolicy.OVERWRITE:
                await self._async_client.table(self.table_name).upsert(row).execute()  # type: ignore[union-attr]
                written += 1
            elif policy == DuplicatePolicy.SKIP:
                existing = await self._async_client.table(self.table_name).select("id").eq("id", doc.id).execute()  # type: ignore[union-attr]
                if not existing.data:
                    await self._async_client.table(self.table_name).insert(row).execute()  # type: ignore[union-attr]
                    written += 1
            elif policy == DuplicatePolicy.FAIL:
                existing = await self._async_client.table(self.table_name).select("id").eq("id", doc.id).execute()  # type: ignore[union-attr]
                if existing.data:
                    msg = f"Document with id {doc.id!r} already exists."
                    raise DuplicateDocumentError(msg)
                await self._async_client.table(self.table_name).insert(row).execute()  # type: ignore[union-attr]
                written += 1
            else:
                await self._async_client.table(self.table_name).insert(row).execute()  # type: ignore[union-attr]
                written += 1

        return written

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes documents matching the given filters.

        :param filters: Filters to select documents for deletion.
        :returns: Number of documents deleted.
        """
        docs = self.filter_documents(filters=filters)
        if not docs:
            return 0
        self.delete_documents([doc.id for doc in docs])
        return len(docs)

    async def delete_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Async version of delete_by_filter.

        :param filters: Filters to select documents for deletion.
        :returns: Number of documents deleted.
        """
        docs = await self.filter_documents_async(filters=filters)
        if not docs:
            return 0
        await self.delete_documents_async([doc.id for doc in docs])
        return len(docs)

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates the metadata of documents matching the given filters.

        Provided meta fields are merged into the existing document metadata.

        :param filters: Filters to select documents to update.
        :param meta: Metadata fields to set on matching documents.
        :returns: Number of documents updated.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)

        docs = self.filter_documents(filters=filters)
        if not docs:
            return 0

        for doc in docs:
            row = {
                "id": doc.id,
                "content": doc.content or "",
                "meta": {**doc.meta, **meta},
                "score": None,
            }
            self._client.table(self.table_name).upsert(row).execute()

        return len(docs)

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Async version of update_by_filter.

        :param filters: Filters to select documents to update.
        :param meta: Metadata fields to set on matching documents.
        :returns: Number of documents updated.
        """
        docs = await self.filter_documents_async(filters=filters)
        if not docs:
            return 0

        for doc in docs:
            row = {
                "id": doc.id,
                "content": doc.content or "",
                "meta": {**doc.meta, **meta},
                "score": None,
            }
            await self._async_client.table(self.table_name).upsert(row).execute()  # type: ignore[union-attr]

        return len(docs)

    def delete_all_documents(self) -> None:
        """
        Deletes all documents from the store.
        """
        if self._client is None:
            msg = "Call warm_up() before using the document store."
            raise RuntimeError(msg)
        self._client.table(self.table_name).delete().neq("id", "").execute()

    async def delete_all_documents_async(self) -> None:
        """
        Async version of delete_all_documents.
        """
        await self._initialize_async_client()
        await self._async_client.table(self.table_name).delete().neq("id", "").execute()  # type: ignore[union-attr]

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

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Async version of delete_documents.

        :param document_ids: List of document IDs to delete.
        """
        if not document_ids:
            return
        await self._initialize_async_client()
        await self._async_client.table(self.table_name).delete().in_("id", document_ids).execute()  # type: ignore[union-attr]

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
            {"query_text": query, "table_name": self.table_name, "top_k": top_k},
        ).execute()

        data = result.data if isinstance(result.data, list) else []
        documents = [self._to_haystack_document(row) for row in data if isinstance(row, dict)]

        # Apply filters post-retrieval if provided
        if filters:
            documents = SupabaseGroongaDocumentStore._filter_documents_in_memory(documents, filters)

        return documents

    async def _groonga_retrieval_async(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Async version of _groonga_retrieval.

        :param query: The text query to search for.
        :param top_k: Maximum number of results to return.
        :param filters: Optional filters to apply after retrieval.
        :returns: List of matching Document objects ranked by relevance.
        """
        await self._initialize_async_client()
        result = await self._async_client.rpc(  # type: ignore[union-attr]
            "groonga_search",
            {"query_text": query, "table_name": self.table_name, "top_k": top_k},
        ).execute()

        data = result.data if isinstance(result.data, list) else []
        documents = [self._to_haystack_document(row) for row in data if isinstance(row, dict)]

        if filters:
            documents = SupabaseGroongaDocumentStore._filter_documents_in_memory(documents, filters)

        return documents

    @staticmethod
    def _filter_documents_in_memory(documents: list[Document], filters: dict[str, Any]) -> list[Document]:
        """
        Filters a list of documents in memory using Haystack's standard filter syntax.

        Supports the full filter syntax: comparison operators (``==``, ``!=``, ``>``,
        ``>=``, ``<``, ``<=``, ``in``, ``not in``) and logical operators (``AND``,
        ``OR``, ``NOT``) including arbitrary nesting.

        :param documents: List of documents to filter.
        :param filters: Haystack filter dict (comparison or logical).
        :returns: Filtered list of documents.
        """
        return [doc for doc in documents if SupabaseGroongaDocumentStore._match_document(doc, filters)]

    @staticmethod
    def _match_document(doc: Document, filters: dict[str, Any]) -> bool:
        """Returns True if *doc* satisfies the filter expression."""
        if not filters:
            return True

        if "field" in filters:
            return SupabaseGroongaDocumentStore._match_condition(doc, filters)

        op = filters.get("operator")
        conditions: list[dict[str, Any]] = filters.get("conditions", [])

        if op == "AND":
            return all(SupabaseGroongaDocumentStore._match_document(doc, c) for c in conditions)
        if op == "OR":
            return any(SupabaseGroongaDocumentStore._match_document(doc, c) for c in conditions)
        if op == "NOT":
            # De Morgan: NOT(A AND B) = NOT_A OR NOT_B — doc passes if it fails at least one condition.
            return not all(SupabaseGroongaDocumentStore._match_document(doc, c) for c in conditions)

        return True

    @staticmethod
    def _match_condition(doc: Document, condition: dict[str, Any]) -> bool:
        """Returns True if *doc* satisfies a single comparison condition."""
        field: str = condition.get("field", "")
        op: str = condition.get("operator", "==")
        value = condition.get("value")

        if field.startswith("meta."):
            doc_value = doc.meta.get(field[len("meta.") :])
        else:
            doc_value = getattr(doc, field, None)

        if op == "==":
            return doc_value == value  # type: ignore[no-any-return]
        if op == "!=":
            return doc_value != value  # type: ignore[no-any-return]
        if op == ">":
            return doc_value is not None and doc_value > value  # type: ignore[operator]
        if op == ">=":
            return doc_value is not None and doc_value >= value  # type: ignore[operator]
        if op == "<":
            return doc_value is not None and doc_value < value  # type: ignore[operator]
        if op == "<=":
            return doc_value is not None and doc_value <= value  # type: ignore[operator]
        if op == "in":
            return doc_value in value  # type: ignore[operator]
        if op == "not in":
            return doc_value not in value  # type: ignore[operator]
        return True

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
