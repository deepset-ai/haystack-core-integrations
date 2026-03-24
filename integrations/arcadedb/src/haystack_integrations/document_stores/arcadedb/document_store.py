# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""ArcadeDB DocumentStore for Haystack 2.x — document storage + vector search via HTTP/JSON API."""

import logging
from http import HTTPStatus
from typing import Any, ClassVar

import requests
from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.utils import Secret

from haystack_integrations.document_stores.arcadedb.converters import (
    _from_arcadedb_to_haystack,
    _from_haystack_to_arcadedb,
)
from haystack_integrations.document_stores.arcadedb.filters import _convert_filters

logger = logging.getLogger(__name__)


class ArcadeDBDocumentStore:
    """
    An ArcadeDB-backed DocumentStore for Haystack 2.x.

    Uses ArcadeDB's HTTP/JSON API for all operations — no special drivers required.
    Supports HNSW vector search (LSM_VECTOR) and SQL metadata filtering.

    Usage example:

    ```python
    from haystack.dataclasses.document import Document
    from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

    document_store = ArcadeDBDocumentStore(
        url="http://localhost:2480",
        database="haystack",
        embedding_dimension=768,
    )
    document_store.write_documents([
        Document(content="This is first", embedding=[0.0]*5),
        Document(content="This is second", embedding=[0.1, 0.2, 0.3, 0.4, 0.5])
    ])
    ```
    """

    # Map user-facing similarity names to ArcadeDB LSM_VECTOR metric keywords
    _SIMILARITY_MAP: ClassVar[dict[str, str]] = {
        "cosine": "COSINE",
        "euclidean": "EUCLIDEAN",
        "dot": "DOT_PRODUCT",
    }

    # Limit for projection documents
    SCHEMA_SAMPLING_LIMIT: ClassVar[int] = 1000

    def __init__(
        self,
        *,
        url: str = "http://localhost:2480",
        database: str = "haystack",
        username: Secret = Secret.from_env_var("ARCADEDB_USERNAME", strict=False),
        password: Secret = Secret.from_env_var("ARCADEDB_PASSWORD", strict=False),
        type_name: str = "Document",
        embedding_dimension: int = 768,
        similarity_function: str = "cosine",
        recreate_type: bool = False,
        create_database: bool = True,
    ) -> None:
        """
        Create an ArcadeDBDocumentStore instance.

        :param url: ArcadeDB HTTP endpoint.
        :param database: Database name.
        :param username: HTTP Basic Auth username (default: ``ARCADEDB_USERNAME`` env var).
        :param password: HTTP Basic Auth password (default: ``ARCADEDB_PASSWORD`` env var).
        :param type_name: Vertex type name for documents.
        :param embedding_dimension: Vector dimension for the HNSW index.
        :param similarity_function: Distance metric — ``"cosine"``, ``"euclidean"``, or ``"dot"``.
        :param recreate_type: If ``True``, drop and recreate the type on initialization.
        :param create_database: If ``True``, create the database if it doesn't exist.
        """
        self._url = url.rstrip("/")
        self._database = database
        self._username = username
        self._password = password
        self._type_name = type_name
        self._embedding_dimension = embedding_dimension
        self._similarity_function = similarity_function
        self._recreate_type = recreate_type
        self._create_database = create_database

        self._session = requests.Session()
        self._initialized = False

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the DocumentStore to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            url=self._url,
            database=self._database,
            username=self._username.to_dict() if self._username else None,
            password=self._password.to_dict() if self._password else None,
            type_name=self._type_name,
            embedding_dimension=self._embedding_dimension,
            similarity_function=self._similarity_function,
            recreate_type=self._recreate_type,
            create_database=self._create_database,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArcadeDBDocumentStore":
        """
        Deserializes the DocumentStore from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized DocumentStore.
        """
        init_params = data.get("init_parameters", {})
        for key in ("username", "password"):
            if init_params.get(key) is not None:
                init_params[key] = Secret.from_dict(init_params[key])
        return default_from_dict(cls, data)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _auth(self) -> tuple[str, str] | None:
        user = self._username.resolve_value() if self._username else None
        pwd = self._password.resolve_value() if self._password else None
        if user and pwd:
            return (user, pwd)
        return None

    def _command(self, sql: str, *, positional_params: list[Any] | None = None) -> list[dict[str, Any]]:
        """Execute an SQL command via the ArcadeDB HTTP API and return result rows."""
        url = f"{self._url}/api/v1/command/{self._database}"
        payload: dict[str, Any] = {"language": "sql", "command": sql}
        if positional_params:
            payload["params"] = positional_params

        resp = self._session.post(url, json=payload, auth=self._auth())
        if resp.status_code >= HTTPStatus.BAD_REQUEST:
            msg = f"ArcadeDB command failed ({resp.status_code}): {resp.text}"
            raise RuntimeError(msg)

        body = resp.json()
        return body.get("result", [])

    def _server_command(self, command: str) -> dict[str, Any]:
        """Execute a server-level command (e.g. CREATE DATABASE)."""
        url = f"{self._url}/api/v1/server"
        resp = self._session.post(url, json={"command": command}, auth=self._auth())
        if resp.status_code >= HTTPStatus.BAD_REQUEST:
            msg = f"ArcadeDB server command failed ({resp.status_code}): {resp.text}"
            raise RuntimeError(msg)
        return resp.json()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        # 1. Optionally create the database
        if self._create_database:
            try:
                self._server_command(f"CREATE DATABASE {self._database}")
                logger.info("Created database '%s'", self._database)
            except RuntimeError:
                logger.debug("Database '%s' already exists or cannot be created", self._database)

        # 2. Optionally drop existing type
        if self._recreate_type:
            try:
                self._command(f"DROP TYPE `{self._type_name}` IF EXISTS UNSAFE")
            except RuntimeError:
                pass

        # 3. Create vertex type + properties
        self._command(f"CREATE VERTEX TYPE `{self._type_name}` IF NOT EXISTS")
        self._command(f"CREATE PROPERTY `{self._type_name}`.id IF NOT EXISTS STRING")
        self._command(f"CREATE PROPERTY `{self._type_name}`.content IF NOT EXISTS STRING")
        self._command(f"CREATE PROPERTY `{self._type_name}`.embedding IF NOT EXISTS ARRAY_OF_FLOATS")
        self._command(f"CREATE PROPERTY `{self._type_name}`.meta IF NOT EXISTS MAP")

        # 4. Unique index on id
        try:
            self._command(f"CREATE INDEX ON `{self._type_name}` (id) UNIQUE")
        except RuntimeError:
            logger.debug("Unique index on id already exists")

        # 5. LSM_VECTOR index on embedding (HNSW-based, ACID-compliant)
        metric = self._SIMILARITY_MAP.get(self._similarity_function, "COSINE")
        try:
            self._command(
                f"CREATE INDEX IF NOT EXISTS ON `{self._type_name}` (embedding) LSM_VECTOR "
                f"METADATA {{ dimensions: {self._embedding_dimension}, similarity: '{metric}' }}"
            )
        except RuntimeError:
            logger.debug("Vector index on embedding already exists")

        self._initialized = True
        logger.info(
            "ArcadeDBDocumentStore initialized: database=%s, type=%s, dim=%d, metric=%s",
            self._database,
            self._type_name,
            self._embedding_dimension,
            metric,
        )

    # ------------------------------------------------------------------
    # DocumentStore protocol
    # ------------------------------------------------------------------

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns:
            Number of documents in the document store.
        """
        self._ensure_initialized()
        rows = self._command(f"SELECT count(*) AS cnt FROM `{self._type_name}`")
        if rows:
            return int(rows[0].get("cnt", 0))
        return 0

    @staticmethod
    def _extract_distinct_values(rows: list[dict[str, Any]]) -> set[str]:
        """
        Extracts and flattens unique non-None strings from 'val' column result rows.
        :param rows: Raw result rows from ``_command``.
        :returns: A set of unique string values.
        """
        result: set[str] = set()
        for row in rows:
            val = row.get("val")
            if isinstance(val, list):
                result.update(str(item) for item in val if item is not None)
            elif val is not None:
                result.add(str(val))
        return result

    def _get_metadata_projection_documents(self) -> list[dict[str, Any]]:
        """
        Private helper to fetch sample documents for schema inference.
        Note: Does not `_ensure_initialized()`. To avoid redundant
        initialization checks during internal calls, the caller is responsible for
        ensuring the document store is initialized before invoking this method.
        """
        sql = f"SELECT content, meta FROM `{self._type_name}` LIMIT {self.SCHEMA_SAMPLING_LIMIT}"
        return self._command(sql)

    @staticmethod
    def _infer_metadata_field_type(values: list[Any]) -> str:
        """
        Infers the metadata field type from a list of sampled values.
        :param values: A list of raw Python values sampled from the field.
        :returns: A type string — one of ``"boolean"``, ``"double"``, ``"long"``, or ``"keyword"``.
            Returns ``"keyword"`` if values are empty or of mixed types.
        """
        inferred_types = set()
        for value in values:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, bool):
                        inferred_types.add("boolean")
                    elif isinstance(item, float):
                        inferred_types.add("double")
                    elif isinstance(item, int):
                        inferred_types.add("long")
                    elif isinstance(item, str):
                        inferred_types.add("keyword")
            elif isinstance(value, bool):
                inferred_types.add("boolean")
            elif isinstance(value, float):
                inferred_types.add("double")
            elif isinstance(value, int):
                inferred_types.add("long")
            elif isinstance(value, str):
                inferred_types.add("keyword")

        if not inferred_types:
            return "keyword"

        if len(inferred_types) > 1:
            logger.warning("Field has mixed metadata types %s. Defaulting to 'keyword'.", inferred_types)
            return "keyword"

        return next(iter(inferred_types))

    def filter_documents(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Return documents matching the given filters.

        :param filters: Haystack filter dictionary.
        :returns: List of matching documents.
        """
        self._ensure_initialized()
        try:
            where = _convert_filters(filters)
        except ValueError as e:
            raise FilterError(str(e)) from e
        sql = f"SELECT * FROM `{self._type_name}`"
        if where:
            sql += f" WHERE {where}"
        rows = self._command(sql)
        return _from_arcadedb_to_haystack(rows)

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Write documents to the store.

        :param documents: List of Haystack Documents to write.
        :param policy: How to handle duplicate document IDs.
        :returns: Number of documents written.
        """
        self._ensure_initialized()
        msg = "documents must be a list of Document objects"
        if not isinstance(documents, list):
            raise ValueError(msg)
        for doc in documents:
            if not isinstance(doc, Document):
                raise ValueError(msg)
        if not documents:
            return 0

        records = _from_haystack_to_arcadedb(documents)
        written = 0

        for record in records:
            emb = record["embedding"]
            if emb is None or not isinstance(emb, list) or len(emb) != self._embedding_dimension:
                emb = [0.0] * self._embedding_dimension
            embedding_str = str(emb)
            meta_str = _map_literal(record["meta"]) if record["meta"] else "{}"

            if policy == DuplicatePolicy.OVERWRITE:
                sql = (
                    f"UPDATE `{self._type_name}` SET "
                    f"content = {_sql_str(record['content'])}, "
                    f"embedding = {embedding_str}, "
                    f"meta = {meta_str} "
                    f"WHERE id = {_sql_str(record['id'])}"
                )
                result = self._command(sql)
                updated = int(result[0].get("count", 0)) if result else 0
                if updated == 0:
                    self._insert_record(record, embedding_str, meta_str)
                written += 1

            elif policy == DuplicatePolicy.SKIP:
                existing = self._command(f"SELECT id FROM `{self._type_name}` WHERE id = {_sql_str(record['id'])}")
                if existing:
                    continue
                self._insert_record(record, embedding_str, meta_str)
                written += 1

            else:
                # DuplicatePolicy.NONE — raise on duplicate
                existing = self._command(f"SELECT id FROM `{self._type_name}` WHERE id = {_sql_str(record['id'])}")
                if existing:
                    msg = f"Document with id '{record['id']}' already exists."
                    raise DuplicateDocumentError(msg)
                self._insert_record(record, embedding_str, meta_str)
                written += 1

        return written

    def _insert_record(self, record: dict[str, Any], embedding_str: str, meta_str: str) -> None:
        sql = (
            f"INSERT INTO `{self._type_name}` SET "
            f"id = {_sql_str(record['id'])}, "
            f"content = {_sql_str(record['content'])}, "
            f"embedding = {embedding_str}, "
            f"meta = {meta_str}"
        )
        self._command(sql)

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by their IDs.

        :param document_ids: List of document IDs to delete.
        """
        self._ensure_initialized()
        if not document_ids:
            return
        ids_str = ", ".join(_sql_str(did) for did in document_ids)
        self._command(f"DELETE FROM `{self._type_name}` WHERE id IN [{ids_str}]")

    def delete_all_documents(self) -> None:
        """
        Deletes all documents in the document store.
        """
        self._ensure_initialized()
        self._command(f"TRUNCATE TYPE `{self._type_name}`")

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        self._ensure_initialized()
        try:
            where = _convert_filters(filters)
        except ValueError as e:
            raise FilterError(str(e)) from e

        if not where:
            msg = "delete_by_filter requires a non-empty filter. Use delete_all_documents() to delete all documents."
            raise FilterError(msg)

        count_result = self._command(f"DELETE FROM `{self._type_name}` WHERE {where}")

        return count_result[0]["count"]

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
        :returns: The number of documents updated.
        """
        self._ensure_initialized()
        try:
            where = _convert_filters(filters)
        except ValueError as e:
            raise FilterError(str(e)) from e

        if not where:
            msg = "update_by_filter requires a non-empty filter."
            raise FilterError(msg)

        sql_set = ",".join(f"meta[{_sql_str(key)}] = {_map_literal_base(value)}" for key, value in meta.items())
        sql = f"UPDATE `{self._type_name}` SET {sql_set} WHERE {where}"
        count_result = self._command(sql)

        return count_result[0]["count"]

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Counts the number of documents matching the provided filter
        :param filters: The filters to apply to the documents
        :returns: The number of documents that match the filter
        """
        self._ensure_initialized()
        try:
            where = _convert_filters(filters)
        except ValueError as e:
            raise FilterError(str(e)) from e

        sql = f"SELECT count(*) AS cnt FROM `{self._type_name}`"
        if where:
            sql += f" WHERE {where}"

        rows = self._command(sql)
        if rows:
            return int(rows[0].get("cnt", 0))
        return 0

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]:
        """
        Counts unique values for each metadata field in documents matching the provided filters.
        :param filters: The filters to apply to the document list.
        :param metadata_fields: Metadata fields for which to count unique values.
        :returns: A dictionary where keys are metadata field names and values are the
            counts of unique values for that field.
        """
        self._ensure_initialized()
        try:
            where = _convert_filters(filters)
        except ValueError as e:
            raise FilterError(str(e)) from e

        if not metadata_fields:
            return {}

        counts = {}
        for field in metadata_fields:  # Arcade doesn't support COUNT(DISTINCT..)
            field_name = field.removeprefix("meta.")
            sql = f"SELECT DISTINCT meta[{_sql_str(field_name)}] AS val FROM `{self._type_name}`"
            if where:
                sql += f" WHERE {where}"
            rows = self._command(sql)
            counts[field_name] = len(self._extract_distinct_values(rows))

        return counts

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns the metadata fields and their corresponding types based on sampled documents.
        :returns: A dictionary mapping field names to dictionaries with a `type` key.
        """
        self._ensure_initialized()
        documents = self._get_metadata_projection_documents()

        if not documents:
            return {}

        fields_info: dict[str, dict[str, str]] = {}

        if any(document.get("content") is not None for document in documents):
            fields_info["content"] = {"type": "text"}

        field_values: dict[str, list[Any]] = {}
        for document in documents:
            for field, value in document.get("meta", {}).items():
                field_values.setdefault(field, []).append(value)

        for field, values in field_values.items():
            fields_info[field] = {"type": self._infer_metadata_field_type(values)}

        return fields_info

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, Any]:
        """
        For a given metadata field, finds its min and max values.
        :param metadata_field: The metadata field to inspect.
        :returns: A dictionary with `min` and `max` keys and their corresponding values.
        """
        self._ensure_initialized()

        field_name = metadata_field.removeprefix("meta.")
        field_ref = f"meta[{_sql_str(field_name)}]"
        sql = f"SELECT MIN({field_ref}) AS min_value, MAX({field_ref}) AS max_value FROM `{self._type_name}`"
        rows = self._command(sql)

        if not rows:
            return {"min": None, "max": None}

        return {"min": rows[0].get("min_value"), "max": rows[0].get("max_value")}

    def get_metadata_field_unique_values(
        self, metadata_field: str, search_term: str | None = None, from_: int = 0, size: int = 10
    ) -> tuple[list[str], int]:
        """
        Retrieves unique values for a field matching a search term or all possible values
        if no search term is given.
        :param metadata_field: The metadata field to inspect.
        :param search_term: Optional case-insensitive substring search term.
        :param from_: The starting index for pagination.
        :param size: The number of values to return.
        :returns: A tuple containing the paginated values and the total count.
        """
        self._ensure_initialized()

        metadata_field = metadata_field.removeprefix("meta.")
        field_ref = f"meta[{_sql_str(metadata_field)}]"
        where = ""

        if search_term:
            search_val = _sql_str(f"%{search_term}%")
            where = f" WHERE {field_ref} ILIKE {search_val}"

        sql = f"SELECT DISTINCT {field_ref} AS val FROM `{self._type_name}`{where}"
        rows = self._command(sql)

        all_values = sorted(self._extract_distinct_values(rows))
        total_count = len(all_values)
        return all_values[from_ : from_ + size], total_count

    # ------------------------------------------------------------------
    # Retrieval (called by Retriever components)
    # ------------------------------------------------------------------

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Retrieve documents by vector similarity using ArcadeDB's LSM_VECTOR index.

        :param query_embedding: The embedding vector to search with.
        :param filters: Optional metadata filters (applied as post-filter).
        :param top_k: Maximum number of documents to return.
        :returns: Documents ordered by descending similarity score.
        """
        self._ensure_initialized()
        embedding_str = str(query_embedding)

        # vectorNeighbors returns a single row with a "neighbors" list of {record, distance}
        sql = f"SELECT vectorNeighbors('{self._type_name}[embedding]', {embedding_str}, {top_k}) AS neighbors"
        rows = self._command(sql)
        if not rows or not rows[0].get("neighbors"):
            return []

        neighbors = rows[0]["neighbors"]
        try:
            where = _convert_filters(filters)
        except ValueError as e:
            raise FilterError(str(e)) from e

        documents = []
        for neighbor in neighbors:
            record = neighbor.get("record", {})
            distance = neighbor.get("distance", 0.0)
            score = 1.0 - distance

            doc = Document(
                id=record.get("id", ""),
                content=record.get("content"),
                meta=record.get("meta") or {},
                score=score,
            )
            documents.append(doc)

        # Post-filter by metadata if specified
        if where and filters:
            filtered_ids = {r["id"] for r in self._command(f"SELECT id FROM `{self._type_name}` WHERE {where}")}
            documents = [d for d in documents if d.id in filtered_ids]

        return documents


def _sql_str(value: str | None) -> str:
    """Escape and quote a string value for ArcadeDB SQL."""
    if value is None:
        return "NULL"
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _map_literal_base(value: object) -> str | float | list[Any]:
    """Map Python type to ArcadeDB type."""
    if isinstance(value, str):
        return _sql_str(value)
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, int | float):
        return value
    elif value is None:
        return "NULL"
    elif isinstance(value, list):
        return value
    else:
        return _sql_str(str(value))


def _map_literal(meta: dict[str, Any]) -> str:
    """Build an ArcadeDB MAP literal from a Python dict."""
    if not meta:
        return "{}"
    pairs = []
    for key, value in meta.items():
        pairs.append(f'"{key}": {_map_literal_base(value)}')
    return "{" + ", ".join(pairs) + "}"
