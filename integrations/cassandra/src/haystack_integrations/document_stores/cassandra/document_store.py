"""
Cassandra Document Store for Haystack.

A document store implementation using Apache Cassandra / Astra DB as backend.
"""

import contextlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from haystack import Document
from haystack.document_stores.types.policy import DuplicatePolicy
from haystack.utils import Secret

try:
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.cluster import Cluster, Session
except ImportError as err:
    msg = (
        "The 'cassandra-driver' package is required to use "
        "CassandraDocumentStore. Install it with: pip install cassandra-driver"
    )
    raise ImportError(msg) from err

logger = logging.getLogger(__name__)


@dataclass
class _CassandraConfig:
    """Configuration for Cassandra connection."""

    contact_points: list[str]
    port: int
    username: str | None
    password: str | None
    keyspace: str
    table: str
    ttl_seconds: int | None


class CassandraDocumentStore:
    """
    Cassandra Document Store for Haystack.

    Stores documents in a Cassandra table with support for:
    - Basic CRUD operations
    - Filtering by metadata
    - Vector search (if using Astra DB with vector search)

    Example:
        ```python
        from haystack_integrations.document_stores.cassandra import CassandraDocumentStore

        document_store = CassandraDocumentStore(
            contact_points=["127.0.0.1"],
            keyspace="haystack",
            table="documents"
        )
        document_store.write_documents([Document(content="Hello", meta={"key": "value"})])
        ```
    """

    def __init__(  # noqa: PLR0913
        self,
        contact_points: list[str] | None = None,
        port: int = 9042,
        username: str | Secret | None = None,
        password: str | Secret | None = None,
        keyspace: str = "haystack",
        table: str = "documents",
        ttl_seconds: int | None = None,
        session: Session | None = None,
    ):
        """
        Initialize CassandraDocumentStore.

        :param contact_points: List of Cassandra contact points (hostnames/IPs).
        :param port: Cassandra port (default: 9042).
        :param username: Username for authentication (can be a Secret).
        :param password: Password for authentication (can be a Secret).
        :param keyspace: Cassandra keyspace name (default: "haystack").
        :param table: Table name for documents (default: "documents").
        :param ttl_seconds: Time-to-live for documents in seconds (default: None, no expiration).
        :param session: Existing Cassandra session to use (optional).
        """
        self._contact_points = contact_points or ["127.0.0.1"]
        self._port = port

        # Handle Secret objects
        self._username = username.resolve_value() if isinstance(username, Secret) else username
        self._password = password.resolve_value() if isinstance(password, Secret) else password

        self._keyspace = keyspace
        self._table = table
        self._ttl_seconds = ttl_seconds
        self._session = session
        self._owns_session = session is None

        # Initialize connection if not provided
        if self._owns_session:
            self._connect()

        # Ensure schema exists
        self._ensure_schema()

    def _connect(self) -> None:
        """Establish Cassandra connection."""
        auth_provider = None
        if self._username and self._password:
            auth_provider = PlainTextAuthProvider(
                username=self._username, password=self._password
            )

        cluster = Cluster(
            self._contact_points,
            port=self._port,
            auth_provider=auth_provider,
        )
        self._session = cluster.connect()
        logger.info(f"Connected to Cassandra at {self._contact_points}:{self._port}")

    def _ensure_schema(self) -> None:
        """Create keyspace and table if they don't exist."""
        # Create keyspace
        self._session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {self._keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
            """
        )
        self._session.set_keyspace(self._keyspace)

        # Create table with metadata as JSON blob
        # Using id as primary key, content and metadata as columns
        ttl_clause = f"AND default_time_to_live = {self._ttl_seconds}" if self._ttl_seconds else ""
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id text PRIMARY KEY,
                content text,
                metadata text,
                embedding blob,
                score float
            ) {ttl_clause}
            """
        )

        # Create indexes for common filter fields
        self._session.execute(
            f"CREATE INDEX IF NOT EXISTS ON {self._keyspace}.{self._table} (metadata)"
        )

        # Prepare statements for performance
        self._insert_stmt = self._session.prepare(
            f"INSERT INTO {self._table} (id, content, metadata, embedding, score) VALUES (?, ?, ?, ?, ?)"
        )
        self._select_stmt = self._session.prepare(
            f"SELECT id, content, metadata, embedding, score FROM {self._table} WHERE id = ?"
        )
        self._select_all_stmt = self._session.prepare(
            f"SELECT id, content, metadata, embedding, score FROM {self._table}"
        )
        self._delete_stmt = self._session.prepare(
            f"DELETE FROM {self._table} WHERE id = ?"
        )
        self._count_stmt = self._session.prepare(
            f"SELECT COUNT(*) FROM {self._table}"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this document store to a dictionary.

        :returns: Dictionary with configuration parameters.
        """
        return {
            "type": self.__class__.__name__,
            "init_parameters": {
                "contact_points": self._contact_points,
                "port": self._port,
                "username": self._username,
                "password": self._password,
                "keyspace": self._keyspace,
                "table": self._table,
                "ttl_seconds": self._ttl_seconds,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CassandraDocumentStore":
        """
        Deserialize document store from a dictionary.

        :param data: Dictionary with configuration parameters.
        :returns: CassandraDocumentStore instance.
        """
        init_params = data.get("init_parameters", {})
        return cls(**init_params)

    def count_documents(self) -> int:
        """
        Return the number of documents in the store.

        :returns: Number of documents.
        """
        result = self._session.execute(self._count_stmt)
        row = result.one()
        return row[0] if row else 0

    def filter_documents(
        self, filters: dict[str, Any] | None = None
    ) -> list[Document]:
        """
        Return documents matching the given filters.

        :param filters: Filters to apply (metadata key-value pairs).
        :returns: List of matching documents.
        """
        # For simple metadata filtering, we fetch all and filter in memory
        # In production, this could be optimized with CQL WHERE clauses
        result = self._session.execute(self._select_all_stmt)
        documents = []

        for row in result:
            doc = self._row_to_document(row)
            if doc and self._matches_filters(doc, filters):
                documents.append(doc)

        return documents

    def write_documents(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Write documents to the store.

        :param documents: List of documents to write.
        :param policy: How to handle duplicates (FAIL, OVERWRITE, SKIP).
        :returns: Number of documents written.
        """
        if not documents:
            return 0

        written = 0
        for doc in documents:
            doc_id = doc.id or str(uuid.uuid4())

            # Check for duplicates
            existing = self._session.execute(self._select_stmt, (doc_id,)).one()
            if existing:
                if policy == DuplicatePolicy.FAIL:
                    msg = f"Document with id '{doc_id}' already exists."
                    raise ValueError(msg)
                if policy == DuplicatePolicy.SKIP:
                    continue
                # NONE or OVERWRITE: proceed to update/overwrite

            # Prepare data
            content = doc.content or ""
            metadata_json = json.dumps(doc.meta) if doc.meta else "{}"
            embedding_blob = doc.embedding or None
            score = doc.score if doc.score is not None else 0.0

            # Execute insert
            self._session.execute(
                self._insert_stmt,
                (doc_id, content, metadata_json, embedding_blob, score),
            )
            written += 1

        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by their IDs.

        :param document_ids: List of document IDs to delete.
        """
        for doc_id in document_ids:
            self._session.execute(self._delete_stmt, (doc_id,))

    def delete_all_documents(self) -> None:
        """Delete all documents from the store."""
        self._session.execute(f"TRUNCATE {self._keyspace}.{self._table}")

    def _row_to_document(self, row: Any) -> Document | None:
        """Convert a Cassandra row to a Haystack Document."""
        if row is None:
            return None

        metadata = {}
        if row.metadata:
            with contextlib.suppress(json.JSONDecodeError):
                metadata = json.loads(row.metadata)

        return Document(
            id=row.id,
            content=row.content,
            meta=metadata,
            embedding=row.embedding,
            score=row.score,
        )

    def _matches_filters(
        self, document: Document, filters: dict[str, Any] | None
    ) -> bool:
        """Check if a document matches the given filters."""
        if not filters:
            return True

        for key, value in filters.items():
            if key not in document.meta:
                return False
            if document.meta[key] != value:
                return False
        return True

    def close(self) -> None:
        """Close the Cassandra connection."""
        if self._owns_session and self._session is not None:
            cluster = self._session.cluster
            self._session.shutdown()
            cluster.shutdown()
            self._session = None
            logger.info("Cassandra connection closed")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
