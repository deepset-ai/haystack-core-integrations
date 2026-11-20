"""
Utility functions for Cassandra Document Store.
"""

import contextlib
import json
import uuid
from typing import Any

from cassandra.cluster import Session


def ensure_keyspace(
    session: Session,
    keyspace: str,
    replication_factor: int = 1,
    strategy_class: str = "SimpleStrategy",
) -> None:
    """
    Ensure a keyspace exists.

    :param session: Cassandra session.
    :param keyspace: Keyspace name.
    :param replication_factor: Replication factor.
    :param strategy_class: Replication strategy class.
    """
    session.execute(
        f"""
        CREATE KEYSPACE IF NOT EXISTS {keyspace}
        WITH replication = {{'class': '{strategy_class}', 'replication_factor': {replication_factor}}}
        """
    )


def ensure_table(
    session: Session,
    keyspace: str,
    table: str,
    ttl_seconds: int | None = None,
) -> None:
    """
    Ensure a table exists for document storage.

    :param session: Cassandra session.
    :param keyspace: Keyspace name.
    :param table: Table name.
    :param ttl_seconds: Optional TTL in seconds.
    """
    session.set_keyspace(keyspace)

    ttl_clause = f"AND default_time_to_live = {ttl_seconds}" if ttl_seconds else ""
    session.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id text PRIMARY KEY,
            content text,
            metadata text,
            embedding blob,
            score float
        ) {ttl_clause}
        """
    )


def prepare_statements(session: Session, table: str) -> dict[str, Any]:
    """
    Prepare commonly used CQL statements.

    :param session: Cassandra session.
    :param table: Table name.
    :returns: Dictionary of prepared statements.
    """
    return {
        "insert": session.prepare(
            f"INSERT INTO {table} (id, content, metadata, embedding, score) VALUES (?, ?, ?, ?, ?)"
        ),
        "select": session.prepare(
            f"SELECT id, content, metadata, embedding, score FROM {table} WHERE id = ?"
        ),
        "select_all": session.prepare(
            f"SELECT id, content, metadata, embedding, score FROM {table}"
        ),
        "delete": session.prepare(
            f"DELETE FROM {table} WHERE id = ?"
        ),
        "count": session.prepare(
            f"SELECT COUNT(*) FROM {table}"
        ),
    }


def document_to_row(doc: Any, doc_id: str | None = None) -> dict[str, Any]:
    """
    Convert a Haystack Document to a dictionary for Cassandra insertion.

    :param doc: Haystack Document.
    :param doc_id: Optional document ID (generated if not provided).
    :returns: Dictionary with row data.
    """
    return {
        "id": doc_id or getattr(doc, "id", None) or str(uuid.uuid4()),
        "content": getattr(doc, "content", "") or "",
        "metadata": json.dumps(getattr(doc, "meta", {}) or {}),
        "embedding": getattr(doc, "embedding", None),
        "score": getattr(doc, "score", 0.0) or 0.0,
    }


def row_to_document(row: Any) -> dict[str, Any] | None:
    """
    Convert a Cassandra row to a dictionary for Haystack Document creation.

    :param row: Cassandra row.
    :returns: Dictionary with document data or None if row is None.
    """
    if row is None:
        return None

    metadata = {}
    if row.metadata:
        with contextlib.suppress(json.JSONDecodeError):
            metadata = json.loads(row.metadata)

    return {
        "id": row.id,
        "content": row.content,
        "meta": metadata,
        "embedding": row.embedding,
        "score": row.score,
    }
