"""
Cassandra Document Store integration for Haystack.

This package provides a Cassandra-backed document store implementation for Haystack,
allowing you to use Apache Cassandra or Astra DB as a document store backend.

Example:
    ```python
    from haystack_integrations.document_stores.cassandra import CassandraDocumentStore

    document_store = CassandraDocumentStore(
        contact_points=["127.0.0.1"],
        keyspace="haystack",
        table="documents"
    )
    document_store.write_documents([
        Document(content="Hello world", meta={"source": "test"})
    ])
    documents = document_store.filter_documents(filters={"source": "test"})
    ```
"""

from haystack_integrations.document_stores.cassandra.document_store import CassandraDocumentStore

__all__ = ["CassandraDocumentStore"]
