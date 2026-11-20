"""
Custom exceptions for Cassandra Document Store.
"""


class CassandraDocumentStoreError(Exception):
    """Base exception for Cassandra Document Store errors."""


class CassandraConnectionError(CassandraDocumentStoreError):
    """Raised when connection to Cassandra fails."""


class CassandraSchemaError(CassandraDocumentStoreError):
    """Raised when schema operations fail."""


class DocumentStoreError(CassandraDocumentStoreError):
    """Raised when document operations fail."""


class DuplicateDocumentError(DocumentStoreError):
    """Raised when attempting to write a duplicate document with FAIL policy."""


class DocumentNotFoundError(DocumentStoreError):
    """Raised when a document is not found."""
