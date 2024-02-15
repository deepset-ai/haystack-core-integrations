import pytest
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.astra import AstraDocumentStore


@pytest.fixture
def document_store() -> AstraDocumentStore:
    """
    This is the most basic requirement for the child class: provide
    an instance of this document store so the base class can use it.

    Auth params will be read from env vars (default behavior)
    """
    return AstraDocumentStore(
        astra_keyspace="astra_haystack_test",
        astra_collection="haystack_integration",
        duplicates_policy=DuplicatePolicy.OVERWRITE,
        embedding_dim=768,
    )
