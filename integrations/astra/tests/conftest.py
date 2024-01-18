import os

import pytest
from haystack.document_stores import DuplicatePolicy

from astra_haystack.document_store import AstraDocumentStore


@pytest.fixture
def document_store() -> AstraDocumentStore:
    """
    This is the most basic requirement for the child class: provide
    an instance of this document store so the base class can use it.
    """
    astra_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT", "")

    astra_application_token = os.getenv(
        "ASTRA_DB_APPLICATION_TOKEN",
        "",
    )

    keyspace_name = "astra_haystack_test"
    collection_name = "haystack_integration"

    astra_store = AstraDocumentStore(
        astra_endpoint=astra_endpoint,
        astra_application_token=astra_application_token,
        astra_keyspace=keyspace_name,
        astra_collection=collection_name,
        duplicates_policy=DuplicatePolicy.OVERWRITE,
        embedding_dim=768,
    )
    return astra_store
