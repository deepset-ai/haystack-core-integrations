import os

import pytest
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.astra import AstraDocumentStore


@pytest.fixture
def document_store() -> AstraDocumentStore:
    """
    This is the most basic requirement for the child class: provide
    an instance of this document store so the base class can use it.
    """
    astra_id = os.getenv("ASTRA_DB_ID", "")
    astra_region = os.getenv("ASTRA_DB_REGION", "us-east-2")

    astra_application_token = os.getenv(
        "ASTRA_DB_APPLICATION_TOKEN",
        "",
    )

    keyspace_name = "astra_haystack_test"
    collection_name = "haystack_integration"

    astra_store = AstraDocumentStore(
        astra_id=astra_id,
        astra_region=astra_region,
        astra_application_token=astra_application_token,
        astra_keyspace=keyspace_name,
        astra_collection=collection_name,
        duplicates_policy=DuplicatePolicy.OVERWRITE,
        embedding_dim=768,
    )
    return astra_store
