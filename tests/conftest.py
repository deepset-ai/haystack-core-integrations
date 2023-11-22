import pytest

from astra_store.document_store import AstraDocumentStore


@pytest.fixture
def astra_obj():
    astra_id = "af29f87d-72e6-47e4-a7c4-0c6279a51933"
    astra_region = "us-east1"
    astra_application_token = (
        "AstraCS:cLLKQjnuXpEOiIoLHKBaouwp:2832c7c700a3256b654942e955eaf65bce7e2f742de64e0396359b7594253f4d"
    )
    keyspace_name = "test"
    collection_name = "movies"

    astra_store = AstraDocumentStore(
        astra_id=astra_id,
        astra_region=astra_region,
        astra_application_token=astra_application_token,
        astra_keyspace=keyspace_name,
        astra_collection=collection_name,
        embedding_dim=1536,
    )
    return astra_store
