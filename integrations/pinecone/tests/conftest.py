import time
from random import randint

import pytest

from pinecone_haystack.document_store import PineconeDocumentStore

# This is the approximate time it takes for the documents to be available
SLEEP_TIME = 20


@pytest.fixture()
def sleep_time():
    return SLEEP_TIME


@pytest.fixture
def document_store(request):
    """
    This is the most basic requirement for the child class: provide
    an instance of this document store so the base class can use it.
    """
    environment = "gcp-starter"
    index = "default"
    # Use a different namespace for each test so we can run them in parallel
    namespace = f"{request.node.name}-{randint(0, 1000)}"  # noqa: S311  Ruff complains about using random numbers for cryptographic purposes
    dimension = 10

    store = PineconeDocumentStore(
        environment=environment,
        index=index,
        namespace=namespace,
        dimension=dimension,
    )

    # Override the count_documents method to wait for the documents to be available
    original_count_documents = store.count_documents

    def wait_and_count_documents():
        time.sleep(SLEEP_TIME)
        return original_count_documents()

    store.count_documents = wait_and_count_documents

    yield store
    store._index.delete(delete_all=True, namespace=namespace)
