import time

import pytest
from haystack.document_stores import DuplicatePolicy

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
    namespace = f"{request.node.name}-{int(time.time())}"
    dimension = 768

    store = PineconeDocumentStore(
        environment=environment,
        index=index,
        namespace=namespace,
        dimension=dimension,
    )

    # Override some methods to wait for the documents to be available
    original_write_documents = store.write_documents

    def write_documents_and_wait(documents, policy=DuplicatePolicy.NONE):
        written_docs = original_write_documents(documents, policy)
        time.sleep(SLEEP_TIME)
        return written_docs

    original_delete_documents = store.delete_documents

    def delete_documents_and_wait(filters):
        original_delete_documents(filters)
        time.sleep(SLEEP_TIME)

    store.write_documents = write_documents_and_wait
    store.delete_documents = delete_documents_and_wait

    yield store
    store._index.delete(delete_all=True, namespace=namespace)
