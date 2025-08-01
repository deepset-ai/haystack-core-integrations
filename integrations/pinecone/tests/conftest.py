import asyncio
import time
import uuid

import pytest
import pytest_asyncio
from haystack.document_stores.types import DuplicatePolicy
from pinecone.exceptions import NotFoundException

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# This is the approximate time in seconds it takes for the documents to be available
SLEEP_TIME_IN_SECONDS = 25


@pytest.fixture()
def sleep_time():
    return SLEEP_TIME_IN_SECONDS


@pytest.fixture
def document_store(request):
    """
    This is the most basic requirement for the child class: provide
    an instance of this document store so the base class can use it.
    """
    index = "default"
    # Use a different namespace for each test so we can run them in parallel
    namespace = f"{request.node.name}-{uuid.uuid4()}"
    dimension = 768

    store = PineconeDocumentStore(
        index=index,
        namespace=namespace,
        dimension=dimension,
    )

    # Override some methods to wait for the documents to be available
    original_write_documents = store.write_documents

    def write_documents_and_wait(documents, policy=DuplicatePolicy.NONE):
        written_docs = original_write_documents(documents, policy)
        time.sleep(SLEEP_TIME_IN_SECONDS)
        return written_docs

    original_delete_documents = store.delete_documents

    def delete_documents_and_wait(filters):
        original_delete_documents(filters)
        time.sleep(SLEEP_TIME_IN_SECONDS)

    store.write_documents = write_documents_and_wait
    store.delete_documents = delete_documents_and_wait

    yield store
    try:
        store._index.delete(delete_all=True, namespace=namespace)
    except NotFoundException:
        pass


@pytest_asyncio.fixture
async def document_store_async(request):
    """
    This is the most basic requirement for the child class: provide
    an instance of this document store so the base class can use it.
    """
    index = "default"
    # Use a different namespace for each test so we can run them in parallel
    namespace = f"{request.node.name}-{uuid.uuid4()}"
    dimension = 768

    store = PineconeDocumentStore(
        index=index,
        namespace=namespace,
        dimension=dimension,
    )

    # Override some methods to wait for the documents to be available
    original_write_documents = store.write_documents_async

    async def write_documents_and_wait_async(documents, policy=DuplicatePolicy.NONE):
        written_docs = await original_write_documents(documents, policy)
        await asyncio.sleep(SLEEP_TIME_IN_SECONDS)
        return written_docs

    original_delete_documents = store.delete_documents_async

    async def delete_documents_and_wait_async(filters):
        await original_delete_documents(filters)
        await asyncio.sleep(SLEEP_TIME_IN_SECONDS)

    store.write_documents_async = write_documents_and_wait_async
    store.delete_documents_async = delete_documents_and_wait_async

    yield store
    try:
        await store._async_index.delete(delete_all=True, namespace=namespace)
        await store.close_async()
    except NotFoundException:
        pass
