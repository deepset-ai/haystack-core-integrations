import os
import time
import uuid

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore

# This is the approximate time in seconds it takes for the documents to be available in Azure Search index
SLEEP_TIME_IN_SECONDS = 10
MAX_WAIT_TIME_FOR_INDEX_DELETION = 10


@pytest.fixture()
def sleep_time():
    return SLEEP_TIME_IN_SECONDS


@pytest.fixture
def document_store(request):
    """
    This is the most basic requirement for the child class: provide
    an instance of this document store so the base class can use it.
    """
    index_name = f"haystack_test_{uuid.uuid4().hex}"
    metadata_fields = getattr(request, "param", {}).get("metadata_fields", None)
    include_search_metadata = getattr(request, "param", {}).get("include_search_metadata", False)

    azure_endpoint = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
    api_key = os.environ["AZURE_AI_SEARCH_API_KEY"]

    client = SearchIndexClient(azure_endpoint, AzureKeyCredential(api_key))
    if index_name in client.list_index_names():
        client.delete_index(index_name)

    store = AzureAISearchDocumentStore(
        index_name=index_name,
        create_index=True,
        embedding_dimension=768,
        metadata_fields=metadata_fields,
        include_search_metadata=include_search_metadata,
    )

    # Override some methods to wait for the documents to be available
    original_write_documents = store.write_documents
    original_delete_documents = store.delete_documents

    def write_documents_and_wait(documents, policy=DuplicatePolicy.OVERWRITE):
        written_docs = original_write_documents(documents, policy)
        time.sleep(SLEEP_TIME_IN_SECONDS)
        return written_docs

    def delete_documents_and_wait(filters):
        original_delete_documents(filters)
        time.sleep(SLEEP_TIME_IN_SECONDS)

    # Helper function to wait for the index to be deleted, needed to cover latency
    def wait_for_index_deletion(client, index_name):
        start_time = time.time()
        while time.time() - start_time < MAX_WAIT_TIME_FOR_INDEX_DELETION:
            if index_name not in client.list_index_names():
                return True
            time.sleep(1)
        return False

    store.write_documents = write_documents_and_wait
    store.delete_documents = delete_documents_and_wait

    yield store
    try:
        client.delete_index(index_name)
        if not wait_for_index_deletion(client, index_name):
            print(f"Index {index_name} was not properly deleted.")
    except ResourceNotFoundError:
        print(f"Index {index_name} was already deleted or not found.")
    except Exception as e:
        print(f"Unexpected error when deleting index {index_name}: {e}")
        raise


@pytest.fixture(scope="session", autouse=True)
def cleanup_indexes():
    """
    Fixture to clean up all remaining indexes at the end of the test session.
    Only runs if Azure credentials are available.
    Automatically runs after all tests.
    """
    azure_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")

    # Skip cleanup if credentials aren't available
    if not azure_endpoint or not api_key:
        yield
        return

    client = SearchIndexClient(azure_endpoint, AzureKeyCredential(api_key))

    yield  # Allow tests to run before performing cleanup

    # Cleanup: Delete all remaining indexes
    print("Starting session-level cleanup of all Azure Search indexes.")
    existing_indexes = client.list_index_names()
    for index in existing_indexes:
        try:
            client.delete_index(index)
        except Exception as e:
            print(f"Failed to delete index during clean up {index}: {e}")
