import os
import time
import uuid

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from haystack import logging
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore

# This is the approximate time in seconds it takes for the documents to be available in Azure Search index
SLEEP_TIME_IN_SECONDS = 10
MAX_WAIT_TIME_FOR_INDEX_DELETION = 5


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

    azure_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    api_key = os.environ["AZURE_SEARCH_API_KEY"]

    client = SearchIndexClient(azure_endpoint, AzureKeyCredential(api_key))
    if index_name in client.list_index_names():
        client.delete_index(index_name)

    store = AzureAISearchDocumentStore(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        index_name=index_name,
        create_index=True,
        embedding_dimension=768,
        metadata_fields=metadata_fields,
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
            logging.error(f"Index {index_name} was not properly deleted.")
    except ResourceNotFoundError:
        logging.info(f"Index {index_name} was already deleted or not found.")
    except Exception as e:
        logging.error(f"Unexpected error when deleting index {index_name}: {e}")
