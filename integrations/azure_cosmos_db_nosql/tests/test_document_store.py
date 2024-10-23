import os
from typing import Any, Dict

import pytest
from azure.cosmos import PartitionKey
from haystack.dataclasses import Document
from haystack.testing.document_store import DocumentStoreBaseTests

from haystack_integrations.document_stores.nosql import \
    AzureCosmosDBNoSqlDocumentStore
from test_utils import get_vector_embedding_policy, get_vector_indexing_policy


@pytest.mark.skipif(
    "AZURE_COSMOS_NOSQL_CONNECTION_STRING" not in os.environ,
    reason="No Azure Cosmos DB connection string provided",
)
@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    @pytest.fixture
    def document_store(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
            database_name="haystack_db",
            container_name="haystack_container",
            vector_embedding_policy=get_vector_embedding_policy("float32", 768, "cosine"),
            indexing_policy=get_vector_indexing_policy("quantized_flat"),
            cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        )
        yield store

    def test_write_documents(self, document_store: AzureCosmosDBNoSqlDocumentStore):
        docs = [Document(content="some text")]
        assert document_store.write_documents(docs) == 1

    def test_complex_filter(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {
            "where": "WHERE c.meta.number=100 and c.meta.chapter='intro'"
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if (d.meta.get("number") == 100)
            ],
        )
        document_store.delete_documents(delete_all=True)
