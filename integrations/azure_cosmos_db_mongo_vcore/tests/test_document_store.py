import os

import pytest
from haystack.dataclasses.document import Document
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.utils import Secret

from haystack_integrations.document_stores.mongodb_vcore import \
    AzureCosmosDBMongoVCoreDocumentStore


@pytest.mark.skipif(
    "AZURE_COSMOS_MONGO_CONNECTION_STRING" not in os.environ,
    reason="No Azure Cosmos DB connection string provided",
)
@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    @pytest.fixture
    def document_store(self):
        vector_search_kwargs = {
            "dimensions": 768,
            "num_lists": 1,
            "similarity": "COS",
            "kind": "vector-hnsw",
            "m": 2,
            "ef_construction": 64,
            "ef_search": 40
        }
        store = AzureCosmosDBMongoVCoreDocumentStore(
            mongo_connection_string=Secret.from_env_var("AZURE_COSMOS_MONGO_CONNECTION_STRING"),
            database_name="haystack_db",
            collection_name="haystack_collection",
            vector_search_index_name="haystack_index",
            vector_search_kwargs=vector_search_kwargs,
        )

        yield store

    def test_write_document(self, document_store: AzureCosmosDBMongoVCoreDocumentStore):
        docs = [Document(content="some text")]
        assert document_store.write_documents(docs) == 1

    def test_complex_filter(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "==", "value": 100},
                        {"field": "meta.chapter", "operator": "==", "value": "intro"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.page", "operator": "==", "value": "90"},
                        {"field": "meta.chapter", "operator": "==", "value": "conclusion"},
                    ],
                },
            ],
        }

        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if (d.meta.get("number") == 100 and d.meta.get("chapter") == "intro")
                   or (d.meta.get("page") == "90" and d.meta.get("chapter") == "conclusion")
            ],
        )
