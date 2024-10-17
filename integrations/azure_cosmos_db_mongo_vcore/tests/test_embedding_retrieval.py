import os
from typing import List

import pytest
from haystack.document_stores.errors.errors import DocumentStoreError
from haystack.utils import Secret

from haystack_integrations.document_stores.mongodb_vcore import \
    AzureCosmosDBMongoVCoreDocumentStore
from test_utils import get_documents

vector_search_kwargs = {
    "dimensions": 768,
    "num_lists": 1,
    "similarity": "COS",
    "kind": "vector-hnsw",
    "m": 2,
    "ef_construction": 64,
    "ef_search": 40
}


@pytest.mark.skipif(
    "AZURE_COSMOS_MONGO_CONNECTION_STRING" not in os.environ,
    reason="No Azure Cosmos DB connection string provided",
)
@pytest.mark.integration
class TestEmbeddingRetrieval:
    def test_embedding_retrieval_cosine(self):
        store = AzureCosmosDBMongoVCoreDocumentStore(
            mongo_connection_string=Secret.from_env_var("AZURE_COSMOS_MONGO_CONNECTION_STRING"),
            database_name="haystack_db",
            collection_name="haystack_collection",
            vector_search_index_name="haystack_index",
            vector_search_kwargs=vector_search_kwargs,
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["similarityScore"] > results[1].meta["similarityScore"]
        store.delete_documents(delete_all=True)

    def test_embedding_retrieval_euclidean(self):
        vector_search_kwargs["similarity"] = "L2"
        store = AzureCosmosDBMongoVCoreDocumentStore(
            mongo_connection_string=Secret.from_env_var("AZURE_COSMOS_MONGO_CONNECTION_STRING"),
            database_name="haystack_db",
            collection_name="haystack_collection",
            vector_search_index_name="haystack_index",
            vector_search_kwargs=vector_search_kwargs,
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["similarityScore"] > results[1].meta["similarityScore"]
        store.delete_documents(delete_all=True)

    def test_embedding_retrieval_inner_product(self):
        vector_search_kwargs["similarity"] = "IP"
        store = AzureCosmosDBMongoVCoreDocumentStore(
            mongo_connection_string=Secret.from_env_var("AZURE_COSMOS_MONGO_CONNECTION_STRING"),
            database_name="haystack_db",
            collection_name="haystack_collection",
            vector_search_index_name="haystack_index",
            vector_search_kwargs=vector_search_kwargs,
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["similarityScore"] > results[1].meta["similarityScore"]
        store.delete_documents(delete_all=True)

    def test_empty_query_embedding(self):
        store = AzureCosmosDBMongoVCoreDocumentStore(
            mongo_connection_string=Secret.from_env_var("AZURE_COSMOS_MONGO_CONNECTION_STRING"),
            database_name="haystack_db",
            collection_name="haystack_collection",
            vector_search_index_name="haystack_index",
            vector_search_kwargs=vector_search_kwargs,
        )
        query_embedding: List[float] = []
        with pytest.raises(DocumentStoreError):
            store._embedding_retrieval(query_embedding=query_embedding)


