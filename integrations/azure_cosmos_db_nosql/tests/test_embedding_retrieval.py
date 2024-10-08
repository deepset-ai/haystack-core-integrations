import os
from typing import Any, Dict, List

import pytest
from azure.cosmos import PartitionKey
from haystack.document_stores.errors.errors import DocumentStoreError

from haystack_integrations.document_stores.nosql import \
    AzureCosmosDBNoSqlDocumentStore
from test_utils import get_documents, get_vector_embedding_policy, get_vector_indexing_policy


@pytest.mark.skipif(
    "AZURE_COSMOS_NOSQL_CONNECTION_STRING" not in os.environ,
    reason="No Azure Cosmos DB connection string provided",
)
class TestEmbeddingRetrieval:
    def test_embedding_retrieval_cosine_flat(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
            database_name="haystack_db",
            container_name="haystack_container",
            vector_embedding_policy=get_vector_embedding_policy("float32", 768, "cosine"),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["SimilarityScore"] > results[1].meta["SimilarityScore"]

    def test_embedding_retrieval_euclidean_flat(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
            database_name="haystack_db",
            container_name="haystack_container",
            vector_embedding_policy=get_vector_embedding_policy("float32", 768, "euclidean"),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["SimilarityScore"] > results[1].meta["SimilarityScore"]

    def test_embedding_retrieval_dot_product_flat(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
            database_name="haystack_db",
            container_name="haystack_container",
            vector_embedding_policy=get_vector_embedding_policy("float32", 768, "dotProduct"),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["SimilarityScore"] > results[1].meta["SimilarityScore"]

    def test_embedding_retrieval_cosine_quantized_flat(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
            database_name="haystack_db",
            container_name="haystack_container",
            vector_embedding_policy=get_vector_embedding_policy("float32", 768, "cosine"),
            indexing_policy=get_vector_indexing_policy("quantized_flat"),
            cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["SimilarityScore"] > results[1].meta["SimilarityScore"]

    def test_embedding_retrieval_euclidean_quantized_flat(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
            database_name="haystack_db",
            container_name="haystack_container",
            vector_embedding_policy=get_vector_embedding_policy("float32", 768, "euclidean"),
            indexing_policy=get_vector_indexing_policy("quantized_flat"),
            cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["SimilarityScore"] > results[1].meta["SimilarityScore"]

    def test_embedding_retrieval_dot_product_quantized_flat(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
            database_name="haystack_db",
            container_name="haystack_container",
            vector_embedding_policy=get_vector_embedding_policy("float32", 768, "dotProduct"),
            indexing_policy=get_vector_indexing_policy("quantized_flat"),
            cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        )
        store.write_documents(get_documents())
        query_embedding = [0.1] * 768
        results = store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].meta["SimilarityScore"] > results[1].meta["SimilarityScore"]

    def test_empty_query_embedding(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
            database_name="haystack_db",
            container_name="haystack_container",
            vector_embedding_policy=get_vector_embedding_policy("float32", 768, "dotProduct"),
            indexing_policy=get_vector_indexing_policy("quantized_flat"),
            cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        )
        query_embedding: List[float] = []
        with pytest.raises(DocumentStoreError):
            store._embedding_retrieval(query_embedding=query_embedding)
