# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import Mock

from haystack.dataclasses import Document
from haystack.utils.auth import EnvVarSecret
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@pytest.fixture
def document_store():
    store = MongoDBAtlasDocumentStore(
        database_name="haystack_integration_test",
        collection_name="test_embeddings_collection",
        vector_search_index="cosine_index",
    )
    return store


class TestRetriever:
    def test_init_default(self, document_store: MongoDBAtlasDocumentStore):
        retriever = MongoDBAtlasEmbeddingRetriever(document_store=document_store)
        assert retriever.document_store == document_store
        assert retriever.filters == {}
        assert retriever.top_k == 10

    def test_init(self, document_store: MongoDBAtlasDocumentStore):
        retriever = MongoDBAtlasEmbeddingRetriever(
            document_store=document_store, filters={"field": "value"}, top_k=5,
        )
        assert retriever.document_store == document_store
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5

    def test_to_dict(self, document_store: MongoDBAtlasDocumentStore):
        retriever = MongoDBAtlasEmbeddingRetriever(
            document_store=document_store, filters={"field": "value"}, top_k=5
        )
        res = retriever.to_dict()
        t = "haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever"
        assert res == {
            "type": t,
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",
                    "init_parameters": {
                        "mongo_connection_string": {"env_vars": ["MONGO_CONNECTION_STRING"], "strict": True, "type": "env_var"},
                        "database_name": "haystack_integration_test",
                        "collection_name": "test_embeddings_collection",
                        "vector_search_index": "cosine_index",
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
            },
        }

    def test_from_dict(self):
        t = "haystack_integrations.components.retrievers.mongodb_atlas.embedding_retriever.MongoDBAtlasEmbeddingRetriever"
        data = {
            "type": t,
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",
                    "init_parameters": {
                        "mongo_connection_string": {"env_vars": ["MONGO_CONNECTION_STRING"], "strict": True, "type": "env_var"},
                        "database_name": "haystack_integration_test",
                        "collection_name": "test_embeddings_collection",
                        "vector_search_index": "cosine_index",
                    },
                },
                "filters": {"field": "value"},
                "top_k": 5,
            },
        }

        retriever = MongoDBAtlasEmbeddingRetriever.from_dict(data)
        document_store = retriever.document_store

        assert isinstance(document_store, MongoDBAtlasDocumentStore)
        assert isinstance(document_store.mongo_connection_string, EnvVarSecret)
        assert document_store.database_name == "haystack_integration_test"
        assert document_store.collection_name == "test_embeddings_collection"
        assert document_store.vector_search_index == "cosine_index"
        assert retriever.filters == {"field": "value"}
        assert retriever.top_k == 5

    def test_run(self):
        mock_store = Mock(spec=MongoDBAtlasDocumentStore)
        doc = Document(content="Test doc", embedding=[0.1, 0.2])
        mock_store.embedding_retrieval.return_value = [doc]

        retriever = MongoDBAtlasEmbeddingRetriever(document_store=mock_store)
        res = retriever.run(query_embedding=[0.3, 0.5])

        mock_store.embedding_retrieval.assert_called_once_with(
            query_embedding=[0.3, 0.5], filters={}, top_k=10
        )

        assert res == {"documents": [doc]}
