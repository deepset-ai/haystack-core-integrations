from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.utils.auth import EnvVarSecret

from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasFullTextRetriever
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class TestFullTextRetriever:
    @pytest.fixture
    def mock_client(self):
        with patch(
            "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient"
        ) as mock_mongo_client:
            mock_connection = MagicMock()
            mock_database = MagicMock()
            mock_collection_names = MagicMock(return_value=["test_collection"])
            mock_database.list_collection_names = mock_collection_names
            mock_connection.__getitem__.return_value = mock_database
            mock_mongo_client.return_value = mock_connection
            yield mock_mongo_client

    def test_init_default(self):
        mock_store = Mock(spec=MongoDBAtlasDocumentStore)
        retriever = MongoDBAtlasFullTextRetriever(document_store=mock_store)
        assert retriever.document_store == mock_store
        assert retriever.top_k == 10

        retriever = MongoDBAtlasFullTextRetriever(document_store=mock_store)

    def test_to_dict(self, mock_client, monkeypatch):  # noqa: ARG002  mock_client appears unused but is required
        monkeypatch.setenv("MONGO_CONNECTION_STRING", "test_conn_str")

        document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_collection",
            vector_search_index="default",
        )

        retriever = MongoDBAtlasFullTextRetriever(document_store=document_store, top_k=5)
        res = retriever.to_dict()
        assert res == {
            "type": "haystack_integrations.components.retrievers.mongodb_atlas.fulltext_retriever.MongoDBAtlasFullTextRetriever",  # noqa: E501
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",  # noqa: E501
                    "init_parameters": {
                        "mongo_connection_string": {
                            "env_vars": ["MONGO_CONNECTION_STRING"],
                            "strict": True,
                            "type": "env_var",
                        },
                        "database_name": "haystack_integration_test",
                        "collection_name": "test_collection",
                        "vector_search_index": "default",
                    },
                },
                "top_k": 5,
            },
        }

    def test_from_dict(self, mock_client, monkeypatch):  # noqa: ARG002  mock_client appears unused but is required
        monkeypatch.setenv("MONGO_CONNECTION_STRING", "test_conn_str")

        data = {
            "type": "haystack_integrations.components.retrievers.mongodb_atlas.fulltext_retriever.MongoDBAtlasFullTextRetriever",  # noqa: E501
            "init_parameters": {
                "document_store": {
                    "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",  # noqa: E501
                    "init_parameters": {
                        "mongo_connection_string": {
                            "env_vars": ["MONGO_CONNECTION_STRING"],
                            "strict": True,
                            "type": "env_var",
                        },
                        "database_name": "haystack_integration_test",
                        "collection_name": "test_collection",
                        "vector_search_index": "default",
                    },
                },
                "top_k": 5,
            },
        }

        retriever = MongoDBAtlasFullTextRetriever.from_dict(data)
        document_store = retriever.document_store

        assert isinstance(document_store, MongoDBAtlasDocumentStore)
        assert isinstance(document_store.mongo_connection_string, EnvVarSecret)
        assert document_store.database_name == "haystack_integration_test"
        assert document_store.collection_name == "test_collection"
        assert document_store.vector_search_index == "default"
        assert retriever.top_k == 5

    def test_run(self):
        mock_store = Mock(spec=MongoDBAtlasDocumentStore)
        doc = Document(content="Test doc")
        mock_store._fulltext_retrieval.return_value = [doc]

        retriever = MongoDBAtlasFullTextRetriever(document_store=mock_store, search_path="desc")
        res = retriever.run(query="text")

        mock_store._fulltext_retrieval.assert_called_once_with(query="text", top_k=10, search_path="desc")

        assert res == {"documents": [doc]}
