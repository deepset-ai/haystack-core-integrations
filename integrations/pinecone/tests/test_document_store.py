from unittest.mock import patch

import numpy as np
import pytest
from haystack import Document
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest

from pinecone_haystack.document_store import PineconeDocumentStore


class TestDocumentStore(CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest):
    def test_write_documents(self, document_store: PineconeDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_fail(self, document_store: PineconeDocumentStore):
        ...

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_skip(self, document_store: PineconeDocumentStore):
        ...

    @patch("pinecone_haystack.document_store.pinecone")
    def test_init(self, mock_pinecone):
        mock_pinecone.Index.return_value.describe_index_stats.return_value = {"dimension": 30}

        document_store = PineconeDocumentStore(
            api_key="fake-api-key",
            environment="gcp-starter",
            index="my_index",
            namespace="test",
            batch_size=50,
            dimension=30,
            metric="euclidean",
        )

        mock_pinecone.init.assert_called_with(api_key="fake-api-key", environment="gcp-starter")

        assert document_store.environment == "gcp-starter"
        assert document_store.index == "my_index"
        assert document_store.namespace == "test"
        assert document_store.batch_size == 50
        assert document_store.dimension == 30
        assert document_store.index_creation_kwargs == {"metric": "euclidean"}

    @patch("pinecone_haystack.document_store.pinecone")
    def test_init_api_key_in_environment_variable(self, mock_pinecone, monkeypatch):
        monkeypatch.setenv("PINECONE_API_KEY", "fake-api-key")

        PineconeDocumentStore(
            environment="gcp-starter",
            index="my_index",
            namespace="test",
            batch_size=50,
            dimension=30,
            metric="euclidean",
        )

        mock_pinecone.init.assert_called_with(api_key="fake-api-key", environment="gcp-starter")

    def test_init_fails_wo_api_key(self, monkeypatch):
        api_key = None
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            PineconeDocumentStore(
                api_key=api_key,
                environment="gcp-starter",
                index="my_index",
            )

    @patch("pinecone_haystack.document_store.pinecone")
    def test_to_dict(self, mock_pinecone):
        mock_pinecone.Index.return_value.describe_index_stats.return_value = {"dimension": 30}
        document_store = PineconeDocumentStore(
            api_key="fake-api-key",
            environment="gcp-starter",
            index="my_index",
            namespace="test",
            batch_size=50,
            dimension=30,
            metric="euclidean",
        )
        assert document_store.to_dict() == {
            "type": "pinecone_haystack.document_store.PineconeDocumentStore",
            "init_parameters": {
                "environment": "gcp-starter",
                "index": "my_index",
                "dimension": 30,
                "namespace": "test",
                "batch_size": 50,
                "metric": "euclidean",
            },
        }

    def test_embedding_retrieval(self, document_store: PineconeDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = np.random.rand(768).tolist()

        docs = [
            Document(content="Most similar document", embedding=most_similar_embedding),
            Document(content="2nd best document", embedding=second_best_embedding),
            Document(content="Not very similar document", embedding=another_embedding),
        ]

        document_store.write_documents(docs)

        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"
