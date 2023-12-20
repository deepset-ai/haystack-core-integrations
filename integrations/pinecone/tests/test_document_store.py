import time
from unittest.mock import patch

from haystack import Document

from pinecone_haystack.document_store import PineconeDocumentStore


class TestDocumentStore:
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

        assert document_store.environment == "gcp-starter"
        assert document_store.index == "my_index"
        assert document_store.namespace == "test"
        assert document_store.batch_size == 50
        assert document_store.dimension == 30
        assert document_store.index_creation_kwargs == {"metric": "euclidean"}

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

    @patch("pinecone_haystack.document_store.pinecone")
    def test_from_dict(self, mock_pinecone):
        mock_pinecone.Index.return_value.describe_index_stats.return_value = {"dimension": 30}

        data = {
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

        document_store = PineconeDocumentStore.from_dict(data)
        assert document_store.environment == "gcp-starter"
        assert document_store.index == "my_index"
        assert document_store.namespace == "test"
        assert document_store.batch_size == 50
        assert document_store.dimension == 30
        assert document_store.index_creation_kwargs == {"metric": "euclidean"}

    def test_embedding_retrieval(self, document_store: PineconeDocumentStore, sleep_time):
        docs = [
            Document(content="Most similar document", embedding=[1.0] * 10),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 0.8, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5]),
            Document(content="Not very similar document", embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ]
        document_store.write_documents(docs)
        time.sleep(sleep_time)
        results = document_store._embedding_retrieval(query_embedding=[0.1] * 10, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"
