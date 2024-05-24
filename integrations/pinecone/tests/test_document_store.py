import os
from unittest.mock import patch

import numpy as np
import pytest
from haystack import Document
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack.utils import Secret

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_init(mock_pinecone):
    mock_pinecone.Index.return_value.describe_index_stats.return_value = {"dimension": 30}

    PineconeDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        environment="gcp-starter",
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    mock_pinecone.assert_called()


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_init_api_key_in_environment_variable(mock_pinecone, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")

    PineconeDocumentStore(
        environment="gcp-starter",
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    mock_pinecone.assert_called_with(api_key="env-api-key", source_tag="haystack")


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_to_from_dict(mock_pinecone, monkeypatch):
    mock_pinecone.Index.return_value.describe_index_stats.return_value = {"dimension": 30}
    monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
    document_store = PineconeDocumentStore(
        environment="gcp-starter",
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    dict_output = {
        "type": "haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore",
        "init_parameters": {
            "api_key": {
                "env_vars": [
                    "PINECONE_API_KEY",
                ],
                "strict": True,
                "type": "env_var",
            },
            "environment": "gcp-starter",
            "index": "my_index",
            "dimension": 30,
            "namespace": "test",
            "batch_size": 50,
            "metric": "euclidean",
        },
    }

    document_store = PineconeDocumentStore.from_dict(dict_output)
    assert document_store.environment == "gcp-starter"
    assert document_store.api_key == Secret.from_env_var("PINECONE_API_KEY", strict=True)
    assert document_store.index == "my_index"
    assert document_store.namespace == "test"
    assert document_store.batch_size == 50


def test_init_fails_wo_api_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        PineconeDocumentStore(
            environment="gcp-starter",
            index="my_index",
        )


@pytest.mark.integration
@pytest.mark.skipif("PINECONE_API_KEY" not in os.environ, reason="PINECONE_API_KEY not set")
class TestDocumentStore(CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest):
    def test_write_documents(self, document_store: PineconeDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1

    @pytest.mark.xfail(
        run=True, reason="Pinecone supports overwriting by default, but it takes a while for it to take effect"
    )
    def test_write_documents_duplicate_overwrite(self, document_store: PineconeDocumentStore): ...

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_fail(self, document_store: PineconeDocumentStore): ...

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_skip(self, document_store: PineconeDocumentStore): ...

    @pytest.mark.skip(reason="Pinecone creates a namespace only when the first document is written")
    def test_delete_documents_empty_document_store(self, document_store: PineconeDocumentStore): ...

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
