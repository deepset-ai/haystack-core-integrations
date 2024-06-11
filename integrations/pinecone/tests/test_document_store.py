import os
import time
from unittest.mock import patch

import numpy as np
import pytest
from haystack import Document
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack.utils import Secret
from pinecone import Pinecone, PodSpec, ServerlessSpec

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_init_is_lazy(_mock_client):
    _ = PineconeDocumentStore(api_key=Secret.from_token("fake-api-key"))
    _mock_client.assert_not_called()


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_init(mock_pinecone):
    mock_pinecone.return_value.Index.return_value.describe_index_stats.return_value = {"dimension": 60}

    document_store = PineconeDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    # Trigger an actual connection
    _ = document_store.index

    mock_pinecone.assert_called_with(api_key="fake-api-key", source_tag="haystack")

    assert document_store.index_name == "my_index"
    assert document_store.namespace == "test"
    assert document_store.batch_size == 50
    assert document_store.dimension == 60
    assert document_store.metric == "euclidean"


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_init_api_key_in_environment_variable(mock_pinecone, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")

    ds = PineconeDocumentStore(
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    # Trigger an actual connection
    _ = ds.index

    mock_pinecone.assert_called_with(api_key="env-api-key", source_tag="haystack")


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_to_from_dict(mock_pinecone, monkeypatch):
    mock_pinecone.return_value.Index.return_value.describe_index_stats.return_value = {"dimension": 60}
    monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
    document_store = PineconeDocumentStore(
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    # Trigger an actual connection
    _ = document_store.index

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
            "index": "my_index",
            "dimension": 60,
            "namespace": "test",
            "batch_size": 50,
            "metric": "euclidean",
            "spec": {"serverless": {"region": "us-east-1", "cloud": "aws"}},
        },
    }
    assert document_store.to_dict() == dict_output

    document_store = PineconeDocumentStore.from_dict(dict_output)
    assert document_store.api_key == Secret.from_env_var("PINECONE_API_KEY", strict=True)
    assert document_store.index_name == "my_index"
    assert document_store.namespace == "test"
    assert document_store.batch_size == 50
    assert document_store.dimension == 60
    assert document_store.metric == "euclidean"
    assert document_store.spec == {"serverless": {"region": "us-east-1", "cloud": "aws"}}


def test_init_fails_wo_api_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        _ = PineconeDocumentStore(
            index="my_index",
        ).index


def test_convert_dict_spec_to_pinecone_object_serverless():
    dict_spec = {"serverless": {"region": "us-east-1", "cloud": "aws"}}
    pinecone_object = PineconeDocumentStore._convert_dict_spec_to_pinecone_object(dict_spec)
    assert isinstance(pinecone_object, ServerlessSpec)
    assert pinecone_object.region == "us-east-1"
    assert pinecone_object.cloud == "aws"


def test_convert_dict_spec_to_pinecone_object_pod():
    dict_spec = {"pod": {"replicas": 1, "shards": 1, "pods": 1, "pod_type": "p1.x1", "environment": "us-west1-gcp"}}
    pinecone_object = PineconeDocumentStore._convert_dict_spec_to_pinecone_object(dict_spec)

    assert isinstance(pinecone_object, PodSpec)
    assert pinecone_object.replicas == 1
    assert pinecone_object.shards == 1
    assert pinecone_object.pods == 1
    assert pinecone_object.pod_type == "p1.x1"
    assert pinecone_object.environment == "us-west1-gcp"


def test_convert_dict_spec_to_pinecone_object_fail():
    dict_spec = {
        "strange_key": {"replicas": 1, "shards": 1, "pods": 1, "pod_type": "p1.x1", "environment": "us-west1-gcp"}
    }
    with pytest.raises(ValueError):
        PineconeDocumentStore._convert_dict_spec_to_pinecone_object(dict_spec)


@pytest.mark.integration
@pytest.mark.skipif("PINECONE_API_KEY" not in os.environ, reason="PINECONE_API_KEY not set")
def test_serverless_index_creation_from_scratch(sleep_time):
    index_name = "my-serverless-index"

    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    try:
        client.delete_index(name=index_name)
    except Exception:  # noqa S110
        pass

    time.sleep(sleep_time)

    ds = PineconeDocumentStore(
        index=index_name,
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
    )
    # Trigger the connection
    _ = ds.index

    index_description = client.describe_index(name=index_name)
    assert index_description["name"] == index_name
    assert index_description["dimension"] == 30
    assert index_description["metric"] == "euclidean"
    assert index_description["spec"]["serverless"]["region"] == "us-east-1"
    assert index_description["spec"]["serverless"]["cloud"] == "aws"

    try:
        client.delete_index(name=index_name)
    except Exception:  # noqa S110
        pass


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
