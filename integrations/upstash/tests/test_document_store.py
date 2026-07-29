import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests

from haystack_integrations.document_stores.upstash import UpstashDocumentStore


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_init_with_defaults(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")
    UpstashDocumentStore()
    mock_index.assert_called_once_with(url="http://test", token="test-token")


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_to_dict_from_dict(mock_index, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")
    store = UpstashDocumentStore()

    data = store.to_dict()
    assert data == {
        "type": "haystack_integrations.document_stores.upstash.document_store.UpstashDocumentStore",
        "init_parameters": {
            "url": {"env_vars": ["UPSTASH_VECTOR_REST_URL"], "strict": True, "type": "env_var"},
            "token": {"env_vars": ["UPSTASH_VECTOR_REST_TOKEN"], "strict": True, "type": "env_var"},
        },
    }

    restored = UpstashDocumentStore.from_dict(data)
    assert restored.url.resolve_value() == "http://test"


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_count_documents(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_info = MagicMock()
    mock_info.vector_count = 42
    mock_instance.info.return_value = mock_info
    mock_index.return_value = mock_instance

    store = UpstashDocumentStore()
    assert store.count_documents() == 42


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_write_documents(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance
    mock_instance.fetch.return_value = []

    store = UpstashDocumentStore()
    docs = [Document(id="1", content="test", embedding=[0.1, 0.2, 0.3])]
    store.write_documents(docs)

    mock_instance.upsert.assert_called_once()
    _args, kwargs = mock_instance.upsert.call_args
    assert len(kwargs["vectors"]) == 1
    assert kwargs["vectors"][0]["id"] == "1"
    assert kwargs["vectors"][0]["vector"] == [0.1, 0.2, 0.3]
    assert kwargs["vectors"][0]["data"] == "test"
    assert "metadata" in kwargs["vectors"][0]


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_write_documents_no_embedding(mock_index, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    store = UpstashDocumentStore()
    docs = [Document(id="1", content="test")]
    with pytest.raises(DocumentStoreError):
        store.write_documents(docs)


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_write_documents_duplicate_fail(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance
    mock_res = MagicMock()
    mock_res.id = "1"
    mock_instance.fetch.return_value = [mock_res]

    store = UpstashDocumentStore()
    docs = [Document(id="1", content="test", embedding=[0.1, 0.2, 0.3])]
    with pytest.raises(DuplicateDocumentError):
        store.write_documents(docs, policy=DuplicatePolicy.FAIL)


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_write_documents_duplicate_skip(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance
    mock_res = MagicMock()
    mock_res.id = "1"
    mock_instance.fetch.return_value = [mock_res]

    store = UpstashDocumentStore()
    docs = [Document(id="1", content="test", embedding=[0.1, 0.2, 0.3])]
    written = store.write_documents(docs, policy=DuplicatePolicy.SKIP)

    assert written == 0
    mock_instance.upsert.assert_not_called()


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_delete_documents(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance

    store = UpstashDocumentStore()
    store.delete_documents(["1", "2"])

    mock_instance.delete.assert_called_once_with(["1", "2"])


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_filter_documents(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_info = MagicMock()
    mock_info.dimension = 3
    mock_instance.info.return_value = mock_info

    mock_res = MagicMock()
    mock_res.id = "1"
    mock_res.data = "test"
    mock_res.vector = [0.1, 0.2, 0.3]
    mock_res.metadata = {"genre": "tech"}
    mock_instance.query.return_value = [mock_res]

    mock_index.return_value = mock_instance

    store = UpstashDocumentStore()
    docs = store.filter_documents(filters={"field": "meta.genre", "operator": "==", "value": "tech"})

    assert len(docs) == 1
    assert docs[0].id == "1"
    assert docs[0].content == "test"
    assert docs[0].meta["genre"] == "tech"

    _args, kwargs = mock_instance.query.call_args
    assert kwargs["vector"] == [1.0, 0.0, 0.0]
    assert kwargs["filter"] == "genre = 'tech'"
    assert kwargs["include_data"] is True


@pytest.mark.integration
@pytest.mark.skipif(
    "UPSTASH_VECTOR_REST_URL" not in os.environ or "UPSTASH_VECTOR_REST_TOKEN" not in os.environ,
    reason="No UPSTASH_VECTOR_REST_URL or UPSTASH_VECTOR_REST_TOKEN provided",
)
class TestUpstashDocumentStore(DocumentStoreBaseTests):
    @pytest.fixture
    def document_store(self) -> UpstashDocumentStore:
        store = UpstashDocumentStore()
        # Clean up before yielding to ensure a pristine state for the base tests
        docs = store.filter_documents()
        if docs:
            store.delete_documents([doc.id for doc in docs])
        yield store

        # Clean up after tests just in case
        docs = store.filter_documents()
        if docs:
            store.delete_documents([doc.id for doc in docs])
