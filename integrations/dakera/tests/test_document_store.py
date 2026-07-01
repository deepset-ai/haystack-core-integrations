# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from dakera import NotFoundError
from haystack import Document
from haystack.dataclasses import ByteStream
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.dakera import DakeraDocumentStore

CLIENT_PATH = "haystack_integrations.document_stores.dakera.document_store.DakeraClient"


@patch(CLIENT_PATH)
def test_init_is_lazy(mock_client):
    _ = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"))
    mock_client.assert_not_called()


@patch(CLIENT_PATH)
def test_init_adopts_existing_dimension(mock_client):
    mock_client.return_value.get_namespace.return_value = SimpleNamespace(dimensions=60, vector_count=0)

    store = DakeraDocumentStore(
        api_key=Secret.from_token("dk-fake"),
        url="http://dakera.internal:3000",
        namespace="docs",
        dimension=30,
        metric="euclidean",
        batch_size=50,
    )
    store._initialize_client()

    mock_client.assert_called_once_with(base_url="http://dakera.internal:3000", api_key="dk-fake")
    assert store.namespace == "docs"
    assert store.batch_size == 50
    assert store.metric == "euclidean"
    # The dimension of the existing namespace takes precedence over the requested one.
    assert store.dimension == 60


@patch(CLIENT_PATH)
def test_init_creates_namespace_when_missing(mock_client):
    mock_client.return_value.get_namespace.side_effect = NotFoundError("missing")

    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="new", dimension=128)
    store._initialize_client()

    mock_client.return_value.configure_namespace.assert_called_once()
    assert store.dimension == 128


@patch(CLIENT_PATH)
def test_init_api_key_in_environment_variable(mock_client, monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "dk-env")
    mock_client.return_value.get_namespace.return_value = SimpleNamespace(dimensions=768, vector_count=0)

    store = DakeraDocumentStore(namespace="docs")
    store._initialize_client()

    mock_client.assert_called_once_with(base_url="http://localhost:3000", api_key="dk-env")


def test_init_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unsupported metric"):
        DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), metric="manhattan")


def test_init_fails_wo_api_key(monkeypatch):
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        DakeraDocumentStore(namespace="docs")._initialize_client()


def test_to_from_dict(monkeypatch):
    monkeypatch.setenv("DAKERA_API_KEY", "dk-env")
    store = DakeraDocumentStore(
        url="http://dakera.internal:3000",
        namespace="docs",
        dimension=384,
        metric="dot_product",
        batch_size=50,
    )

    expected = {
        "type": "haystack_integrations.document_stores.dakera.document_store.DakeraDocumentStore",
        "init_parameters": {
            "api_key": {"env_vars": ["DAKERA_API_KEY"], "strict": True, "type": "env_var"},
            "url": "http://dakera.internal:3000",
            "namespace": "docs",
            "dimension": 384,
            "metric": "dot_product",
            "batch_size": 50,
        },
    }
    assert store.to_dict() == expected

    restored = DakeraDocumentStore.from_dict(expected)
    assert restored.api_key == Secret.from_env_var("DAKERA_API_KEY", strict=True)
    assert restored.url == "http://dakera.internal:3000"
    assert restored.namespace == "docs"
    assert restored.dimension == 384
    assert restored.metric == "dot_product"
    assert restored.batch_size == 50


@patch(CLIENT_PATH)
def test_count_documents(mock_client):
    mock_client.return_value.get_namespace.return_value = SimpleNamespace(dimensions=768, vector_count=7)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    assert store.count_documents() == 7


@patch(CLIENT_PATH)
def test_count_documents_missing_namespace(mock_client):
    mock_client.return_value.get_namespace.side_effect = NotFoundError("missing")
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    assert store.count_documents() == 0


@patch(CLIENT_PATH)
def test_write_documents_upserts_vectors(mock_client):
    mock_client.return_value.get_namespace.return_value = SimpleNamespace(dimensions=3, vector_count=0)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs", dimension=3, batch_size=2)

    docs = [
        Document(id="a", content="alpha", embedding=[0.1, 0.2, 0.3], meta={"category": "x"}),
        Document(id="b", content="beta", embedding=[0.4, 0.5, 0.6]),
        Document(id="c", content="gamma", embedding=[0.7, 0.8, 0.9]),
    ]
    written = store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

    assert written == 3
    # 3 documents with batch_size=2 => two upsert calls.
    assert mock_client.return_value.upsert.call_count == 2
    first_batch = mock_client.return_value.upsert.call_args_list[0].kwargs["vectors"]
    assert first_batch[0].id == "a"
    assert first_batch[0].values == [0.1, 0.2, 0.3]
    # Content is stored as metadata alongside the document meta.
    assert first_batch[0].metadata == {"category": "x", "content": "alpha"}


@patch(CLIENT_PATH)
def test_delete_documents_noop_on_empty(mock_client):
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    store.delete_documents([])
    mock_client.assert_not_called()


@patch(CLIENT_PATH)
def test_delete_documents(mock_client):
    mock_client.return_value.get_namespace.return_value = SimpleNamespace(dimensions=768, vector_count=0)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    store.delete_documents(["a", "b"])
    mock_client.return_value.delete.assert_called_once_with("docs", ids=["a", "b"])


def test_discard_invalid_meta():
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"))
    doc = Document(
        id="a",
        content="alpha",
        meta={"keep_str": "x", "keep_int": 1, "keep_list": ["a", "b"], "drop_dict": {"nested": 1}},
    )
    cleaned = store._discard_invalid_meta(doc)
    assert cleaned.meta == {"keep_str": "x", "keep_int": 1, "keep_list": ["a", "b"]}


def test_prepare_documents_rejects_non_documents():
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"))
    with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
        store._prepare_documents_for_writing(["not-a-document"], DuplicatePolicy.NONE)


def test_convert_documents_to_vectors_handles_missing_embedding_and_blob():
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), dimension=3)
    doc = Document(id="a", content="alpha", blob=ByteStream(b"data"))
    vectors = store._convert_documents_to_vectors([doc])
    # A missing embedding falls back to the dummy vector; blob is ignored but content kept.
    assert vectors[0].values == store._dummy_vector
    assert vectors[0].metadata == {"content": "alpha"}


def test_convert_query_result_drops_dummy_embedding():
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), dimension=3)
    result = SimpleNamespace(
        results=[
            SimpleNamespace(id="a", score=0.9, values=store._dummy_vector, metadata={"content": "alpha"}),
            SimpleNamespace(id="b", score=0.8, values=[0.1, 0.2, 0.3], metadata={"content": "beta", "k": "v"}),
        ]
    )
    docs = store._convert_query_result_to_documents(result)
    assert docs[0].id == "a"
    assert docs[0].content == "alpha"
    assert docs[0].embedding is None
    assert docs[1].embedding == [0.1, 0.2, 0.3]
    assert docs[1].meta == {"k": "v"}


def test_embedding_retrieval_rejects_empty_query():
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"))
    with pytest.raises(ValueError, match="non-empty list of floats"):
        store._embedding_retrieval(query_embedding=[])


@patch(CLIENT_PATH)
def test_embedding_retrieval_passes_filters_and_top_k(mock_client):
    mock_client.return_value.get_namespace.return_value = SimpleNamespace(dimensions=3, vector_count=0)
    mock_client.return_value.query.return_value = SimpleNamespace(results=[])
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs", dimension=3)

    filters = {"field": "meta.category", "operator": "==", "value": "x"}
    store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3], filters=filters, top_k=5)

    call = mock_client.return_value.query.call_args
    assert call.args[0] == "docs"
    assert call.kwargs["top_k"] == 5
    assert call.kwargs["filter"] == {"category": {"$eq": "x"}}
    assert call.kwargs["include_metadata"] is True
