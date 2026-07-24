# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the async methods of DakeraDocumentStore (SDK mocked with AsyncMock)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from dakera import NotFoundError
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.dakera import DakeraDocumentStore

ASYNC_CLIENT_PATH = "haystack_integrations.document_stores.dakera.document_store.AsyncDakeraClient"


def _mock_async_client(mock_cls, *, dimensions=3, vector_count=0, query_results=None):
    client = mock_cls.return_value
    client.get_namespace = AsyncMock(return_value=SimpleNamespace(dimensions=dimensions, vector_count=vector_count))
    client.configure_namespace = AsyncMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.query = AsyncMock(return_value=SimpleNamespace(results=query_results or []))
    client.close = AsyncMock()
    return client


@patch(ASYNC_CLIENT_PATH)
async def test_initialize_async_client_adopts_existing_dimension(mock_cls):
    client = _mock_async_client(mock_cls, dimensions=60)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs", dimension=30)

    result = await store._initialize_async_client()

    assert result is client
    mock_cls.assert_called_once_with(base_url="http://localhost:3000", api_key="dk-fake")
    assert store.dimension == 60
    # The client is cached: a second call does not construct a new one.
    assert await store._initialize_async_client() is client
    assert mock_cls.call_count == 1


@patch(ASYNC_CLIENT_PATH)
async def test_initialize_async_client_creates_namespace_when_missing(mock_cls):
    client = _mock_async_client(mock_cls)
    client.get_namespace = AsyncMock(side_effect=NotFoundError("missing"))
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="new", dimension=128)

    await store._initialize_async_client()

    client.configure_namespace.assert_awaited_once()
    assert store.dimension == 128


@patch(ASYNC_CLIENT_PATH)
async def test_count_documents_async(mock_cls):
    _mock_async_client(mock_cls, dimensions=768, vector_count=7)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    assert await store.count_documents_async() == 7


@patch(ASYNC_CLIENT_PATH)
async def test_count_documents_async_missing_namespace(mock_cls):
    client = _mock_async_client(mock_cls)
    client.get_namespace = AsyncMock(side_effect=NotFoundError("missing"))
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    assert await store.count_documents_async() == 0


@patch(ASYNC_CLIENT_PATH)
async def test_write_documents_async_upserts_in_batches(mock_cls):
    client = _mock_async_client(mock_cls, dimensions=3)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs", dimension=3, batch_size=2)

    docs = [
        Document(id="a", content="alpha", embedding=[0.1, 0.2, 0.3], meta={"category": "x"}),
        Document(id="b", content="beta", embedding=[0.4, 0.5, 0.6]),
        Document(id="c", content="gamma", embedding=[0.7, 0.8, 0.9]),
    ]
    written = await store.write_documents_async(docs, policy=DuplicatePolicy.OVERWRITE)

    assert written == 3
    # 3 documents with batch_size=2 => two upsert calls.
    assert client.upsert.await_count == 2
    first_batch = client.upsert.await_args_list[0].kwargs["vectors"]
    assert first_batch[0].id == "a"
    assert first_batch[0].metadata == {"category": "x", "_dakera_content": "alpha"}


@patch(ASYNC_CLIENT_PATH)
async def test_delete_documents_async(mock_cls):
    client = _mock_async_client(mock_cls)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    await store.delete_documents_async(["a", "b"])
    client.delete.assert_awaited_once_with("docs", ids=["a", "b"])


@patch(ASYNC_CLIENT_PATH)
async def test_delete_documents_async_noop_on_empty(mock_cls):
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    await store.delete_documents_async([])
    mock_cls.assert_not_called()


@patch(ASYNC_CLIENT_PATH)
async def test_delete_documents_async_missing_namespace_is_swallowed(mock_cls):
    client = _mock_async_client(mock_cls)
    client.delete = AsyncMock(side_effect=NotFoundError("missing"))
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    # Should not raise.
    await store.delete_documents_async(["a"])


@patch(ASYNC_CLIENT_PATH)
async def test_embedding_retrieval_async_passes_filters_and_top_k(mock_cls):
    client = _mock_async_client(mock_cls, dimensions=3)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs", dimension=3)

    filters = {"field": "meta.category", "operator": "==", "value": "x"}
    await store._embedding_retrieval_async(query_embedding=[0.1, 0.2, 0.3], filters=filters, top_k=5)

    call = client.query.await_args
    assert call.args[0] == "docs"
    assert call.kwargs["top_k"] == 5
    assert call.kwargs["filter"] == {"category": {"$eq": "x"}}
    assert call.kwargs["include_metadata"] is True


async def test_embedding_retrieval_async_rejects_empty_query():
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"))
    with pytest.raises(ValueError, match="non-empty list of floats"):
        await store._embedding_retrieval_async(query_embedding=[])


@patch(ASYNC_CLIENT_PATH)
async def test_filter_documents_async(mock_cls):
    _mock_async_client(
        mock_cls,
        dimensions=3,
        query_results=[
            SimpleNamespace(id="a", score=0.9, values=[0.1, 0.2, 0.3], metadata={"_dakera_content": "alpha", "k": "v"})
        ],
    )
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs", dimension=3)

    docs = await store.filter_documents_async(filters={"field": "meta.k", "operator": "==", "value": "v"})

    assert len(docs) == 1
    assert docs[0].content == "alpha"
    assert docs[0].meta == {"k": "v"}
    # A dummy query vector produces meaningless scores, so they are dropped.
    assert docs[0].score is None


@patch(ASYNC_CLIENT_PATH)
async def test_close_async(mock_cls):
    client = _mock_async_client(mock_cls)
    store = DakeraDocumentStore(api_key=Secret.from_token("dk-fake"), namespace="docs")
    await store._initialize_async_client()

    await store.close_async()

    client.close.assert_awaited_once()
    assert store._async_client is None
    # Closing again is a no-op.
    await store.close_async()
