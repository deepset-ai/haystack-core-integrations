# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.azure_documentdb import (
    AzureDocumentDBEmbeddingRetriever,
    AzureDocumentDBFullTextRetriever,
)
from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore


@pytest.fixture
def mock_store():
    return Mock(spec=AzureDocumentDBDocumentStore)


def test_embedding_retriever_run(mock_store):
    document = Document(content="Azure DocumentDB")
    mock_store._embedding_retrieval.return_value = [document]
    retriever = AzureDocumentDBEmbeddingRetriever(document_store=mock_store, top_k=3)
    assert retriever.run(query_embedding=[0.1, 0.2]) == {"documents": [document]}
    mock_store._embedding_retrieval.assert_called_once_with(query_embedding=[0.1, 0.2], filters={}, top_k=3)


async def test_embedding_retriever_run_async(mock_store):
    document = Document(content="Azure DocumentDB")
    mock_store._embedding_retrieval_async = AsyncMock(return_value=[document])
    retriever = AzureDocumentDBEmbeddingRetriever(document_store=mock_store)
    assert await retriever.run_async(query_embedding=[0.1]) == {"documents": [document]}
    mock_store._embedding_retrieval_async.assert_awaited_once_with(query_embedding=[0.1], filters={}, top_k=10)


def test_full_text_retriever_merges_filters(mock_store):
    document = Document(content="Azure DocumentDB")
    mock_store._full_text_retrieval.return_value = [document]
    initial = {"field": "meta.kind", "operator": "==", "value": "guide"}
    runtime = {"field": "meta.year", "operator": ">=", "value": 2026}
    retriever = AzureDocumentDBFullTextRetriever(
        document_store=mock_store, filters=initial, filter_policy=FilterPolicy.MERGE
    )
    assert retriever.run(query="azure", filters=runtime) == {"documents": [document]}
    expected = {"operator": "AND", "conditions": [initial, runtime]}
    mock_store._full_text_retrieval.assert_called_once_with(query="azure", fuzzy=None, filters=expected, top_k=10)


@pytest.mark.parametrize("retriever_class", [AzureDocumentDBEmbeddingRetriever, AzureDocumentDBFullTextRetriever])
def test_retriever_rejects_wrong_store(retriever_class):
    with pytest.raises(ValueError, match="AzureDocumentDBDocumentStore"):
        retriever_class(document_store=object())


def test_retriever_serialization():
    store = AzureDocumentDBDocumentStore(
        database_name="db", collection_name="collection", cluster_name="cluster", mongo_connection_string=None
    )
    retriever = AzureDocumentDBEmbeddingRetriever(document_store=store, top_k=5)
    restored = AzureDocumentDBEmbeddingRetriever.from_dict(retriever.to_dict())
    assert restored.document_store.database_name == "db"
    assert restored.document_store.cluster_name == "cluster"
    assert restored.top_k == 5


async def test_full_text_retriever_run_async(mock_store):
    document = Document(content="Azure DocumentDB")
    mock_store._full_text_retrieval_async = AsyncMock(return_value=[document])
    retriever = AzureDocumentDBFullTextRetriever(document_store=mock_store, filter_policy="replace")
    assert await retriever.run_async(query="azure", fuzzy={"maxEdits": 1}, top_k=2) == {"documents": [document]}
    mock_store._full_text_retrieval_async.assert_awaited_once_with(
        query="azure", fuzzy={"maxEdits": 1}, filters={}, top_k=2
    )


def test_full_text_retriever_serialization():
    store = AzureDocumentDBDocumentStore(
        database_name="db",
        collection_name="collection",
        cluster_name="cluster",
        full_text_search_index="text-index",
        mongo_connection_string=None,
    )
    retriever = AzureDocumentDBFullTextRetriever(document_store=store, top_k=5, filter_policy="merge")
    restored = AzureDocumentDBFullTextRetriever.from_dict(retriever.to_dict())
    assert restored.document_store.full_text_search_index == "text-index"
    assert restored.top_k == 5
    assert restored.filter_policy is FilterPolicy.MERGE


async def test_retriever_cleanup(mock_store):
    mock_store.close_async = AsyncMock()
    retriever = AzureDocumentDBEmbeddingRetriever(document_store=mock_store)
    retriever.close()
    await retriever.close_async()
    mock_store.close.assert_called_once_with()
    mock_store.close_async.assert_awaited_once_with()


@pytest.mark.parametrize("retriever_class", [AzureDocumentDBEmbeddingRetriever, AzureDocumentDBFullTextRetriever])
def test_retriever_rejects_invalid_top_k(mock_store, retriever_class):
    with pytest.raises(ValueError, match="top_k"):
        retriever_class(document_store=mock_store, top_k=0)
