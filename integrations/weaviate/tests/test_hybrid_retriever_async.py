# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest

from haystack_integrations.components.retrievers.weaviate import WeaviateHybridRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@pytest.mark.asyncio
async def test_run_async_calls_async_retrieval():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval_async = AsyncMock(return_value=[])

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)

    await retriever.run_async(
        query="test query",
        query_embedding=[0.1, 0.2, 0.3],
        filters={"field": "content", "operator": "==", "value": "test"},
        top_k=5,
        alpha=0.7,
        max_vector_distance=0.5,
    )

    mock_document_store._hybrid_retrieval_async.assert_called_once_with(
        query="test query",
        query_embedding=[0.1, 0.2, 0.3],
        filters={"field": "content", "operator": "==", "value": "test"},
        top_k=5,
        alpha=0.7,
        max_vector_distance=0.5,
    )


@pytest.mark.asyncio
async def test_run_async_with_init_parameters():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._hybrid_retrieval_async = AsyncMock(return_value=[])

    retriever = WeaviateHybridRetriever(
        document_store=mock_document_store,
        top_k=20,
        alpha=0.3,
        max_vector_distance=0.9,
    )

    await retriever.run_async(query="test", query_embedding=[0.1, 0.2, 0.3])

    mock_document_store._hybrid_retrieval_async.assert_called_once_with(
        query="test",
        query_embedding=[0.1, 0.2, 0.3],
        filters={},
        top_k=20,
        alpha=0.3,
        max_vector_distance=0.9,
    )


@pytest.mark.asyncio
async def test_run_async_with_invalid_alpha():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    with pytest.raises(ValueError, match=r"alpha \(-0.1\) must be in the range \[0.0, 1.0\]"):
        await retriever.run_async(query="q", query_embedding=[0.1], alpha=-0.1)
    with pytest.raises(ValueError, match=r"alpha \(1.5\) must be in the range \[0.0, 1.0\]"):
        await retriever.run_async(query="q", query_embedding=[0.1], alpha=1.5)
