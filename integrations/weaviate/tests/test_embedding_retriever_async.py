# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest

from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@pytest.mark.asyncio
async def test_run_async_calls_async_retrieval():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._embedding_retrieval_async = AsyncMock(return_value=[])

    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store)

    await retriever.run_async(
        query_embedding=[0.1, 0.2, 0.3],
        filters={"field": "content", "operator": "==", "value": "test"},
        top_k=5,
        distance=0.5,
    )

    mock_document_store._embedding_retrieval_async.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        filters={"field": "content", "operator": "==", "value": "test"},
        top_k=5,
        distance=0.5,
        certainty=None,
    )


@pytest.mark.asyncio
async def test_run_async_distance_and_certainty_error():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    retriever = WeaviateEmbeddingRetriever(document_store=mock_document_store)

    with pytest.raises(ValueError, match="Can't use 'distance' and 'certainty' parameters together"):
        await retriever.run_async(query_embedding=[0.1, 0.2, 0.3], distance=0.5, certainty=0.8)
