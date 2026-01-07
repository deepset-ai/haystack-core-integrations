# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest

from haystack_integrations.components.retrievers.weaviate import WeaviateBM25Retriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@pytest.mark.asyncio
async def test_run_async_calls_async_retrieval():
    mock_document_store = Mock(spec=WeaviateDocumentStore)
    mock_document_store._bm25_retrieval_async = AsyncMock(return_value=[])

    retriever = WeaviateBM25Retriever(document_store=mock_document_store)

    await retriever.run_async(
        query="test query",
        filters={"field": "content", "operator": "==", "value": "test"},
        top_k=5,
    )

    mock_document_store._bm25_retrieval_async.assert_called_once_with(
        query="test query",
        filters={"field": "content", "operator": "==", "value": "test"},
        top_k=5,
    )
