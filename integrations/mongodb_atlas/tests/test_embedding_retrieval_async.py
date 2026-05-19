# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock

import pytest
from haystack.document_stores.errors import DocumentStoreError


class TestEmbeddingRetrievalUnit:
    async def test_raises_for_empty_query_embedding(self, mocked_store_collection_async):
        store, _ = mocked_store_collection_async
        with pytest.raises(ValueError, match="Query embedding must not be empty"):
            await store._embedding_retrieval_async(query_embedding=[])

    async def test_wraps_exception_with_filters_hint(self, mocked_store_collection_async):
        store, collection = mocked_store_collection_async
        collection.aggregate = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(DocumentStoreError, match="vector_search_index"):
            await store._embedding_retrieval_async(
                query_embedding=[0.1, 0.2],
                filters={"field": "meta.f", "operator": "==", "value": "x"},
            )
