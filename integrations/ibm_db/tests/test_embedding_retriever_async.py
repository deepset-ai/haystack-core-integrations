# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Async tests for IBMDb2EmbeddingRetriever."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.utils import Secret

from haystack_integrations.components.retrievers.ibm_db import IBMDb2EmbeddingRetriever
from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_document_store() -> IBMDb2DocumentStore:
    """Return an IBMDb2DocumentStore with all async DB methods mocked out."""
    store = IBMDb2DocumentStore(
        database="testdb",
        hostname="localhost",
        username=Secret.from_token("user"),
        password=Secret.from_token("pass"),
        table_name="haystack_mock",
        embedding_dim=768,
    )
    store._embedding_retrieval_async = AsyncMock(return_value=[])  # type: ignore[method-assign]
    store.close_async = AsyncMock()  # type: ignore[method-assign]
    return store


@pytest.fixture
def retriever(mock_document_store: IBMDb2DocumentStore) -> IBMDb2EmbeddingRetriever:
    return IBMDb2EmbeddingRetriever(document_store=mock_document_store, top_k=5)


# ---------------------------------------------------------------------------
# Unit tests — no live DB2 required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_async_calls_embedding_retrieval(retriever: IBMDb2EmbeddingRetriever) -> None:
    """run_async() delegates to _embedding_retrieval_async with the right arguments."""
    query = [0.1] * 768
    expected = [Document(content="doc1"), Document(content="doc2")]
    retriever.document_store._embedding_retrieval_async.return_value = expected  # type: ignore[attr-defined]

    result = await retriever.run_async(query_embedding=query)

    retriever.document_store._embedding_retrieval_async.assert_awaited_once_with(  # type: ignore[attr-defined]
        query, filters={}, top_k=5
    )
    assert result == {"documents": expected}


@pytest.mark.asyncio
async def test_run_async_respects_top_k_override(retriever: IBMDb2EmbeddingRetriever) -> None:
    query = [0.1] * 768
    await retriever.run_async(query_embedding=query, top_k=3)

    retriever.document_store._embedding_retrieval_async.assert_awaited_once_with(  # type: ignore[attr-defined]
        query, filters={}, top_k=3
    )


@pytest.mark.asyncio
async def test_run_async_merges_filters_replace_policy(mock_document_store: IBMDb2DocumentStore) -> None:
    """FilterPolicy.REPLACE: runtime filters fully replace constructor filters."""
    retriever = IBMDb2EmbeddingRetriever(
        document_store=mock_document_store,
        filters={"field": "meta.env", "operator": "==", "value": "prod"},
        filter_policy=FilterPolicy.REPLACE,
    )
    runtime_filters = {"field": "meta.env", "operator": "==", "value": "test"}

    await retriever.run_async(query_embedding=[0.0] * 768, filters=runtime_filters)

    mock_document_store._embedding_retrieval_async.assert_awaited_once_with(  # type: ignore[attr-defined]
        [0.0] * 768, filters=runtime_filters, top_k=10
    )


@pytest.mark.asyncio
async def test_run_async_merges_filters_merge_policy(mock_document_store: IBMDb2DocumentStore) -> None:
    """FilterPolicy.MERGE: runtime and constructor filters are combined."""
    constructor_filters = {"field": "meta.env", "operator": "==", "value": "prod"}
    retriever = IBMDb2EmbeddingRetriever(
        document_store=mock_document_store,
        filters=constructor_filters,
        filter_policy=FilterPolicy.MERGE,
    )
    runtime_filters = {"field": "meta.region", "operator": "==", "value": "eu"}

    await retriever.run_async(query_embedding=[0.0] * 768, filters=runtime_filters)

    call_kwargs = mock_document_store._embedding_retrieval_async.call_args  # type: ignore[attr-defined]
    merged = call_kwargs.kwargs["filters"]
    # Merged result must reference both conditions
    assert merged is not None


@pytest.mark.asyncio
async def test_run_async_no_filters(retriever: IBMDb2EmbeddingRetriever) -> None:
    """run_async() without runtime filters passes the constructor filters."""
    await retriever.run_async(query_embedding=[0.0] * 768)
    retriever.document_store._embedding_retrieval_async.assert_awaited_once_with(  # type: ignore[attr-defined]
        [0.0] * 768, filters={}, top_k=5
    )


@pytest.mark.asyncio
async def test_run_async_returns_correct_output_shape(retriever: IBMDb2EmbeddingRetriever) -> None:
    """run_async() output always has the 'documents' key."""
    result = await retriever.run_async(query_embedding=[0.0] * 768)
    assert "documents" in result
    assert isinstance(result["documents"], list)


@pytest.mark.asyncio
async def test_close_async_only_closes_async_connection(retriever: IBMDb2EmbeddingRetriever) -> None:
    """close_async() on the retriever delegates to document_store.close_async() only."""
    await retriever.close_async()
    retriever.document_store.close_async.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_close_async_does_not_call_sync_close(retriever: IBMDb2EmbeddingRetriever) -> None:
    """Calling close_async() must not touch the synchronous close()."""
    retriever.document_store.close = MagicMock()  # type: ignore[method-assign]
    await retriever.close_async()
    retriever.document_store.close.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Integration tests — require a live DB2 instance (docker-compose)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
class TestEmbeddingRetrieverAsync:
    """Live async integration tests for IBMDb2EmbeddingRetriever.run_async()."""

    @pytest.fixture
    def retriever_live(self, document_store: IBMDb2DocumentStore) -> IBMDb2EmbeddingRetriever:
        return IBMDb2EmbeddingRetriever(document_store=document_store, top_k=3)

    async def test_run_async_returns_top_k_results(
        self, retriever_live: IBMDb2EmbeddingRetriever, document_store: IBMDb2DocumentStore
    ) -> None:
        query = [0.1] * 768
        most_similar = [0.8] * 768
        second_best = [0.8] * 700 + [0.1] * 68
        # Use a non-zero vector — zero vectors cause SQL0801N (division by zero) with COSINE metric
        other = [0.01] * 768

        docs = [
            Document(content="Most similar", embedding=most_similar),
            Document(content="Second best", embedding=second_best),
            Document(content="Other", embedding=other),
        ]
        await document_store.write_documents_async(docs)

        result = await retriever_live.run_async(query_embedding=query)
        assert "documents" in result
        assert len(result["documents"]) == 3
        # First result should be most similar
        assert result["documents"][0].content == "Most similar"

    async def test_run_async_with_filters(
        self, retriever_live: IBMDb2EmbeddingRetriever, document_store: IBMDb2DocumentStore
    ) -> None:
        docs = [
            Document(content="Cat A doc 1", embedding=[0.1] * 768, meta={"category": "A"}),
            Document(content="Cat A doc 2", embedding=[0.2] * 768, meta={"category": "A"}),
            Document(content="Cat B doc", embedding=[0.3] * 768, meta={"category": "B"}),
        ]
        await document_store.write_documents_async(docs)

        result = await retriever_live.run_async(
            query_embedding=[0.1] * 768,
            filters={"field": "meta.category", "operator": "==", "value": "A"},
        )
        assert all(d.meta["category"] == "A" for d in result["documents"])
        assert len(result["documents"]) == 2

    async def test_run_async_empty_store(self, retriever_live: IBMDb2EmbeddingRetriever) -> None:
        result = await retriever_live.run_async(query_embedding=[0.1] * 768)
        assert result == {"documents": []}

    async def test_close_async_does_not_affect_sync_in_live(
        self, retriever_live: IBMDb2EmbeddingRetriever, document_store: IBMDb2DocumentStore
    ) -> None:
        """close_async on retriever only tears down async state; sync path stays usable."""
        _ = document_store.count_documents()  # establish sync connection
        sync_conn = document_store._connection

        await retriever_live.close_async()
        assert document_store._async_connection is None
        assert document_store._connection is sync_conn
