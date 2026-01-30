# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.opensearch import OpenSearchMetadataRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


@pytest.mark.asyncio
async def test_run_async_with_runtime_document_store():
    mock_store1 = Mock(spec=OpenSearchDocumentStore)
    mock_store2 = Mock(spec=OpenSearchDocumentStore)
    mock_store2._metadata_search_async = AsyncMock(return_value=[{"category": "Java"}])
    retriever = OpenSearchMetadataRetriever(document_store=mock_store1, metadata_fields=["category"])

    result = await retriever.run_async(query="Java", document_store=mock_store2)

    assert result == {"metadata": [{"category": "Java"}]}
    mock_store2._metadata_search_async.assert_called_once()
    # Check that the call includes the new parameters with default values
    call_args = mock_store2._metadata_search_async.call_args
    assert call_args.kwargs["fuzziness"] == 2
    assert call_args.kwargs["prefix_length"] == 0
    assert call_args.kwargs["max_expansions"] == 200
    assert call_args.kwargs["tie_breaker"] == 0.7
    assert call_args.kwargs["jaccard_n"] == 3
    mock_store1._metadata_search_async.assert_not_called()


@pytest.mark.asyncio
async def test_run_async_with_invalid_mode():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, metadata_fields=["category"])

    with pytest.raises(ValueError, match="mode must be either 'strict' or 'fuzzy'"):
        await retriever.run_async(query="test", mode="invalid")


@pytest.mark.asyncio
async def test_run_async_with_fuzzy_parameters():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search_async = AsyncMock(return_value=[{"category": "Python"}])
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, metadata_fields=["category"])

    result = await retriever.run_async(
        query="Python", fuzziness=1, prefix_length=2, max_expansions=100, tie_breaker=0.5, jaccard_n=5, mode="fuzzy"
    )

    assert result == {"metadata": [{"category": "Python"}]}
    call_args = mock_store._metadata_search_async.call_args
    assert call_args.kwargs["fuzziness"] == 1
    assert call_args.kwargs["prefix_length"] == 2
    assert call_args.kwargs["max_expansions"] == 100
    assert call_args.kwargs["tie_breaker"] == 0.5
    assert call_args.kwargs["jaccard_n"] == 5
    assert call_args.kwargs["mode"] == "fuzzy"


@pytest.mark.asyncio
async def test_run_async_with_failure_raises():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search_async = AsyncMock(side_effect=Exception("Search failed"))
    retriever = OpenSearchMetadataRetriever(
        document_store=mock_store, metadata_fields=["category"], raise_on_failure=True
    )

    with pytest.raises(Exception, match="Search failed"):
        await retriever.run_async(query="test")


@pytest.mark.asyncio
async def test_run_async_with_failure_no_raise():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search_async = AsyncMock(side_effect=Exception("Search failed"))
    retriever = OpenSearchMetadataRetriever(
        document_store=mock_store, metadata_fields=["category"], raise_on_failure=False
    )

    result = await retriever.run_async(query="test")

    assert result == {"metadata": []}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_retriever_async_integration(document_store: OpenSearchDocumentStore):
    """Async integration test for OpenSearchMetadataRetriever."""
    docs = [
        Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status"],
        top_k=10,
    )

    result = await retriever.run_async(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0
    assert all(isinstance(row, dict) for row in result["metadata"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_retriever_async_with_filters(document_store: OpenSearchDocumentStore):
    """Async integration test for OpenSearchMetadataRetriever with filters."""
    docs = [
        Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Doc 2", meta={"category": "Python", "status": "inactive", "priority": 2}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status"],
        top_k=10,
    )

    filters = {"field": "priority", "operator": "==", "value": 1}
    result = await retriever.run_async(query="Python", filters=filters)

    assert "metadata" in result
    assert isinstance(result["metadata"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_retriever_async_strict_mode(document_store: OpenSearchDocumentStore):
    """Async integration test for OpenSearchMetadataRetriever in strict mode."""
    docs = [
        Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "Python", "status": "inactive", "priority": 3}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status"],
        top_k=10,
        mode="strict",
    )

    result = await retriever.run_async(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0
    assert all(isinstance(row, dict) for row in result["metadata"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_retriever_async_fuzzy_parameters(document_store: OpenSearchDocumentStore):
    """Async integration test for OpenSearchMetadataRetriever with custom fuzzy parameters."""
    docs = [
        Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "Python", "status": "inactive", "priority": 3}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status"],
        top_k=10,
        mode="fuzzy",
        fuzziness="AUTO",
        prefix_length=2,
        max_expansions=100,
    )

    result = await retriever.run_async(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0
    assert all(isinstance(row, dict) for row in result["metadata"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_retriever_async_comma_separated_query(document_store: OpenSearchDocumentStore):
    """Async integration test for OpenSearchMetadataRetriever with comma-separated query."""
    docs = [
        Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Doc 2", meta={"category": "Java", "status": "active", "priority": 2}),
        Document(content="Doc 3", meta={"category": "Python", "status": "inactive", "priority": 3}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status"],
        top_k=10,
    )

    result = await retriever.run_async(query="Python, active")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0
    assert all(isinstance(row, dict) for row in result["metadata"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_retriever_async_top_k(document_store: OpenSearchDocumentStore):
    """Async integration test for OpenSearchMetadataRetriever with top_k parameter."""
    docs = [Document(content=f"Doc {i}", meta={"category": "Python", "index": i}) for i in range(15)]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category"],
        top_k=5,
    )

    result = await retriever.run_async(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) <= 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_retriever_async_empty_fields(document_store: OpenSearchDocumentStore):
    """Async integration test for OpenSearchMetadataRetriever with empty fields."""
    docs = [
        Document(content="Doc 1", meta={"category": "Python"}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category"],
        top_k=10,
    )

    result = await retriever.run_async(query="Python", metadata_fields=[])

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_retriever_async_deduplication(document_store: OpenSearchDocumentStore):
    """Async integration test for OpenSearchMetadataRetriever deduplication."""
    docs = [
        Document(content="Doc 1", meta={"category": "Python", "status": "active"}),
        Document(content="Doc 2", meta={"category": "Python", "status": "active"}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status"],
        top_k=10,
    )

    result = await retriever.run_async(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    # Check for deduplication - same metadata should appear only once
    seen = []
    for row in result["metadata"]:
        row_tuple = tuple(sorted(row.items()))
        assert row_tuple not in seen, "Duplicate metadata found"
        seen.append(row_tuple)
