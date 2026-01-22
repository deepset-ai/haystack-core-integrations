# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.opensearch import OpenSearchMetadataRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, fields=["category", "status"])
    assert retriever._document_store == mock_store
    assert retriever._fields == ["category", "status"]
    assert retriever._top_k == 20
    assert retriever._exact_match_weight == 0.6
    assert retriever._mode == "fuzzy"
    assert retriever._raise_on_failure is True


def test_init_custom():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchMetadataRetriever(
        document_store=mock_store,
        fields=["category"],
        top_k=10,
        exact_match_weight=0.8,
        mode="strict",
        raise_on_failure=False,
    )
    assert retriever._top_k == 10
    assert retriever._exact_match_weight == 0.8
    assert retriever._mode == "strict"
    assert retriever._raise_on_failure is False


def test_init_invalid_document_store():
    with pytest.raises(ValueError, match="document_store must be an instance of OpenSearchDocumentStore"):
        OpenSearchMetadataRetriever(document_store="not a document store", fields=["category"])


def test_init_empty_fields():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    with pytest.raises(ValueError, match="fields must be a non-empty list"):
        OpenSearchMetadataRetriever(document_store=mock_store, fields=[])


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        fields=["category", "status"],
        top_k=15,
        exact_match_weight=0.7,
        mode="strict",
    )
    res = retriever.to_dict()
    assert (
        res["type"]
        == "haystack_integrations.components.retrievers.opensearch.metadata_retriever.OpenSearchMetadataRetriever"
    )
    assert res["init_parameters"]["fields"] == ["category", "status"]
    assert res["init_parameters"]["top_k"] == 15
    assert res["init_parameters"]["exact_match_weight"] == 0.7
    assert res["init_parameters"]["mode"] == "strict"
    assert "document_store" in res["init_parameters"]


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchMetadataRetriever(document_store=document_store, fields=["category", "status"], top_k=10)
    data = retriever.to_dict()
    retriever_from_dict = OpenSearchMetadataRetriever.from_dict(data)
    assert retriever_from_dict._fields == ["category", "status"]
    assert retriever_from_dict._top_k == 10
    assert retriever_from_dict._mode == "fuzzy"


def test_run_with_runtime_document_store():
    mock_store1 = Mock(spec=OpenSearchDocumentStore)
    mock_store2 = Mock(spec=OpenSearchDocumentStore)
    mock_store2._metadata_search = Mock(return_value=[{"category": "Java"}])
    retriever = OpenSearchMetadataRetriever(document_store=mock_store1, fields=["category"])

    result = retriever.run(query="Java", document_store=mock_store2)

    assert result == {"metadata": [{"category": "Java"}]}
    mock_store2._metadata_search.assert_called_once()
    mock_store1._metadata_search.assert_not_called()


@pytest.mark.asyncio
async def test_run_async_with_runtime_document_store():
    mock_store1 = Mock(spec=OpenSearchDocumentStore)
    mock_store2 = Mock(spec=OpenSearchDocumentStore)
    mock_store2._metadata_search_async = Mock(return_value=[{"category": "Java"}])
    retriever = OpenSearchMetadataRetriever(document_store=mock_store1, fields=["category"])

    result = await retriever.run_async(query="Java", document_store=mock_store2)

    assert result == {"metadata": [{"category": "Java"}]}
    mock_store2._metadata_search_async.assert_called_once()
    mock_store1._metadata_search_async.assert_not_called()


def test_run_with_invalid_mode():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, fields=["category"])

    with pytest.raises(ValueError, match="mode must be either 'strict' or 'fuzzy'"):
        retriever.run(query="test", mode="invalid")


@pytest.mark.asyncio
async def test_run_async_with_invalid_mode():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, fields=["category"])

    with pytest.raises(ValueError, match="mode must be either 'strict' or 'fuzzy'"):
        await retriever.run_async(query="test", mode="invalid")


def test_run_with_failure_raises():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search = Mock(side_effect=Exception("Search failed"))
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, fields=["category"], raise_on_failure=True)

    with pytest.raises(Exception, match="Search failed"):
        retriever.run(query="test")


def test_run_with_failure_no_raise():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search = Mock(side_effect=Exception("Search failed"))
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, fields=["category"], raise_on_failure=False)

    result = retriever.run(query="test")

    assert result == {"metadata": []}


@pytest.mark.asyncio
async def test_run_async_with_failure_raises():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search_async = Mock(side_effect=Exception("Search failed"))
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, fields=["category"], raise_on_failure=True)

    with pytest.raises(Exception, match="Search failed"):
        await retriever.run_async(query="test")


@pytest.mark.asyncio
async def test_run_async_with_failure_no_raise():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search_async = Mock(side_effect=Exception("Search failed"))
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, fields=["category"], raise_on_failure=False)

    result = await retriever.run_async(query="test")

    assert result == {"metadata": []}


@pytest.mark.integration
def test_metadata_retriever_integration(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever."""
    docs = [
        Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "Python", "status": "inactive", "priority": 3}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        fields=["category", "status"],
        top_k=10,
    )

    result = retriever.run(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0
    assert all(isinstance(row, dict) for row in result["metadata"])
    # Results should only contain the specified fields
    for row in result["metadata"]:
        assert all(key in ["category", "status"] for key in row.keys())


@pytest.mark.integration
def test_metadata_retriever_with_filters(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever with filters."""
    docs = [
        Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Doc 2", meta={"category": "Python", "status": "inactive", "priority": 2}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        fields=["category", "status"],
        top_k=10,
    )

    filters = {"field": "priority", "operator": "==", "value": 1}
    result = retriever.run(query="Python", filters=filters)

    assert "metadata" in result
    assert isinstance(result["metadata"], list)


@pytest.mark.integration
def test_metadata_retriever_strict_mode(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever in strict mode."""
    docs = [
        Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "Python", "status": "inactive", "priority": 3}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        fields=["category", "status"],
        top_k=10,
        mode="strict",
    )

    result = retriever.run(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0
    assert all(isinstance(row, dict) for row in result["metadata"])


@pytest.mark.integration
def test_metadata_retriever_comma_separated_query(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever with comma-separated query."""
    docs = [
        Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
        Document(content="Doc 2", meta={"category": "Java", "status": "active", "priority": 2}),
        Document(content="Doc 3", meta={"category": "Python", "status": "inactive", "priority": 3}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        fields=["category", "status"],
        top_k=10,
    )

    result = retriever.run(query="Python, active")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0
    assert all(isinstance(row, dict) for row in result["metadata"])


@pytest.mark.integration
def test_metadata_retriever_top_k(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever with top_k parameter."""
    docs = [Document(content=f"Doc {i}", meta={"category": "Python", "index": i}) for i in range(15)]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        fields=["category"],
        top_k=5,
    )

    result = retriever.run(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) <= 5


@pytest.mark.integration
def test_metadata_retriever_empty_fields(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever with empty fields."""
    docs = [
        Document(content="Doc 1", meta={"category": "Python"}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        fields=["category"],
        top_k=10,
    )

    result = retriever.run(query="Python", fields=[])

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) == 0


@pytest.mark.integration
def test_metadata_retriever_deduplication(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever deduplication."""
    docs = [
        Document(content="Doc 1", meta={"category": "Python", "status": "active"}),
        Document(content="Doc 2", meta={"category": "Python", "status": "active"}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        fields=["category", "status"],
        top_k=10,
    )

    result = retriever.run(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    # Check for deduplication - same metadata should appear only once
    seen = []
    for row in result["metadata"]:
        row_tuple = tuple(sorted(row.items()))
        assert row_tuple not in seen, "Duplicate metadata found"
        seen.append(row_tuple)


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
        fields=["category", "status"],
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
        fields=["category", "status"],
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
        fields=["category", "status"],
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
        fields=["category", "status"],
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
        fields=["category"],
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
        fields=["category"],
        top_k=10,
    )

    result = await retriever.run_async(query="Python", fields=[])

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
        fields=["category", "status"],
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
