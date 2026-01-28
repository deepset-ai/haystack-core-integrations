# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.opensearch import OpenSearchSQLRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchSQLRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._raise_on_failure is True


def test_init_custom():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchSQLRetriever(document_store=mock_store, raise_on_failure=False, fetch_size=50)
    assert retriever._raise_on_failure is False
    assert retriever._fetch_size == 50


def test_init_invalid_document_store():
    with pytest.raises(ValueError, match="document_store must be an instance of OpenSearchDocumentStore"):
        OpenSearchSQLRetriever(document_store="not a document store")


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchSQLRetriever(document_store=document_store, fetch_size=100)
    res = retriever.to_dict()
    assert res["type"] == "haystack_integrations.components.retrievers.opensearch.sql_retriever.OpenSearchSQLRetriever"
    assert res["init_parameters"]["raise_on_failure"] is True
    assert res["init_parameters"]["fetch_size"] == 100
    assert "document_store" in res["init_parameters"]


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchSQLRetriever(document_store=document_store)
    data = retriever.to_dict()
    retriever_from_dict = OpenSearchSQLRetriever.from_dict(data)
    assert retriever_from_dict._raise_on_failure is True


@pytest.mark.integration
def test_sql_retriever_basic_query_hits_format(document_store: OpenSearchDocumentStore):
    """Test regular SELECT query - verifies raw JSON response with hits structure"""
    docs = [
        Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
        Document(content="JavaScript development", meta={"category": "C", "status": "active", "priority": 1}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)
    sql_query = (
        f"SELECT content, category, status, priority FROM {document_store._index} "  # noqa: S608
        f"WHERE category = 'A' ORDER BY priority"
    )
    result = retriever.run(query=sql_query)

    assert "result" in result
    response = result["result"]
    assert isinstance(response, dict)

    # Verify raw OpenSearch JSON response structure
    assert "_shards" in response
    assert "hits" in response
    assert "took" in response
    assert "timed_out" in response

    # Verify hits structure
    hits = response["hits"]
    assert "total" in hits
    assert "hits" in hits
    assert len(hits["hits"]) == 2

    # Verify each hit contains _source with selected fields
    for hit in hits["hits"]:
        assert "_source" in hit
        assert "_index" in hit
        source = hit["_source"]
        assert "content" in source
        assert "category" in source
        assert source["category"] == "A"


@pytest.mark.integration
def test_sql_retriever_count_query_aggregations_format(document_store: OpenSearchDocumentStore):
    """Test aggregate query (COUNT) - verifies raw JSON response with aggregations structure"""
    docs = [
        Document(content="Doc 1", meta={"category": "A"}),
        Document(content="Doc 2", meta={"category": "B"}),
        Document(content="Doc 3", meta={"category": "A"}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)
    count_query = f"SELECT COUNT(*) as total FROM {document_store._index}"  # noqa: S608
    result = retriever.run(query=count_query)

    assert "result" in result
    response = result["result"]
    assert isinstance(response, dict)

    # Verify raw OpenSearch JSON response structure
    assert "_shards" in response
    assert "aggregations" in response
    assert "hits" in response
    assert "took" in response
    assert "timed_out" in response

    # Verify aggregations structure
    aggregations = response["aggregations"]
    assert "total" in aggregations
    assert aggregations["total"]["value"] == 3

    # Verify hits structure (should be empty for aggregate queries)
    hits = response["hits"]
    assert "total" in hits
    assert len(hits.get("hits", [])) == 0


@pytest.mark.integration
def test_sql_retriever_metadata_extraction(document_store: OpenSearchDocumentStore):
    """Test extracting metadata fields - verifies raw JSON response structure"""
    docs = [
        Document(
            content="Python tutorial",
            meta={"author": "John Doe", "year": 2023, "tags": ["programming", "python"], "rating": 4.5},
        ),
        Document(
            content="Java guide",
            meta={"author": "Jane Smith", "year": 2024, "tags": ["programming", "java"], "rating": 4.8},
        ),
        Document(
            content="SQL handbook",
            meta={"author": "Bob Johnson", "year": 2022, "tags": ["database", "sql"], "rating": 4.2},
        ),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)

    # extract only metadata fields
    sql_query = (
        f"SELECT author, year, rating FROM {document_store._index} "  # noqa: S608
        f"WHERE year >= 2023 ORDER BY rating DESC"
    )
    result = retriever.run(query=sql_query)

    assert "result" in result
    response = result["result"]
    assert isinstance(response, dict)

    # Verify raw response structure
    assert "hits" in response
    hits = response["hits"]
    assert len(hits["hits"]) == 2

    # Verify _source contains only selected metadata fields
    authors = []
    for hit in hits["hits"]:
        source = hit["_source"]
        assert "author" in source
        assert "year" in source
        assert "rating" in source
        assert "content" not in source
        assert source["year"] >= 2023
        authors.append(source["author"])

    assert "Jane Smith" in authors
    assert "John Doe" in authors

    # Verify ordering by rating DESC
    assert hits["hits"][0]["_source"]["rating"] >= hits["hits"][1]["_source"]["rating"]


@pytest.mark.integration
def test_sql_retriever_runtime_document_store_switching(
    document_store: OpenSearchDocumentStore, document_store_2: OpenSearchDocumentStore
):
    """Test switching document stores at runtime"""
    docs1 = [
        Document(content="Python programming", meta={"category": "A"}),
        Document(content="Java programming", meta={"category": "B"}),
    ]
    document_store.write_documents(docs1, refresh=True)

    docs2 = [
        Document(content="JavaScript development", meta={"category": "C"}),
        Document(content="TypeScript development", meta={"category": "D"}),
    ]
    document_store_2.write_documents(docs2, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)

    # Query first store
    sql_query1 = f"SELECT content, category FROM {document_store._index} WHERE category = 'A'"  # noqa: S608
    result1 = retriever.run(query=sql_query1)
    assert len(result1["result"]["hits"]["hits"]) == 1
    assert "Python" in result1["result"]["hits"]["hits"][0]["_source"]["content"]

    # Query second store at runtime
    sql_query2 = f"SELECT content, category FROM {document_store_2._index} WHERE category = 'C'"  # noqa: S608
    result2 = retriever.run(query=sql_query2, document_store=document_store_2)
    assert len(result2["result"]["hits"]["hits"]) == 1
    assert "JavaScript" in result2["result"]["hits"]["hits"][0]["_source"]["content"]

    # Verify results are different
    assert (
        result1["result"]["hits"]["hits"][0]["_source"]["content"]
        != result2["result"]["hits"]["hits"][0]["_source"]["content"]
    )


@pytest.mark.integration
def test_sql_retriever_error_handling(document_store: OpenSearchDocumentStore):
    """Test error handling for invalid SQL queries"""
    retriever = OpenSearchSQLRetriever(document_store=document_store, raise_on_failure=True)

    invalid_query = "SELECT * FROM non_existent_index"
    with pytest.raises(Exception, match="Failed to execute SQL query"):
        retriever.run(query=invalid_query)

    # Test with raise_on_failure=False
    retriever_no_raise = OpenSearchSQLRetriever(document_store=document_store, raise_on_failure=False)
    result = retriever_no_raise.run(query=invalid_query)
    assert result["result"] is None


@pytest.mark.integration
def test_sql_retriever_with_fetch_size(document_store: OpenSearchDocumentStore):
    """Test SQL retriever with fetch_size parameter"""
    docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(15)]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store, fetch_size=5)
    sql_query = (
        f"SELECT content, category, index FROM {document_store._index} "  # noqa: S608
        f"WHERE category = 'A' ORDER BY index"
    )

    # Test with fetch_size from initialization
    result = retriever.run(query=sql_query)
    assert "result" in result
    assert isinstance(result["result"], dict)
    assert "hits" in result["result"]
    assert len(result["result"]["hits"]["hits"]) > 0

    # Test with runtime fetch_size override
    result2 = retriever.run(query=sql_query, fetch_size=10)
    assert "result" in result2
    assert isinstance(result2["result"], dict)
    assert "hits" in result2["result"]
    assert len(result2["result"]["hits"]["hits"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_basic_query(document_store: OpenSearchDocumentStore):
    """Test basic async SQL query execution"""
    docs = [
        Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
    ]
    await document_store.write_documents_async(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)
    sql_query = (
        f"SELECT content, category, status FROM {document_store._index} "  # noqa: S608
        f"WHERE category = 'A' ORDER BY priority"
    )
    result = await retriever.run_async(query=sql_query)

    assert "result" in result
    response = result["result"]
    assert isinstance(response, dict)
    assert "hits" in response
    assert len(response["hits"]["hits"]) == 2

    categories = [hit["_source"]["category"] for hit in response["hits"]["hits"]]
    assert all(cat == "A" for cat in categories)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_runtime_document_store_switching(
    document_store: OpenSearchDocumentStore, document_store_2: OpenSearchDocumentStore
):
    """Test async switching document stores at runtime"""
    docs1 = [
        Document(content="Python programming", meta={"category": "A"}),
        Document(content="Java programming", meta={"category": "B"}),
    ]
    await document_store.write_documents_async(docs1, refresh=True)

    docs2 = [
        Document(content="JavaScript development", meta={"category": "C"}),
        Document(content="TypeScript development", meta={"category": "D"}),
    ]
    await document_store_2.write_documents_async(docs2, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store)

    # Query first store
    sql_query1 = f"SELECT content, category FROM {document_store._index} WHERE category = 'A'"  # noqa: S608
    result1 = await retriever.run_async(query=sql_query1)
    assert len(result1["result"]["hits"]["hits"]) == 1
    assert "Python" in result1["result"]["hits"]["hits"][0]["_source"]["content"]

    # Query second store at runtime
    sql_query2 = f"SELECT content, category FROM {document_store_2._index} WHERE category = 'C'"  # noqa: S608
    result2 = await retriever.run_async(query=sql_query2, document_store=document_store_2)
    assert len(result2["result"]["hits"]["hits"]) == 1
    assert "JavaScript" in result2["result"]["hits"]["hits"][0]["_source"]["content"]

    # Verify results are different
    assert (
        result1["result"]["hits"]["hits"][0]["_source"]["content"]
        != result2["result"]["hits"]["hits"][0]["_source"]["content"]
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_error_handling(document_store: OpenSearchDocumentStore):
    """Test async error handling for invalid SQL queries"""
    retriever = OpenSearchSQLRetriever(document_store=document_store, raise_on_failure=True)

    invalid_query = "SELECT * FROM non_existent_index"
    with pytest.raises(Exception, match="Failed to execute SQL query"):
        await retriever.run_async(query=invalid_query)

    # Test with raise_on_failure=False
    retriever_no_raise = OpenSearchSQLRetriever(document_store=document_store, raise_on_failure=False)
    result = await retriever_no_raise.run_async(query=invalid_query)
    assert result["result"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_with_fetch_size(document_store: OpenSearchDocumentStore):
    """Test async SQL retriever with fetch_size parameter"""
    docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(15)]
    await document_store.write_documents_async(docs, refresh=True)

    retriever = OpenSearchSQLRetriever(document_store=document_store, fetch_size=5)
    sql_query = (
        f"SELECT content, category, index FROM {document_store._index} "  # noqa: S608
        f"WHERE category = 'A' ORDER BY index"
    )

    # Test with fetch_size from initialization
    result = await retriever.run_async(query=sql_query)
    assert "result" in result
    assert isinstance(result["result"], dict)
    assert "hits" in result["result"]
    assert len(result["result"]["hits"]["hits"]) > 0

    # Test with runtime fetch_size override
    result2 = await retriever.run_async(query=sql_query, fetch_size=10)
    assert "result" in result2
    assert isinstance(result2["result"], dict)
    assert "hits" in result2["result"]
    assert len(result2["result"]["hits"]["hits"]) > 0
