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
    """Test regular SELECT query - verifies raw response"""
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

    # Verify hits structure
    assert "total" in response
    assert "size" in response
    assert "status" in response

    # Verify the schema contains all the selected fields
    assert all(
        field in [entry["name"] for entry in response["schema"]]
        for field in ["content", "category", "status", "priority"]
    )
    # Verify datarows contain the expected number of rows and columns
    assert len(response["datarows"]) == 2


@pytest.mark.integration
def test_sql_retriever_count_query_aggregations_format(document_store: OpenSearchDocumentStore):
    """Test aggregate query (COUNT) - verifies tabular SQL response format."""
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

    # Verify tabular SQL response structure
    assert "schema" in response
    assert "datarows" in response
    assert "size" in response
    assert "status" in response

    # Verify COUNT schema and value
    assert len(response["schema"]) == 1
    assert response["schema"][0]["name"] == "COUNT(*)"
    assert response["schema"][0].get("alias") == "total"
    assert response["schema"][0]["type"] == "long"

    assert len(response["datarows"]) == 1
    assert response["datarows"][0] == [3]


@pytest.mark.integration
def test_sql_retriever_metadata_extraction(document_store: OpenSearchDocumentStore):
    """Test extracting metadata fields - verifies tabular SQL response structure."""
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

    # Verify tabular SQL response structure
    assert "schema" in response
    assert "datarows" in response
    assert "size" in response
    assert "status" in response

    # Verify schema contains only selected metadata fields in query order
    assert [entry["name"] for entry in response["schema"]] == ["author", "year", "rating"]

    # Verify returned rows
    rows = response["datarows"]
    assert len(rows) == 2

    authors = [row[0] for row in rows]
    years = [row[1] for row in rows]
    ratings = [row[2] for row in rows]

    assert "Jane Smith" in authors
    assert "John Doe" in authors
    assert all(year >= 2023 for year in years)

    # Verify ordering by rating DESC
    assert ratings[0] >= ratings[1]


@pytest.mark.integration
def test_sql_retriever_runtime_document_store_switching(
    document_store: OpenSearchDocumentStore, document_store_2: OpenSearchDocumentStore
):
    """Test switching document stores at runtime with tabular SQL responses."""
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
    response1 = result1["result"]
    assert [entry["name"] for entry in response1["schema"]] == ["content", "category"]
    assert len(response1["datarows"]) == 1
    assert "Python" in response1["datarows"][0][0]

    # Query second store at runtime
    sql_query2 = f"SELECT content, category FROM {document_store_2._index} WHERE category = 'C'"  # noqa: S608
    result2 = retriever.run(query=sql_query2, document_store=document_store_2)
    response2 = result2["result"]
    assert [entry["name"] for entry in response2["schema"]] == ["content", "category"]
    assert len(response2["datarows"]) == 1
    assert "JavaScript" in response2["datarows"][0][0]

    # Verify results are different
    assert response1["datarows"][0][0] != response2["datarows"][0][0]


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
    assert result["result"] == {}


@pytest.mark.integration
def test_sql_retriever_with_fetch_size(document_store: OpenSearchDocumentStore):
    """Test SQL retriever with fetch_size parameter and tabular response format."""
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
    response = result["result"]
    assert "schema" in response
    assert "datarows" in response
    assert "size" in response
    assert "status" in response
    assert [entry["name"] for entry in response["schema"]] == ["content", "category", "index"]
    assert len(response["datarows"]) > 0
    assert len(response["datarows"]) <= 5
    assert response.get("cursor") is not None

    # Test with runtime fetch_size override
    result2 = retriever.run(query=sql_query, fetch_size=10)
    assert "result" in result2
    assert isinstance(result2["result"], dict)
    response2 = result2["result"]
    assert "schema" in response2
    assert "datarows" in response2
    assert "size" in response2
    assert "status" in response2
    assert [entry["name"] for entry in response2["schema"]] == ["content", "category", "index"]
    assert len(response2["datarows"]) > 0
    assert len(response2["datarows"]) <= 10


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_basic_query(document_store: OpenSearchDocumentStore):
    """Test basic async SQL query execution with tabular SQL response."""
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
    assert "schema" in response
    assert "datarows" in response
    assert "size" in response
    assert "status" in response
    assert [entry["name"] for entry in response["schema"]] == ["content", "category", "status"]
    assert len(response["datarows"]) == 2

    categories = [row[1] for row in response["datarows"]]
    assert all(category == "A" for category in categories)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_runtime_document_store_switching(
    document_store: OpenSearchDocumentStore, document_store_2: OpenSearchDocumentStore
):
    """Test async switching document stores at runtime with tabular SQL responses."""
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
    response1 = result1["result"]
    assert [entry["name"] for entry in response1["schema"]] == ["content", "category"]
    assert len(response1["datarows"]) == 1
    assert "Python" in response1["datarows"][0][0]

    # Query second store at runtime
    sql_query2 = f"SELECT content, category FROM {document_store_2._index} WHERE category = 'C'"  # noqa: S608
    result2 = await retriever.run_async(query=sql_query2, document_store=document_store_2)
    response2 = result2["result"]
    assert [entry["name"] for entry in response2["schema"]] == ["content", "category"]
    assert len(response2["datarows"]) == 1
    assert "JavaScript" in response2["datarows"][0][0]

    # Verify results are different
    assert response1["datarows"][0][0] != response2["datarows"][0][0]


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
    assert result["result"] == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_retriever_async_with_fetch_size(document_store: OpenSearchDocumentStore):
    """Test async SQL retriever with fetch_size using tabular SQL responses."""
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
    response = result["result"]
    assert "schema" in response
    assert "datarows" in response
    assert "size" in response
    assert "status" in response
    assert [entry["name"] for entry in response["schema"]] == ["content", "category", "index"]
    assert len(response["datarows"]) > 0
    assert len(response["datarows"]) <= 5
    assert response.get("cursor") is not None

    # Test with runtime fetch_size override
    result2 = await retriever.run_async(query=sql_query, fetch_size=10)
    assert "result" in result2
    assert isinstance(result2["result"], dict)
    response2 = result2["result"]
    assert "schema" in response2
    assert "datarows" in response2
    assert "size" in response2
    assert "status" in response2
    assert [entry["name"] for entry in response2["schema"]] == ["content", "category", "index"]
    assert len(response2["datarows"]) > 0
    assert len(response2["datarows"]) <= 10
