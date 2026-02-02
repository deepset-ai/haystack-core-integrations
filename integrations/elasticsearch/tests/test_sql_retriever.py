# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchSQLRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSQLRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._raise_on_failure is True


def test_init_custom():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, raise_on_failure=False, fetch_size=50)
    assert retriever._raise_on_failure is False
    assert retriever._fetch_size == 50


def test_init_invalid_document_store():
    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        ElasticsearchSQLRetriever(document_store="not a document store")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host")
    retriever = ElasticsearchSQLRetriever(document_store=document_store, fetch_size=100)
    res = retriever.to_dict()
    assert (
        res["type"]
        == "haystack_integrations.components.retrievers.elasticsearch.sql_retriever.ElasticsearchSQLRetriever"
    )
    assert res["init_parameters"]["raise_on_failure"] is True
    assert res["init_parameters"]["fetch_size"] == 100
    assert "document_store" in res["init_parameters"]


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host")
    retriever = ElasticsearchSQLRetriever(document_store=document_store)
    data = retriever.to_dict()
    retriever_from_dict = ElasticsearchSQLRetriever.from_dict(data)
    assert retriever_from_dict._raise_on_failure is True


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.return_value = {
        "columns": [{"name": "content", "type": "text"}],
        "rows": [["Test content"]],
    }
    retriever = ElasticsearchSQLRetriever(document_store=mock_store)
    result = retriever.run(query="SELECT content FROM test_index")

    mock_store._query_sql.assert_called_once_with(query="SELECT content FROM test_index", fetch_size=None)
    assert "result" in result
    assert result["result"]["columns"][0]["name"] == "content"
    assert result["result"]["rows"][0][0] == "Test content"


def test_run_with_fetch_size():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.return_value = {"columns": [], "rows": []}
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, fetch_size=10)
    retriever.run(query="SELECT * FROM test_index")

    mock_store._query_sql.assert_called_once_with(query="SELECT * FROM test_index", fetch_size=10)


def test_run_with_runtime_fetch_size():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.return_value = {"columns": [], "rows": []}
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, fetch_size=10)
    retriever.run(query="SELECT * FROM test_index", fetch_size=20)

    mock_store._query_sql.assert_called_once_with(query="SELECT * FROM test_index", fetch_size=20)


def test_run_invalid_runtime_document_store():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchSQLRetriever(document_store=mock_store)

    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        retriever.run(query="SELECT * FROM test_index", document_store="invalid")


def test_run_error_handling_raise():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.side_effect = Exception("SQL error")
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, raise_on_failure=True)

    with pytest.raises(Exception, match="SQL error"):
        retriever.run(query="SELECT * FROM test_index")


def test_run_error_handling_no_raise():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._query_sql.side_effect = Exception("SQL error")
    retriever = ElasticsearchSQLRetriever(document_store=mock_store, raise_on_failure=False)

    result = retriever.run(query="SELECT * FROM test_index")
    assert result["result"] == {}


def _row_to_dict(columns: list[dict], row: list) -> dict:
    """Map a row array to a dict keyed by column name (for assertions)."""
    return {col["name"]: row[i] for i, col in enumerate(columns)}


@pytest.mark.integration
def test_sql_retriever_basic_query_columns_rows_format(document_store: ElasticsearchDocumentStore):
    """Test regular SELECT query - verifies raw JSON response with columns/rows structure"""
    docs = [
        Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
        Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
        Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
        Document(content="JavaScript development", meta={"category": "C", "status": "active", "priority": 1}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = ElasticsearchSQLRetriever(document_store=document_store)
    sql_query = (
        f'SELECT content, category, status, priority FROM "{document_store._index}" '  # noqa: S608
        f"WHERE category = 'A' ORDER BY priority"
    )
    result = retriever.run(query=sql_query)

    assert "result" in result
    response = result["result"]
    assert isinstance(response, dict)

    # Verify raw Elasticsearch SQL JSON response structure (columns/rows)
    assert "columns" in response
    assert "rows" in response

    columns = response["columns"]
    rows = response["rows"]
    column_names = [col["name"] for col in columns]
    assert "content" in column_names
    assert "category" in column_names
    assert "status" in column_names
    assert "priority" in column_names

    assert len(rows) == 2
    for row in rows:
        row_dict = _row_to_dict(columns, row)
        assert row_dict["category"] == "A"


@pytest.mark.integration
def test_sql_retriever_count_query_aggregations_format(document_store: ElasticsearchDocumentStore):
    """Test aggregate query (COUNT) - verifies raw JSON response with columns/rows structure"""
    docs = [
        Document(content="Doc 1", meta={"category": "A"}),
        Document(content="Doc 2", meta={"category": "B"}),
        Document(content="Doc 3", meta={"category": "A"}),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = ElasticsearchSQLRetriever(document_store=document_store)
    count_query = f'SELECT COUNT(*) as total FROM "{document_store._index}"'  # noqa: S608
    result = retriever.run(query=count_query)

    assert "result" in result
    response = result["result"]
    assert isinstance(response, dict)

    assert "columns" in response
    assert "rows" in response
    rows = response["rows"]
    assert len(rows) == 1
    assert rows[0][0] == 3


@pytest.mark.integration
def test_sql_retriever_metadata_extraction(document_store: ElasticsearchDocumentStore):
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

    retriever = ElasticsearchSQLRetriever(document_store=document_store)

    sql_query = (
        f'SELECT author, year, rating FROM "{document_store._index}" '  # noqa: S608
        f"WHERE year >= 2023 ORDER BY rating DESC"
    )
    result = retriever.run(query=sql_query)

    assert "result" in result
    response = result["result"]
    assert isinstance(response, dict)

    assert "columns" in response
    assert "rows" in response
    columns = response["columns"]
    rows = response["rows"]
    assert len(rows) == 2

    column_names = [col["name"] for col in columns]
    assert "author" in column_names
    assert "year" in column_names
    assert "rating" in column_names
    assert "content" not in column_names

    authors = []
    for row in rows:
        row_dict = _row_to_dict(columns, row)
        assert row_dict["year"] >= 2023
        authors.append(row_dict["author"])

    assert "Jane Smith" in authors
    assert "John Doe" in authors
    assert rows[0][column_names.index("rating")] >= rows[1][column_names.index("rating")]


@pytest.mark.integration
def test_sql_retriever_runtime_document_store_switching(
    document_store: ElasticsearchDocumentStore, document_store_2: ElasticsearchDocumentStore
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

    retriever = ElasticsearchSQLRetriever(document_store=document_store)

    sql_query1 = f"SELECT content, category FROM \"{document_store._index}\" WHERE category = 'A'"  # noqa: S608
    result1 = retriever.run(query=sql_query1)
    assert len(result1["result"]["rows"]) == 1
    content_col_idx = next(i for i, c in enumerate(result1["result"]["columns"]) if c["name"] == "content")
    assert "Python" in result1["result"]["rows"][0][content_col_idx]

    sql_query2 = f"SELECT content, category FROM \"{document_store_2._index}\" WHERE category = 'C'"  # noqa: S608
    result2 = retriever.run(query=sql_query2, document_store=document_store_2)
    assert len(result2["result"]["rows"]) == 1
    assert "JavaScript" in result2["result"]["rows"][0][content_col_idx]

    assert result1["result"]["rows"][0][content_col_idx] != result2["result"]["rows"][0][content_col_idx]


@pytest.mark.integration
def test_sql_retriever_error_handling(document_store: ElasticsearchDocumentStore):
    """Test error handling for invalid SQL queries"""
    retriever = ElasticsearchSQLRetriever(document_store=document_store, raise_on_failure=True)

    invalid_query = "SELECT * FROM non_existent_index"
    with pytest.raises(Exception, match="Failed to execute SQL query"):
        retriever.run(query=invalid_query)

    retriever_no_raise = ElasticsearchSQLRetriever(document_store=document_store, raise_on_failure=False)
    result = retriever_no_raise.run(query=invalid_query)
    assert result["result"] == {}


@pytest.mark.integration
def test_sql_retriever_with_fetch_size(document_store: ElasticsearchDocumentStore):
    """Test SQL retriever with fetch_size parameter"""
    docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(15)]
    document_store.write_documents(docs, refresh=True)

    retriever = ElasticsearchSQLRetriever(document_store=document_store, fetch_size=5)
    sql_query = (
        f'SELECT content, category, "index" FROM "{document_store._index}" '  # noqa: S608
        f"WHERE category = 'A' ORDER BY \"index\""
    )

    result = retriever.run(query=sql_query)
    assert "result" in result
    assert isinstance(result["result"], dict)
    assert "columns" in result["result"]
    assert "rows" in result["result"]
    assert len(result["result"]["rows"]) > 0

    result2 = retriever.run(query=sql_query, fetch_size=10)
    assert "result" in result2
    assert isinstance(result2["result"], dict)
    assert "rows" in result2["result"]
    assert len(result2["result"]["rows"]) > 0
