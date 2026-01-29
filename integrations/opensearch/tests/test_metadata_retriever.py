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
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, metadata_fields=["category", "status"])
    assert retriever._document_store == mock_store
    assert retriever._metadata_fields == ["category", "status"]
    assert retriever._top_k == 20
    assert retriever._exact_match_weight == 0.6
    assert retriever._mode == "fuzzy"
    assert retriever._fuzziness == 2
    assert retriever._prefix_length == 0
    assert retriever._max_expansions == 200
    assert retriever._raise_on_failure is True


def test_init_custom():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchMetadataRetriever(
        document_store=mock_store,
        metadata_fields=["category"],
        top_k=10,
        exact_match_weight=0.8,
        mode="strict",
        fuzziness=1,
        prefix_length=2,
        max_expansions=100,
        raise_on_failure=False,
    )
    assert retriever._top_k == 10
    assert retriever._exact_match_weight == 0.8
    assert retriever._mode == "strict"
    assert retriever._fuzziness == 1
    assert retriever._prefix_length == 2
    assert retriever._max_expansions == 100
    assert retriever._raise_on_failure is False


def test_init_invalid_document_store():
    with pytest.raises(ValueError, match="document_store must be an instance of OpenSearchDocumentStore"):
        OpenSearchMetadataRetriever(document_store="not a document store", metadata_fields=["category"])


def test_init_empty_fields():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    with pytest.raises(ValueError, match="fields must be a non-empty list"):
        OpenSearchMetadataRetriever(document_store=mock_store, metadata_fields=[])


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status"],
        top_k=15,
        exact_match_weight=0.7,
        mode="strict",
        fuzziness=1,
        prefix_length=2,
        max_expansions=100,
    )
    res = retriever.to_dict()
    assert (
        res["type"]
        == "haystack_integrations.components.retrievers.opensearch.metadata_retriever.OpenSearchMetadataRetriever"
    )
    assert res["init_parameters"]["metadata_fields"] == ["category", "status"]
    assert res["init_parameters"]["top_k"] == 15
    assert res["init_parameters"]["exact_match_weight"] == 0.7
    assert res["init_parameters"]["mode"] == "strict"
    assert res["init_parameters"]["fuzziness"] == 1
    assert res["init_parameters"]["prefix_length"] == 2
    assert res["init_parameters"]["max_expansions"] == 100
    assert "document_store" in res["init_parameters"]


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some fake host")
    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["category", "status"],
        top_k=10,
        fuzziness="AUTO",
        prefix_length=1,
        max_expansions=50,
    )
    data = retriever.to_dict()
    retriever_from_dict = OpenSearchMetadataRetriever.from_dict(data)
    assert retriever_from_dict._metadata_fields == ["category", "status"]
    assert retriever_from_dict._top_k == 10
    assert retriever_from_dict._mode == "fuzzy"
    assert retriever_from_dict._fuzziness == "AUTO"
    assert retriever_from_dict._prefix_length == 1
    assert retriever_from_dict._max_expansions == 50


def test_run_with_runtime_document_store():
    mock_store1 = Mock(spec=OpenSearchDocumentStore)
    mock_store2 = Mock(spec=OpenSearchDocumentStore)
    mock_store2._metadata_search = Mock(return_value=[{"category": "Java"}])
    retriever = OpenSearchMetadataRetriever(document_store=mock_store1, metadata_fields=["category"])

    result = retriever.run(query="Java", document_store=mock_store2)

    assert result == {"metadata": [{"category": "Java"}]}
    mock_store2._metadata_search.assert_called_once()
    # Check that the call includes the new parameters with default values
    call_args = mock_store2._metadata_search.call_args
    assert call_args.kwargs["fuzziness"] == 2
    assert call_args.kwargs["prefix_length"] == 0
    assert call_args.kwargs["max_expansions"] == 200
    mock_store1._metadata_search.assert_not_called()


def test_run_with_invalid_mode():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, metadata_fields=["category"])

    with pytest.raises(ValueError, match="mode must be either 'strict' or 'fuzzy'"):
        retriever.run(query="test", mode="invalid")


def test_run_with_fuzzy_parameters():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search = Mock(return_value=[{"category": "Python"}])
    retriever = OpenSearchMetadataRetriever(document_store=mock_store, metadata_fields=["category"])

    result = retriever.run(query="Python", fuzziness="AUTO", prefix_length=1, max_expansions=50, mode="fuzzy")

    assert result == {"metadata": [{"category": "Python"}]}
    call_args = mock_store._metadata_search.call_args
    assert call_args.kwargs["fuzziness"] == "AUTO"
    assert call_args.kwargs["prefix_length"] == 1
    assert call_args.kwargs["max_expansions"] == 50
    assert call_args.kwargs["mode"] == "fuzzy"


def test_run_with_failure_raises():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search = Mock(side_effect=Exception("Search failed"))
    retriever = OpenSearchMetadataRetriever(
        document_store=mock_store, metadata_fields=["category"], raise_on_failure=True
    )

    with pytest.raises(Exception, match="Search failed"):
        retriever.run(query="test")


def test_run_with_failure_no_raise():
    mock_store = Mock(spec=OpenSearchDocumentStore)
    mock_store._metadata_search = Mock(side_effect=Exception("Search failed"))
    retriever = OpenSearchMetadataRetriever(
        document_store=mock_store, metadata_fields=["category"], raise_on_failure=False
    )

    result = retriever.run(query="test")

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
        metadata_fields=["category", "status"],
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
        metadata_fields=["category", "status"],
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
        metadata_fields=["category", "status"],
        top_k=10,
        mode="strict",
    )

    result = retriever.run(query="Python")

    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0
    assert all(isinstance(row, dict) for row in result["metadata"])


@pytest.mark.integration
def test_metadata_retriever_fuzzy_parameters(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever with custom fuzzy parameters."""
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
        fuzziness=1,
        prefix_length=1,
        max_expansions=50,
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
        metadata_fields=["category", "status"],
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
        metadata_fields=["category"],
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
        metadata_fields=["category"],
        top_k=10,
    )

    result = retriever.run(query="Python", metadata_fields=[])

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
        metadata_fields=["category", "status"],
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
def test_metadata_retriever_list_metadata(document_store: OpenSearchDocumentStore):
    """Integration test for OpenSearchMetadataRetriever with list metadata fields."""
    # Use only list-of-strings for searchable fields; avoid mixed-type lists (e.g. [str, int, bool])
    # as OpenSearch dynamic mapping does not allow one field to have conflicting types.
    docs = [
        Document(
            content="Doc 1",
            meta={
                "tags": ["python", "programming", "tutorial"],
                "categories": ["tech", "coding"],
                "numbers": [1, 2, 3],
            },
        ),
        Document(
            content="Doc 2",
            meta={
                "tags": ["java", "programming"],
                "categories": ["tech"],
                "numbers": [4, 5, 6],
            },
        ),
        Document(
            content="Doc 3",
            meta={
                "tags": ["python", "advanced"],
                "categories": ["tech", "coding", "advanced"],
                "numbers": [1, 2, 3],
            },
        ),
    ]
    document_store.write_documents(docs, refresh=True)

    retriever = OpenSearchMetadataRetriever(
        document_store=document_store,
        metadata_fields=["tags", "categories"],
        top_k=10,
    )

    # Search for values that appear in list-of-string fields (query "python" matches tags)
    result = retriever.run(query="python")
    assert "metadata" in result
    assert isinstance(result["metadata"], list)
    assert len(result["metadata"]) > 0

    # Verify list fields are returned correctly
    for entry in result["metadata"]:
        if "tags" in entry:
            assert isinstance(entry["tags"], list)
        if "categories" in entry:
            assert isinstance(entry["categories"], list)
        if "numbers" in entry:
            assert isinstance(entry["numbers"], list)
