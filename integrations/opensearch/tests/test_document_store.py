# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
)
from opensearchpy.exceptions import RequestError, TransportError

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.opensearch.document_store import DEFAULT_MAX_CHUNK_BYTES
from tests.test_document_store_common import OpenSearchDocumentStoreTestMixin


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some hosts")
    res = document_store.to_dict()
    assert res == {
        "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
        "init_parameters": {
            "embedding_dim": 768,
            "hosts": "some hosts",
            "index": "default",
            "mappings": {
                "dynamic_templates": [{"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}],
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                },
            },
            "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
            "method": None,
            "settings": {"index.knn": True},
            "return_embedding": False,
            "create_index": True,
            "http_auth": [
                {"type": "env_var", "env_vars": ["OPENSEARCH_USERNAME"], "strict": False},
                {"type": "env_var", "env_vars": ["OPENSEARCH_PASSWORD"], "strict": False},
            ],
            "use_ssl": None,
            "verify_certs": None,
            "timeout": None,
            "nested_fields": None,
        },
    }


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict_with_http_auth_str(_mock_opensearch_client):
    """
    Verify that plain strings secrets are not serialized.
    """
    document_store = OpenSearchDocumentStore(hosts="some hosts", http_auth=("admin", "admin"))
    res = document_store.to_dict()
    assert res["init_parameters"]["http_auth"] is None


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    data = {
        "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "index": "default",
            "max_chunk_bytes": 1000,
            "embedding_dim": 1536,
            "create_index": False,
            "return_embedding": True,
            "aws_service": "es",
            "http_auth": ("admin", "admin"),
            "use_ssl": True,
            "verify_certs": True,
            "timeout": 60,
        },
    }
    document_store = OpenSearchDocumentStore.from_dict(data)
    assert document_store._hosts == "some hosts"
    assert document_store._index == "default"
    assert document_store._max_chunk_bytes == 1000
    assert document_store._embedding_dim == 1536
    assert document_store._method is None
    assert document_store._mappings == {
        "properties": {
            "embedding": {"type": "knn_vector", "index": True, "dimension": 1536},
            "content": {"type": "text"},
        },
        "dynamic_templates": [
            {
                "strings": {
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"},
                }
            }
        ],
    }
    assert document_store._settings == {"index.knn": True}
    assert document_store._return_embedding is True
    assert document_store._create_index is False
    assert document_store._http_auth == ("admin", "admin")
    assert document_store._use_ssl is True
    assert document_store._verify_certs is True
    assert document_store._timeout == 60


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict_with_http_auth_str(_mock_opensearch_client):
    """
    Verify that serialized plain strings secrets can be properly deserialized.
    """
    data = {
        "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "index": "default",
            "http_auth": ("admin", "admin"),
        },
    }
    document_store = OpenSearchDocumentStore.from_dict(data)
    assert document_store._http_auth == ("admin", "admin")


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_resolve_http_auth_with_secrets(_mock_opensearch_client, monkeypatch):
    monkeypatch.setenv("OPENSEARCH_USERNAME", "admin")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "secret")
    store = OpenSearchDocumentStore(hosts="testhost")
    assert store._resolve_http_auth() == ["admin", "secret"]


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_resolve_http_auth_with_no_secrets(_mock_opensearch_client, monkeypatch):
    monkeypatch.delenv("OPENSEARCH_USERNAME", raising=False)
    monkeypatch.delenv("OPENSEARCH_PASSWORD", raising=False)
    store = OpenSearchDocumentStore(hosts="testhost")
    assert store._resolve_http_auth() is None


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_resolve_http_auth_with_partial_secrets(_mock_opensearch_client, monkeypatch):
    monkeypatch.setenv("OPENSEARCH_USERNAME", "admin")
    monkeypatch.delenv("OPENSEARCH_PASSWORD", raising=False)
    store = OpenSearchDocumentStore(hosts="testhost")
    with pytest.raises(DocumentStoreError, match="http_auth requires both username and password"):
        store._resolve_http_auth()


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_resolve_http_auth_with_plain_strings(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost", http_auth=("admin", "admin"))
    assert store._resolve_http_auth() == ("admin", "admin")


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_init_is_lazy(_mock_opensearch_client):
    OpenSearchDocumentStore(hosts="testhost")
    _mock_opensearch_client.assert_not_called()


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_get_default_mappings(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost", embedding_dim=1536, method={"name": "hnsw"})
    assert store._mappings["properties"]["embedding"] == {
        "type": "knn_vector",
        "index": True,
        "dimension": 1536,
        "method": {"name": "hnsw"},
    }


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
def test_routing_extracted_from_metadata(mock_bulk, _mock_opensearch_client):
    """Test routing extraction from document metadata"""
    mock_bulk.return_value = (2, [])

    store = OpenSearchDocumentStore(hosts="testhost", http_auth=("admin", "admin"))

    docs = [
        Document(id="1", content="Doc", meta={"_routing": "user_a", "other": "data"}),
        Document(id="2", content="Doc"),
    ]
    store.write_documents(docs)

    actions = list(mock_bulk.call_args.kwargs["actions"])

    # Routing should be at action level, not in _source
    assert actions[0]["_routing"] == "user_a"
    assert "_routing" not in actions[0]["_source"].get("meta", {})

    # Other metadata should be preserved
    assert actions[0]["_source"]["other"] == "data"

    # Second doc has no routing
    assert "_routing" not in actions[1]
    assert "_routing" not in actions[1]["_source"].get("meta", {})


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
def test_routing_in_delete(mock_bulk, _mock_opensearch_client):
    """Test routing parameter in delete operations"""
    mock_bulk.return_value = (2, [])

    store = OpenSearchDocumentStore(hosts="testhost", http_auth=("admin", "admin"))

    routing_map = {"1": "user_a", "2": "user_b"}
    store.delete_documents(["1", "2", "3"], routing=routing_map)

    actions = list(mock_bulk.call_args.kwargs["actions"])

    assert actions[0]["_routing"] == "user_a"
    assert actions[1]["_routing"] == "user_b"
    assert "_routing" not in actions[2]


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_bm25_retrieval_retries_with_fuzziness_zero_on_too_many_clauses(_mock_opensearch_client, caplog):
    store = OpenSearchDocumentStore(hosts="testhost")
    store._client = MagicMock()

    too_many_clauses_error = TransportError(
        500, "search_phase_execution_exception", "too_many_clauses: maxClauseCount is set to 1024"
    )
    store._client.search.side_effect = [
        too_many_clauses_error,
        {"hits": {"hits": []}},
    ]

    results = store._bm25_retrieval("a very long query", fuzziness="AUTO")

    assert results == []
    assert store._client.search.call_count == 2
    # Verify the retry used fuzziness=0
    second_call_body = store._client.search.call_args_list[1].kwargs["body"]
    assert second_call_body["query"]["bool"]["must"][0]["multi_match"]["fuzziness"] == 0
    assert "Retrying with fuzziness=0" in caplog.text


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_bm25_retrieval_no_retry_when_fuzziness_already_zero(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost")
    store._client = MagicMock()

    too_many_clauses_error = TransportError(
        500, "search_phase_execution_exception", "too_many_clauses: maxClauseCount is set to 1024"
    )
    store._client.search.side_effect = too_many_clauses_error

    with pytest.raises(TransportError):
        store._bm25_retrieval("a very long query", fuzziness=0)

    assert store._client.search.call_count == 1


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_bm25_retrieval_no_retry_with_custom_query(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost")
    store._client = MagicMock()

    too_many_clauses_error = TransportError(
        500, "search_phase_execution_exception", "too_many_clauses: maxClauseCount is set to 1024"
    )
    store._client.search.side_effect = too_many_clauses_error

    custom_query = {"query": {"match": {"content": "$query"}}}
    with pytest.raises(TransportError):
        store._bm25_retrieval("a very long query", fuzziness="AUTO", custom_query=custom_query)

    assert store._client.search.call_count == 1


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_bm25_retrieval_reraises_other_transport_errors(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost")
    store._client = MagicMock()

    other_error = TransportError(500, "parsing_exception", {"error": {"reason": "some other error"}})
    store._client.search.side_effect = other_error

    with pytest.raises(TransportError):
        store._bm25_retrieval("some query", fuzziness="AUTO")

    assert store._client.search.call_count == 1


@pytest.mark.asyncio
@patch("haystack_integrations.document_stores.opensearch.document_store.AsyncOpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
async def test_bm25_retrieval_async_retries_with_fuzziness_zero_on_too_many_clauses(
    _mock_opensearch_client, _mock_async_client, caplog
):
    store = OpenSearchDocumentStore(hosts="testhost")
    store._async_client = AsyncMock()

    too_many_clauses_error = TransportError(
        500, "search_phase_execution_exception", "too_many_clauses: maxClauseCount is set to 1024"
    )
    store._async_client.search.side_effect = [
        too_many_clauses_error,
        {"hits": {"hits": []}},
    ]

    results = await store._bm25_retrieval_async("a very long query", fuzziness="AUTO")

    assert results == []
    assert store._async_client.search.call_count == 2
    second_call_body = store._async_client.search.call_args_list[1].kwargs["body"]
    assert second_call_body["query"]["bool"]["must"][0]["multi_match"]["fuzziness"] == 0
    assert "Retrying with fuzziness=0" in caplog.text


@pytest.mark.asyncio
@patch("haystack_integrations.document_stores.opensearch.document_store.AsyncOpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
async def test_bm25_retrieval_async_no_retry_when_fuzziness_already_zero(_mock_opensearch_client, _mock_async_client):
    store = OpenSearchDocumentStore(hosts="testhost")
    store._async_client = AsyncMock()

    too_many_clauses_error = TransportError(
        500, "search_phase_execution_exception", "too_many_clauses: maxClauseCount is set to 1024"
    )
    store._async_client.search.side_effect = too_many_clauses_error

    with pytest.raises(TransportError):
        await store._bm25_retrieval_async("a very long query", fuzziness=0)

    assert store._async_client.search.call_count == 1


@pytest.mark.asyncio
@patch("haystack_integrations.document_stores.opensearch.document_store.AsyncOpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
async def test_bm25_retrieval_async_no_retry_with_custom_query(_mock_opensearch_client, _mock_async_client):
    store = OpenSearchDocumentStore(hosts="testhost")
    store._async_client = AsyncMock()

    too_many_clauses_error = TransportError(
        500, "search_phase_execution_exception", "too_many_clauses: maxClauseCount is set to 1024"
    )
    store._async_client.search.side_effect = too_many_clauses_error

    custom_query = {"query": {"match": {"content": "$query"}}}
    with pytest.raises(TransportError):
        await store._bm25_retrieval_async("a very long query", fuzziness="AUTO", custom_query=custom_query)

    assert store._async_client.search.call_count == 1


@pytest.mark.asyncio
@patch("haystack_integrations.document_stores.opensearch.document_store.AsyncOpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
async def test_bm25_retrieval_async_reraises_other_transport_errors(_mock_opensearch_client, _mock_async_client):
    store = OpenSearchDocumentStore(hosts="testhost")
    store._async_client = AsyncMock()

    other_error = TransportError(500, "parsing_exception", {"error": {"reason": "some other error"}})
    store._async_client.search.side_effect = other_error

    with pytest.raises(TransportError):
        await store._bm25_retrieval_async("some query", fuzziness="AUTO")

    assert store._async_client.search.call_count == 1


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict_with_nested_fields(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some hosts", nested_fields=["attributes", "tags"])
    res = document_store.to_dict()
    assert res["init_parameters"]["nested_fields"] == ["attributes", "tags"]
    props = res["init_parameters"]["mappings"]["properties"]
    assert props["attributes"] == {"type": "nested"}
    assert props["tags"] == {"type": "nested"}


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict_with_nested_fields(_mock_opensearch_client):
    data = {
        "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "index": "default",
            "nested_fields": ["attributes"],
            "http_auth": ("admin", "admin"),
        },
    }
    document_store = OpenSearchDocumentStore.from_dict(data)
    assert document_store._nested_fields == ["attributes"]
    assert document_store._resolved_nested_fields == {"attributes"}


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_detect_nested_fields_from_documents(_mock_opensearch_client):
    docs = [
        Document(
            content="doc1",
            meta={
                "attributes": [{"color": "red"}, {"color": "blue"}],
                "tags": [{"name": "sale"}],
                "status": "active",
                "scores": [1, 2, 3],
            },
        ),
        Document(content="doc2", meta={"other_nested": [{"key": "val"}]}),
        Document(content="doc3"),
    ]
    result = OpenSearchDocumentStore._detect_nested_fields_from_documents(docs)
    assert result == {"attributes", "tags", "other_nested"}


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_detect_nested_fields_empty_list(_mock_opensearch_client):
    docs = [Document(content="doc1", meta={"items": []})]
    result = OpenSearchDocumentStore._detect_nested_fields_from_documents(docs)
    assert result == set()


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_extract_nested_fields_from_mapping(_mock_opensearch_client):
    mapping_properties = {
        "attributes": {"type": "nested"},
        "tags": {"type": "nested"},
        "content": {"type": "text"},
        "embedding": {"type": "knn_vector", "dimension": 768},
        "status": {"type": "keyword"},
    }
    result = OpenSearchDocumentStore._extract_nested_fields_from_mapping(mapping_properties)
    assert result == {"attributes", "tags"}


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_extract_nested_fields_from_mapping_none_nested(_mock_opensearch_client):
    mapping_properties = {
        "content": {"type": "text"},
        "status": {"type": "keyword"},
    }
    result = OpenSearchDocumentStore._extract_nested_fields_from_mapping(mapping_properties)
    assert result == set()


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_get_default_mappings_with_nested_fields(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost", nested_fields=["attributes", "tags"])
    props = store._mappings["properties"]
    assert props["attributes"] == {"type": "nested"}
    assert props["tags"] == {"type": "nested"}
    assert props["content"] == {"type": "text"}
    assert "embedding" in props


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_get_default_mappings_with_wildcard(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost", nested_fields="*")
    props = store._mappings["properties"]
    for val in props.values():
        assert val.get("type") != "nested"


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_populate_nested_fields_from_mapping(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost", nested_fields=["attributes"])
    assert store._resolved_nested_fields == {"attributes"}

    mapping_properties = {
        "attributes": {"type": "nested"},
        "tags": {"type": "nested"},
        "content": {"type": "text"},
    }
    store._populate_nested_fields_from_mapping(mapping_properties)
    assert store._resolved_nested_fields == {"attributes", "tags"}


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_resolved_nested_fields_with_wildcard(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost", nested_fields="*")
    assert store._resolved_nested_fields == set()
    assert store._nested_fields == "*"


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
def test_wildcard_nested_fields_detected_on_write(mock_bulk, _mock_opensearch_client):
    mock_bulk.return_value = (2, [])
    store = OpenSearchDocumentStore(hosts="testhost", nested_fields="*")
    store._client = MagicMock()
    store._initialized = True

    docs = [
        Document(
            content="doc1",
            meta={
                "refs": [{"law": "bgb", "section": "1"}],
                "tags": [{"name": "important"}],
                "status": "active",
            },
        ),
        Document(content="doc2", meta={"refs": [{"law": "stgb"}]}),
    ]

    store.write_documents(docs)

    assert "refs" in store._resolved_nested_fields
    assert "tags" in store._resolved_nested_fields
    assert "status" not in store._resolved_nested_fields
    # put_mapping should have been called for each detected nested field
    assert store._client.indices.put_mapping.call_count == 2


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
def test_wildcard_nested_fields_incremental_detection(mock_bulk, _mock_opensearch_client):
    mock_bulk.return_value = (1, [])
    store = OpenSearchDocumentStore(hosts="testhost", nested_fields="*")
    store._client = MagicMock()
    store._initialized = True

    # First batch: detects "refs"
    store.write_documents([Document(content="d1", meta={"refs": [{"law": "bgb"}]})])
    assert store._resolved_nested_fields == {"refs"}
    assert store._client.indices.put_mapping.call_count == 1

    # Second batch: detects "tags" (refs already known)
    store.write_documents([Document(content="d2", meta={"refs": [{"law": "stgb"}], "tags": [{"name": "x"}]})])
    assert store._resolved_nested_fields == {"refs", "tags"}
    assert store._client.indices.put_mapping.call_count == 2  # only one additional call for "tags"


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
@patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
def test_explicit_nested_fields_no_detection_on_write(mock_bulk, _mock_opensearch_client):
    mock_bulk.return_value = (1, [])
    store = OpenSearchDocumentStore(hosts="testhost", nested_fields=["refs"])
    store._client = MagicMock()
    store._initialized = True

    store.write_documents([Document(content="d1", meta={"tags": [{"name": "x"}]})])

    # "tags" should NOT be added because we used explicit nested_fields, not wildcard
    assert "tags" not in store._resolved_nested_fields
    assert store._resolved_nested_fields == {"refs"}
    store._client.indices.put_mapping.assert_not_called()


@pytest.mark.integration
class TestDocumentStore(
    OpenSearchDocumentStoreTestMixin,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    """
    Common test cases will be provided by `DocumentStoreBaseExtendedTests` but you can add more to this class.
    """

    @pytest.fixture
    def document_store(self, document_store):
        """Override base class fixture to provide OpenSearch document store."""
        yield document_store

    def test_write_documents(self, document_store: OpenSearchDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_documents_readonly(self, document_store_readonly: OpenSearchDocumentStore):
        docs = [Document(id="1")]
        with pytest.raises(DocumentStoreError, match="index_not_found_exception"):
            document_store_readonly.write_documents(docs)

    def test_create_index(self, document_store_readonly: OpenSearchDocumentStore):
        document_store_readonly.create_index()
        assert document_store_readonly._client.indices.exists(index=document_store_readonly._index)

    def test_bm25_retrieval(self, document_store: OpenSearchDocumentStore, test_documents: list[Document]):
        document_store.write_documents(test_documents)
        res = document_store._bm25_retrieval("functional", top_k=3)

        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_bm25_retrieval_pagination(self, document_store: OpenSearchDocumentStore, test_documents: list[Document]):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """
        document_store.write_documents(test_documents)
        res = document_store._bm25_retrieval("programming", top_k=11)

        assert len(res) == 11
        assert all("programming" in doc.content for doc in res)

    def test_bm25_retrieval_all_terms_must_match(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)
        res = document_store._bm25_retrieval("functional Haskell", top_k=3, all_terms_must_match=True)

        assert len(res) == 1
        assert "Haskell is a functional programming language" in res[0].content

    def test_bm25_retrieval_all_terms_must_match_false(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)
        res = document_store._bm25_retrieval("functional Haskell", top_k=10, all_terms_must_match=False)

        assert len(res) == 5
        assert all("functional" in doc.content for doc in res)

    def test_bm25_retrieval_with_fuzziness(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)

        query_with_typo = "functinal"
        # Query without fuzziness to search for the exact match
        res = document_store._bm25_retrieval(query_with_typo, top_k=3, fuzziness="0")
        # Nothing is found as the query contains a typo
        assert res == []

        # Query with fuzziness with the same query
        res = document_store._bm25_retrieval(query_with_typo, top_k=3, fuzziness="1")
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_bm25_retrieval_with_fuzziness_overflow(self, document_store: OpenSearchDocumentStore, caplog):
        """
        Test that a long query with fuzziness="AUTO" that exceeds OpenSearch's maxClauseCount
        is automatically retried with fuzziness=0 instead of raising an error.
        """
        # Build an index vocabulary of similar 5-character words. With fuzziness="AUTO",
        # 5-char words get edit distance 1, so each query term fuzzy-matches many similar
        # indexed terms, causing clause expansion beyond the default maxClauseCount (1024).
        # With fuzziness=0, each term produces exactly 1 clause, staying well under the limit.
        words = [f"foo{chr(97 + i)}{chr(97 + j)}" for i in range(20) for j in range(26)]  # 520 words

        chunk_size = 52
        docs = [
            Document(content=" ".join(words[i : i + chunk_size]), id=str(idx))
            for idx, i in enumerate(range(0, len(words), chunk_size))
        ]
        document_store.write_documents(docs)

        # Query with a subset of words. With fuzziness="AUTO", each 5-char term expands
        # to match ~45 similar indexed terms, pushing total clauses well above 1024.
        long_query = " ".join(words[:100])

        # This should not raise: the too_many_clauses error is caught and retried with fuzziness=0
        res = document_store._bm25_retrieval(long_query, top_k=3, fuzziness="AUTO")
        assert isinstance(res, list)
        assert "Retrying with fuzziness=0" in caplog.text

    def test_bm25_retrieval_with_filters(self, document_store: OpenSearchDocumentStore, test_documents: list[Document]):
        document_store.write_documents(test_documents)
        res = document_store._bm25_retrieval(
            "programming",
            top_k=10,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )
        assert len(res) == 5
        retrieved_ids = sorted([doc.id for doc in res])
        assert retrieved_ids == ["1", "2", "3", "4", "5"]

    def test_bm25_retrieval_with_custom_query(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)

        custom_query = {
            "query": {
                "function_score": {
                    "query": {"bool": {"must": {"match": {"content": "$query"}}, "filter": "$filters"}},
                    "field_value_factor": {"field": "likes", "factor": 0.1, "modifier": "log1p", "missing": 0},
                }
            }
        }

        res = document_store._bm25_retrieval(
            "functional",
            top_k=3,
            custom_query=custom_query,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )
        assert len(res) == 3
        assert "1" == res[0].id
        assert "2" == res[1].id
        assert "3" == res[2].id

    def test_bm25_retrieval_with_custom_query_empty_filters(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)

        custom_query = {
            "query": {
                "function_score": {
                    "query": {"bool": {"must": {"match": {"content": "$query"}}, "filter": "$filters"}},
                    "field_value_factor": {"field": "likes", "factor": 0.1, "modifier": "log1p", "missing": 0},
                }
            }
        }

        res = document_store._bm25_retrieval(
            "functional",
            top_k=3,
            custom_query=custom_query,
        )
        assert len(res) == 3
        assert "1" == res[0].id
        assert "2" == res[1].id
        assert "3" == res[2].id

    def test_embedding_retrieval(self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)
        results = document_store_embedding_dim_4_no_emb_returned._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

    def test_embedding_retrieval_with_filters(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}
        # we set top_k=3, to make the test pass as we are not sure whether efficient filtering is supported for nmslib
        # TODO: remove top_k=3, when efficient filtering is supported for nmslib
        results = document_store_embedding_dim_4_no_emb_returned._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3, filters=filters
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    def test_embedding_retrieval_with_filters_efficient_filtering(
        self, document_store_embedding_dim_4_no_emb_returned_faiss: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4_no_emb_returned_faiss.write_documents(docs)

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}
        results = document_store_embedding_dim_4_no_emb_returned_faiss._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1],
            filters=filters,
            efficient_filtering=True,
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    def test_embedding_retrieval_pagination(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """

        docs = [
            Document(content=f"Document {i}", embedding=[random.random() for _ in range(4)])  # noqa: S311
            for i in range(20)
        ]

        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)
        results = document_store_embedding_dim_4_no_emb_returned._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=11, filters={}
        )
        assert len(results) == 11

    def test_embedding_retrieval_with_custom_query(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        custom_query = {
            "query": {
                "bool": {"must": [{"knn": {"embedding": {"vector": "$query_embedding", "k": 3}}}], "filter": "$filters"}
            }
        }

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}
        results = document_store_embedding_dim_4_no_emb_returned._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, filters=filters, custom_query=custom_query
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    def test_embedding_retrieval_query_documents_different_embedding_sizes(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        """
        Test that the retrieval fails if the query embedding and the documents have different embedding sizes.
        """
        docs = [Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4])]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        with pytest.raises(RequestError):
            document_store_embedding_dim_4_no_emb_returned._embedding_retrieval(query_embedding=[0.1, 0.1])

    def test_write_documents_different_embedding_sizes_fail(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        """
        Test that write_documents fails if the documents have different embedding sizes.
        """
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Hello world", embedding=[0.1, 0.2]),
        ]

        with pytest.raises(DocumentStoreError):
            document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

    @patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
    def test_write_documents_with_badly_formatted_bulk_errors(self, mock_bulk, document_store):
        error = {"some_key": "some_value"}
        mock_bulk.return_value = ([], [error])

        with pytest.raises(DocumentStoreError) as e:
            document_store.write_documents([Document(content="Hello world")])
            e.match(f"{error}")

    @patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
    def test_write_documents_max_chunk_bytes(self, mock_bulk, document_store):
        mock_bulk.return_value = (1, [])
        document_store.write_documents([Document(content="Hello world")])

        assert mock_bulk.call_args.kwargs["max_chunk_bytes"] == DEFAULT_MAX_CHUNK_BYTES

    def test_embedding_retrieval_but_dont_return_embeddings_for_embedding_retrieval(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)
        results = document_store_embedding_dim_4_no_emb_returned._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].embedding is None

    def test_embedding_retrieval_but_dont_return_embeddings_for_bm25_retrieval(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)
        results = document_store_embedding_dim_4_no_emb_returned._bm25_retrieval("document", top_k=2)
        assert len(results) == 2
        assert results[0].embedding is None

    def test_filter_documents_no_embedding_returned(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)
        results = document_store_embedding_dim_4_no_emb_returned.filter_documents()

        assert len(results) == 3
        assert results[0].embedding is None
        assert results[1].embedding is None
        assert results[2].embedding is None

    def test_delete_all_documents_index_recreation(self, document_store: OpenSearchDocumentStore):
        # populate the index with some documents
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        document_store.write_documents(docs)

        # capture index structure before deletion
        assert document_store._client is not None
        index_info_before = document_store._client.indices.get(index=document_store._index)
        mappings_before = index_info_before[document_store._index]["mappings"]
        settings_before = index_info_before[document_store._index]["settings"]

        # delete all documents
        document_store.delete_all_documents(recreate_index=True)
        assert document_store.count_documents() == 0

        # verify index structure is preserved
        index_info_after = document_store._client.indices.get(index=document_store._index)
        mappings_after = index_info_after[document_store._index]["mappings"]
        settings_after = index_info_after[document_store._index]["settings"]

        assert mappings_after == mappings_before, "delete_all_documents should preserve index mappings"

        settings_after["index"].pop("uuid", None)
        settings_after["index"].pop("creation_date", None)
        settings_before["index"].pop("uuid", None)
        settings_before["index"].pop("creation_date", None)
        assert settings_after == settings_before, "delete_all_documents should preserve index settings"

        new_doc = Document(id="4", content="New document after delete all")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

        results = document_store.filter_documents()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"

    def test_get_metadata_field_unique_values(self, document_store: OpenSearchDocumentStore):
        # Test with string values
        docs = [
            Document(content="Python programming", meta={"category": "A", "language": "Python"}),
            Document(content="Java programming", meta={"category": "B", "language": "Java"}),
            Document(content="Python scripting", meta={"category": "A", "language": "Python"}),
            Document(content="JavaScript development", meta={"category": "C", "language": "JavaScript"}),
            Document(content="Python data science", meta={"category": "A", "language": "Python"}),
            Document(content="Java backend", meta={"category": "B", "language": "Java"}),
        ]
        document_store.write_documents(docs)

        # Test getting all unique values without search term
        unique_values, after_key = document_store.get_metadata_field_unique_values("meta.category", None, 10)
        assert set(unique_values) == {"A", "B", "C"}
        # after_key should be None when all results are returned
        assert after_key is None

        # Test with "meta." prefix
        unique_languages, _ = document_store.get_metadata_field_unique_values("meta.language", None, 10)
        assert set(unique_languages) == {"Python", "Java", "JavaScript"}

        # Test pagination - first page
        unique_values_page1, after_key_page1 = document_store.get_metadata_field_unique_values("meta.category", None, 2)
        assert len(unique_values_page1) == 2
        assert all(val in ["A", "B", "C"] for val in unique_values_page1)
        # Should have an after_key for pagination
        assert after_key_page1 is not None

        # Test pagination - second page using after_key
        unique_values_page2, after_key_page2 = document_store.get_metadata_field_unique_values(
            "meta.category", None, 2, after=after_key_page1
        )
        assert len(unique_values_page2) == 1
        assert unique_values_page2[0] in ["A", "B", "C"]
        # Should have no more results
        assert after_key_page2 is None

        # Test with search term - filter by content matching "Python"
        unique_values_filtered, _ = document_store.get_metadata_field_unique_values("meta.category", "Python", 10)
        assert set(unique_values_filtered) == {"A"}  # Only category A has documents with "Python" in content

        # Test with search term - filter by content matching "Java"
        unique_values_java, _ = document_store.get_metadata_field_unique_values("meta.category", "Java", 10)
        assert set(unique_values_java) == {"B"}  # Only category B has documents with "Java" in content

        # Test with integer values
        int_docs = [
            Document(content="Doc 1", meta={"priority": 1}),
            Document(content="Doc 2", meta={"priority": 2}),
            Document(content="Doc 3", meta={"priority": 1}),
            Document(content="Doc 4", meta={"priority": 3}),
        ]
        document_store.write_documents(int_docs)
        unique_priorities, _ = document_store.get_metadata_field_unique_values("meta.priority", None, 10)
        assert set(unique_priorities) == {"1", "2", "3"}

        # Test with search term on integer field
        unique_priorities_filtered, _ = document_store.get_metadata_field_unique_values("meta.priority", "Doc 1", 10)
        assert set(unique_priorities_filtered) == {"1"}

    def test_write_with_routing(self, document_store: OpenSearchDocumentStore):
        """Test writing documents with routing metadata"""
        docs = [
            Document(id="1", content="User A doc", meta={"_routing": "user_a", "category": "test"}),
            Document(id="2", content="User B doc", meta={"_routing": "user_b"}),
            Document(id="3", content="No routing"),
        ]

        written = document_store.write_documents(docs)
        assert written == 3
        assert document_store.count_documents() == 3

        # Verify _routing not stored in metadata
        retrieved = document_store.filter_documents()
        retrieved_by_id = {doc.id: doc for doc in retrieved}

        # Check _routing is not stored for any document
        for doc in retrieved:
            assert "_routing" not in doc.meta

        assert retrieved_by_id["1"].meta["category"] == "test"

        assert retrieved_by_id["2"].meta == {}

        assert retrieved_by_id["3"].meta == {}

    def test_delete_with_routing(self, document_store: OpenSearchDocumentStore):
        """Test deleting documents with routing"""
        docs = [
            Document(id="1", content="Doc 1", meta={"_routing": "user_a"}),
            Document(id="2", content="Doc 2", meta={"_routing": "user_b"}),
            Document(id="3", content="Doc 3"),
        ]
        document_store.write_documents(docs)

        routing_map = {"1": "user_a", "2": "user_b"}
        document_store.delete_documents(["1", "2"], routing=routing_map)

        assert document_store.count_documents() == 1

    def test_metadata_search_fuzzy_mode(self, document_store: OpenSearchDocumentStore):
        """Test metadata search in fuzzy mode."""
        docs = [
            Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "Python", "status": "inactive", "priority": 3}),
            Document(
                content="JavaScript development", meta={"category": "JavaScript", "status": "active", "priority": 1}
            ),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search for "Python" in category field
        result = document_store._metadata_search(
            query="Python",
            fields=["category"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) >= 2  # At least 2 documents with category "Python"
        assert all(isinstance(row, dict) for row in result)
        assert all("category" in row for row in result)
        # Verify all results contain "Python" in category (fuzzy match might include variations)
        categories = [row.get("category", "").lower() for row in result]
        assert any("python" in cat for cat in categories)

    def test_metadata_search_strict_mode(self, document_store: OpenSearchDocumentStore):
        """Test metadata search in strict mode."""
        docs = [
            Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "Python", "status": "inactive", "priority": 3}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search for "Python" in category field with strict mode
        result = document_store._metadata_search(
            query="Python",
            fields=["category"],
            mode="strict",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) == 1  # At least 1 document with category "Python" due to metadat deduplication
        assert all(isinstance(row, dict) for row in result)
        assert all("category" in row for row in result)

    def test_metadata_search_multiple_fields(self, document_store: OpenSearchDocumentStore):
        """Test metadata search across multiple fields."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Java", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "Python", "status": "inactive", "priority": 3}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search for "active" across both category and status fields
        result = document_store._metadata_search(
            query="active",
            fields=["category", "status"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(row, dict) for row in result)
        # Results should only contain the specified fields
        for row in result:
            assert all(key in ["category", "status"] for key in row.keys())

    def test_metadata_search_comma_separated_query(self, document_store: OpenSearchDocumentStore):
        """Test metadata search with comma-separated query parts."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Java", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "Python", "status": "inactive", "priority": 3}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search for "Python, active" - should match documents with both
        result = document_store._metadata_search(
            query="Python, active",
            fields=["category", "status"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(row, dict) for row in result)

    def test_metadata_search_top_k(self, document_store: OpenSearchDocumentStore):
        """Test metadata search respects top_k parameter."""
        docs = [Document(content=f"Doc {i}", meta={"category": "Python", "index": i}) for i in range(15)]
        document_store.write_documents(docs, refresh=True)

        # Request top 5 results
        result = document_store._metadata_search(
            query="Python",
            fields=["category"],
            mode="fuzzy",
            top_k=5,
        )

        assert isinstance(result, list)
        assert len(result) <= 5

    def test_metadata_search_with_filters(self, document_store: OpenSearchDocumentStore):
        """Test metadata search with additional filters."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Python", "status": "inactive", "priority": 2}),
            Document(content="Doc 3", meta={"category": "Java", "status": "active", "priority": 1}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search with filter for priority == 1
        filters = {"field": "priority", "operator": "==", "value": 1}
        result = document_store._metadata_search(
            query="Python",
            fields=["category"],
            mode="fuzzy",
            top_k=10,
            filters=filters,
        )

        assert isinstance(result, list)
        # Should only return documents with priority == 1
        assert len(result) >= 1

    def test_metadata_search_empty_fields(self, document_store: OpenSearchDocumentStore):
        """Test metadata search with empty fields list returns empty result."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python"}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = document_store._metadata_search(
            query="Python",
            fields=[],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) == 0

    def test_metadata_search_deduplication(self, document_store: OpenSearchDocumentStore):
        """Test that metadata search deduplicates results."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active"}),
            Document(content="Doc 2", meta={"category": "Python", "status": "active"}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = document_store._metadata_search(
            query="Python",
            fields=["category", "status"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        # Check for deduplication - same metadata should appear only once
        seen = []
        for row in result:
            row_tuple = tuple(sorted(row.items()))
            assert row_tuple not in seen, "Duplicate metadata found"
            seen.append(row_tuple)

    def test_query_sql(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
            Document(content="JavaScript development", meta={"category": "C", "status": "active", "priority": 1}),
        ]
        document_store.write_documents(docs, refresh=True)

        # SQL query returns raw JSON response from OpenSearch SQL API
        sql_query = (
            f"SELECT content, category, status, priority FROM {document_store._index} "  # noqa: S608
            f"WHERE category = 'A' ORDER BY priority"
        )
        result = document_store._query_sql(sql_query)

        # Verify raw JSON response structure
        assert isinstance(result, dict)
        assert "schema" in result
        assert "datarows" in result
        assert "size" in result
        assert "status" in result
        assert [entry["name"] for entry in result["schema"]] == ["content", "category", "status", "priority"]
        assert len(result["datarows"]) == 2  # Two documents with category A

        categories = [row[1] for row in result["datarows"]]
        assert all(category == "A" for category in categories)

        # error handling for invalid SQL query
        invalid_query = "SELECT * FROM non_existent_index"
        with pytest.raises(DocumentStoreError, match="Failed to execute SQL query"):
            document_store._query_sql(invalid_query)

    def test_explicit_nested_fields_filter(self, document_store_nested: OpenSearchDocumentStore):
        """Filtering on explicitly declared nested fields returns correct documents."""
        docs = [
            Document(
                content="doc about bgb 1a",
                meta={"refs": [{"law": "bgb", "section": "1", "paragraph": "a"}], "status": "active"},
            ),
            Document(
                content="doc about bgb 2",
                meta={"refs": [{"law": "bgb", "section": "2"}], "status": "active"},
            ),
            Document(
                content="doc about stgb",
                meta={"refs": [{"law": "stgb", "section": "1"}], "status": "active"},
            ),
        ]
        document_store_nested.write_documents(docs)

        # Filter for refs.law == bgb
        results = document_store_nested.filter_documents(
            filters={"field": "meta.refs.law", "operator": "==", "value": "bgb"}
        )
        assert len(results) == 2
        assert all("bgb" in str(doc.meta["refs"]) for doc in results)

    def test_explicit_nested_fields_combined_filter(self, document_store_nested: OpenSearchDocumentStore):
        """AND filter across sub-fields of the same nested path matches within the same array element."""
        docs = [
            Document(
                content="bgb section 1",
                meta={"refs": [{"law": "bgb", "section": "1"}, {"law": "stgb", "section": "2"}]},
            ),
            Document(
                content="bgb section 2",
                meta={"refs": [{"law": "bgb", "section": "2"}]},
            ),
            Document(
                content="stgb section 1",
                meta={"refs": [{"law": "stgb", "section": "1"}]},
            ),
        ]
        document_store_nested.write_documents(docs)

        # Filter: refs.law == bgb AND refs.section == 1 (must match within same nested object)
        results = document_store_nested.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.refs.law", "operator": "==", "value": "bgb"},
                    {"field": "meta.refs.section", "operator": "==", "value": "1"},
                ],
            }
        )
        assert len(results) == 1
        assert results[0].content == "bgb section 1"

    def test_explicit_nested_fields_mixed_nested_and_flat(self, document_store_nested: OpenSearchDocumentStore):
        """Filtering on both nested and flat fields works correctly."""
        docs = [
            Document(content="d1", meta={"refs": [{"law": "bgb"}], "status": "active"}),
            Document(content="d2", meta={"refs": [{"law": "bgb"}], "status": "inactive"}),
            Document(content="d3", meta={"refs": [{"law": "stgb"}], "status": "active"}),
        ]
        document_store_nested.write_documents(docs)

        results = document_store_nested.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.refs.law", "operator": "==", "value": "bgb"},
                    {"field": "meta.status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert len(results) == 1
        assert results[0].content == "d1"

    def test_wildcard_nested_fields_auto_detection(self, document_store_wildcard_nested: OpenSearchDocumentStore):
        """With nested_fields=['*'], writing docs with list[dict] metadata auto-detects and maps nested fields."""
        assert document_store_wildcard_nested._resolved_nested_fields == set()

        docs = [
            Document(content="d1", meta={"refs": [{"law": "bgb", "section": "1"}], "status": "active"}),
            Document(content="d2", meta={"refs": [{"law": "stgb"}], "tags": [{"name": "important"}]}),
        ]
        document_store_wildcard_nested.write_documents(docs)

        # After writing, nested fields should be detected
        assert "refs" in document_store_wildcard_nested._resolved_nested_fields
        assert "tags" in document_store_wildcard_nested._resolved_nested_fields
        assert "status" not in document_store_wildcard_nested._resolved_nested_fields

    def test_wildcard_nested_fields_filter(self, document_store_wildcard_nested: OpenSearchDocumentStore):
        """With wildcard auto-detection, nested filtering works correctly."""
        docs = [
            Document(content="bgb section 1", meta={"refs": [{"law": "bgb", "section": "1"}]}),
            Document(content="bgb section 2", meta={"refs": [{"law": "bgb", "section": "2"}]}),
            Document(content="stgb section 1", meta={"refs": [{"law": "stgb", "section": "1"}]}),
        ]
        document_store_wildcard_nested.write_documents(docs)

        # Nested AND filter on refs.law + refs.section
        results = document_store_wildcard_nested.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.refs.law", "operator": "==", "value": "bgb"},
                    {"field": "meta.refs.section", "operator": "==", "value": "1"},
                ],
            }
        )
        assert len(results) == 1
        assert results[0].content == "bgb section 1"

    def test_wildcard_nested_fields_incremental_detection(
        self, document_store_wildcard_nested: OpenSearchDocumentStore
    ):
        """A second write batch discovers new nested fields not seen in the first batch."""
        # First batch: only refs
        document_store_wildcard_nested.write_documents([Document(content="d1", meta={"refs": [{"law": "bgb"}]})])
        assert "refs" in document_store_wildcard_nested._resolved_nested_fields
        assert "tags" not in document_store_wildcard_nested._resolved_nested_fields

        # Second batch: introduces tags
        document_store_wildcard_nested.write_documents([Document(content="d2", meta={"tags": [{"name": "sale"}]})])
        assert "tags" in document_store_wildcard_nested._resolved_nested_fields

        # Both nested fields are now filterable
        results = document_store_wildcard_nested.filter_documents(
            filters={"field": "meta.refs.law", "operator": "==", "value": "bgb"}
        )
        assert len(results) == 1
        assert results[0].content == "d1"

        results = document_store_wildcard_nested.filter_documents(
            filters={"field": "meta.tags.name", "operator": "==", "value": "sale"}
        )
        assert len(results) == 1
        assert results[0].content == "d2"

    def test_nested_fields_or_filter(self, document_store_nested: OpenSearchDocumentStore):
        """OR filter on nested sub-fields works correctly."""
        docs = [
            Document(content="bgb doc", meta={"refs": [{"law": "bgb"}]}),
            Document(content="stgb doc", meta={"refs": [{"law": "stgb"}]}),
            Document(content="other doc", meta={"refs": [{"law": "zpo"}]}),
        ]
        document_store_nested.write_documents(docs)

        results = document_store_nested.filter_documents(
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "meta.refs.law", "operator": "==", "value": "bgb"},
                    {"field": "meta.refs.law", "operator": "==", "value": "stgb"},
                ],
            }
        )
        assert len(results) == 2
        contents = sorted([r.content for r in results])
        assert contents == ["bgb doc", "stgb doc"]

    def test_nested_fields_not_filter(self, document_store_nested: OpenSearchDocumentStore):
        """NOT filter on nested sub-fields excludes matching documents."""
        docs = [
            Document(
                content="bgb section 1",
                meta={"refs": [{"law": "bgb", "section": "1"}]},
            ),
            Document(
                content="bgb section 2",
                meta={"refs": [{"law": "bgb", "section": "2"}]},
            ),
            Document(
                content="stgb section 1",
                meta={"refs": [{"law": "stgb", "section": "1"}]},
            ),
        ]
        document_store_nested.write_documents(docs)

        # NOT (refs.law == bgb AND refs.section == 1) — only the first doc has both in the same nested object
        results = document_store_nested.filter_documents(
            filters={
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.refs.law", "operator": "==", "value": "bgb"},
                    {"field": "meta.refs.section", "operator": "==", "value": "1"},
                ],
            }
        )
        assert len(results) == 2
        contents = sorted([r.content for r in results])
        assert contents == ["bgb section 2", "stgb section 1"]

    def test_nested_fields_different_paths_filter(self, document_store_nested: OpenSearchDocumentStore):
        """AND filter across different nested paths works correctly."""
        docs = [
            Document(
                content="both",
                meta={"refs": [{"law": "bgb"}], "tags": [{"name": "important"}]},
            ),
            Document(
                content="refs only",
                meta={"refs": [{"law": "bgb"}], "tags": [{"name": "unimportant"}]},
            ),
            Document(
                content="tags only",
                meta={"refs": [{"law": "stgb"}], "tags": [{"name": "important"}]},
            ),
        ]
        document_store_nested.write_documents(docs)

        results = document_store_nested.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.refs.law", "operator": "==", "value": "bgb"},
                    {"field": "meta.tags.name", "operator": "==", "value": "important"},
                ],
            }
        )
        assert len(results) == 1
        assert results[0].content == "both"
