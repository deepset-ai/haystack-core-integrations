# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import patch

import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseExtendedTests
from opensearchpy.exceptions import RequestError

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.opensearch.document_store import DEFAULT_MAX_CHUNK_BYTES


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
            "http_auth": None,
            "use_ssl": None,
            "verify_certs": None,
            "timeout": None,
        },
    }


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


@patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
def test_routing_extracted_from_metadata(mock_bulk, document_store):
    """Test routing extraction from document metadata"""
    mock_bulk.return_value = (2, [])

    docs = [
        Document(id="1", content="Doc", meta={"_routing": "user_a", "other": "data"}),
        Document(id="2", content="Doc"),
    ]
    document_store.write_documents(docs)

    actions = list(mock_bulk.call_args.kwargs["actions"])

    # Routing should be at action level, not in _source
    assert actions[0]["_routing"] == "user_a"
    assert "_routing" not in actions[0]["_source"].get("meta", {})

    # Other metadata should be preserved
    assert actions[0]["_source"]["other"] == "data"

    # Second doc has no routing
    assert "_routing" not in actions[1]
    assert "_routing" not in actions[1]["_source"].get("meta", {})


@patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
def test_routing_in_delete(mock_bulk, document_store):
    """Test routing parameter in delete operations"""
    mock_bulk.return_value = (2, [])

    routing_map = {"1": "user_a", "2": "user_b"}
    document_store.delete_documents(["1", "2", "3"], routing=routing_map)

    actions = list(mock_bulk.call_args.kwargs["actions"])

    assert actions[0]["_routing"] == "user_a"
    assert actions[1]["_routing"] == "user_b"
    assert "_routing" not in actions[2]


@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseExtendedTests):
    """
    Common test cases will be provided by `DocumentStoreBaseExtendedTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self, document_store):
        """Override base class fixture to provide OpenSearch document store."""
        yield document_store

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        The OpenSearchDocumentStore.filter_documents() method returns documents with their score set.

        We don't want to compare the score, so we set it to None before comparing.

        Embeddings are not exactly the same when retrieved from OpenSearch (float round-trip),
        so we compare them approximately and then set both to None for the final equality check.
        """
        assert len(received) == len(expected)
        received = sorted(received, key=lambda x: x.id)
        expected = sorted(expected, key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            received_doc.score = None
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)
            received_doc.embedding, expected_doc.embedding = None, None
            assert received_doc == expected_doc

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

    def test_count_documents_by_filter(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        count_a = document_store.count_documents_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count_a == 3

        count_a_active = document_store.count_documents_by_filter(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert count_a_active == 2

    def test_count_unique_metadata_by_filter(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 5

        # Count distinct values for all documents
        distinct_counts = document_store.count_unique_metadata_by_filter(
            filters={}, metadata_fields=["category", "status", "priority"]
        )
        assert distinct_counts["category"] == 3  # A, B, C
        assert distinct_counts["status"] == 2  # active, inactive
        assert distinct_counts["priority"] == 3  # 1, 2, 3

        # Count distinct values for documents with category="A"
        distinct_counts_a = document_store.count_unique_metadata_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_a["category"] == 1  # Only A
        assert distinct_counts_a["status"] == 2  # active, inactive
        assert distinct_counts_a["priority"] == 2  # 1, 3

        # Count distinct values for documents with status="active"
        distinct_counts_active = document_store.count_unique_metadata_by_filter(
            filters={"field": "meta.status", "operator": "==", "value": "active"},
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_active["category"] == 3  # A, B, C
        assert distinct_counts_active["status"] == 1  # Only active
        assert distinct_counts_active["priority"] == 3  # 1, 2, 3

        # Count distinct values with complex filter (category="A" AND status="active")
        distinct_counts_a_active = document_store.count_unique_metadata_by_filter(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.status", "operator": "==", "value": "active"},
                ],
            },
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_a_active["category"] == 1  # Only A
        assert distinct_counts_a_active["status"] == 1  # Only active
        assert distinct_counts_a_active["priority"] == 2  # 1, 3

        # Test with only a subset of fields
        distinct_counts_subset = document_store.count_unique_metadata_by_filter(
            filters={}, metadata_fields=["category", "status"]
        )
        assert distinct_counts_subset["category"] == 3
        assert distinct_counts_subset["status"] == 2
        assert "priority" not in distinct_counts_subset

        # Test field name normalization (with "meta." prefix)
        distinct_counts_normalized = document_store.count_unique_metadata_by_filter(
            filters={}, metadata_fields=["meta.category", "status", "meta.priority"]
        )
        assert distinct_counts_normalized["category"] == 3
        assert distinct_counts_normalized["status"] == 2
        assert distinct_counts_normalized["priority"] == 3

        # Test error handling when field doesn't exist
        with pytest.raises(ValueError, match="Fields not found in index mapping"):
            document_store.count_unique_metadata_by_filter(filters={}, metadata_fields=["nonexistent_field"])

    def test_get_metadata_fields_info(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "inactive"}),
        ]
        document_store.write_documents(docs)

        fields_info = document_store.get_metadata_fields_info()

        # Verify that fields_info contains expected fields
        assert "category" in fields_info
        assert "status" in fields_info
        assert "priority" in fields_info

        assert fields_info["category"]["type"] == "keyword"
        assert fields_info["status"]["type"] == "keyword"
        assert fields_info["priority"]["type"] == "long"

    def test_get_metadata_field_min_max(self, document_store: OpenSearchDocumentStore):
        # Test with integer values
        docs = [
            Document(content="Doc 1", meta={"priority": 1, "age": 10}),
            Document(content="Doc 2", meta={"priority": 5, "age": 20}),
            Document(content="Doc 3", meta={"priority": 3, "age": 15}),
            Document(content="Doc 4", meta={"priority": 10, "age": 5}),
            Document(content="Doc 6", meta={"rating": 10.5}),
            Document(content="Doc 7", meta={"rating": 20.3}),
            Document(content="Doc 8", meta={"rating": 15.7}),
            Document(content="Doc 9", meta={"rating": 5.2}),
        ]
        document_store.write_documents(docs)

        # Test with "meta." prefix for integer field
        min_max_priority = document_store.get_metadata_field_min_max("meta.priority")
        assert min_max_priority["min"] == 1
        assert min_max_priority["max"] == 10

        # Test with "meta." prefix for another integer field
        min_max_rating = document_store.get_metadata_field_min_max("meta.age")
        assert min_max_rating["min"] == 5
        assert min_max_rating["max"] == 20

        # Test with single value
        single_doc = [Document(content="Doc 5", meta={"single_value": 42})]
        document_store.write_documents(single_doc)
        min_max_single = document_store.get_metadata_field_min_max("meta.single_value")
        assert min_max_single["min"] == 42
        assert min_max_single["max"] == 42

        # Test with float values
        min_max_score = document_store.get_metadata_field_min_max("meta.rating")
        assert min_max_score["min"] == pytest.approx(5.2)
        assert min_max_score["max"] == pytest.approx(20.3)

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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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
