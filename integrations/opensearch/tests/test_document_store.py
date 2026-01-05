# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
import time
from unittest.mock import patch

import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
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


@pytest.mark.integration
class TestDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        The OpenSearchDocumentStore.filter_documents() method returns a Documents with their score set.
        We don't want to compare the score, so we set it to None before comparing the documents.
        """
        for doc in received:
            doc.score = None
        assert received == expected

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

    @pytest.fixture
    def document_store_embedding_dim_4_no_emb_returned(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=4,
            return_embedding=False,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})

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

    def test_delete_all_documents_no_index_recreation(self, document_store: OpenSearchDocumentStore):
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_all_documents(recreate_index=False)
        time.sleep(2)  # need to wait for the deletion to be reflected in count_documents
        assert document_store.count_documents() == 0

        new_doc = Document(id="3", content="New document after delete all")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

        results = document_store.filter_documents()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"

    def test_delete_by_filter(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # Delete documents with category="A"
        deleted_count = document_store.delete_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        time.sleep(2)  # wait for deletion to be reflected
        assert deleted_count == 2
        assert document_store.count_documents() == 1

        # Verify only category B remains
        remaining_docs = document_store.filter_documents()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    def test_update_by_filter(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # Update status for category="A" documents
        updated_count = document_store.update_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        time.sleep(2)  # wait for update to be reflected
        assert updated_count == 2

        # Verify the updates
        published_docs = document_store.filter_documents(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["category"] == "A"
            assert doc.meta["status"] == "published"

        # Verify category B still has draft status
        draft_docs = document_store.filter_documents(
            filters={"field": "meta.status", "operator": "==", "value": "draft"}
        )
        assert len(draft_docs) == 1
        assert draft_docs[0].meta["category"] == "B"

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

    def test_count_distinct_values_by_filter(self, document_store: OpenSearchDocumentStore):
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
        distinct_counts = document_store.count_distinct_values_by_filter(filters={})
        assert distinct_counts["category"] == 3  # A, B, C
        assert distinct_counts["status"] == 2  # active, inactive
        assert distinct_counts["priority"] == 3  # 1, 2, 3

        # Count distinct values for documents with category="A"
        distinct_counts_a = document_store.count_distinct_values_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert distinct_counts_a["category"] == 1  # Only A
        assert distinct_counts_a["status"] == 2  # active, inactive
        assert distinct_counts_a["priority"] == 2  # 1, 3

        # Count distinct values for documents with status="active"
        distinct_counts_active = document_store.count_distinct_values_by_filter(
            filters={"field": "meta.status", "operator": "==", "value": "active"}
        )
        assert distinct_counts_active["category"] == 3  # A, B, C
        assert distinct_counts_active["status"] == 1  # Only active
        assert distinct_counts_active["priority"] == 3  # 1, 2, 3

        # Count distinct values with complex filter (category="A" AND status="active")
        distinct_counts_a_active = document_store.count_distinct_values_by_filter(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert distinct_counts_a_active["category"] == 1  # Only A
        assert distinct_counts_a_active["status"] == 1  # Only active
        assert distinct_counts_a_active["priority"] == 2  # 1, 3

    def test_get_fields_info(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "inactive"}),
        ]
        document_store.write_documents(docs)

        fields_info = document_store.get_fields_info()

        # Verify that fields_info contains expected fields
        assert "content" in fields_info
        assert "embedding" in fields_info
        assert "category" in fields_info
        assert "status" in fields_info
        assert "priority" in fields_info

        # Verify field types
        assert fields_info["content"]["type"] == "text"
        assert fields_info["embedding"]["type"] == "knn_vector"
        # Metadata fields should be keyword type (from dynamic templates)
        assert fields_info["category"]["type"] == "keyword"
        assert fields_info["status"]["type"] == "keyword"
        assert fields_info["priority"]["type"] == "long"

    def test_get_field_min_max(self, document_store: OpenSearchDocumentStore):
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
        min_max_priority = document_store.get_field_min_max("meta.priority")
        assert min_max_priority["min"] == 1
        assert min_max_priority["max"] == 10

        # Test with "meta." prefix for another integer field
        min_max_rating = document_store.get_field_min_max("meta.age")
        assert min_max_rating["min"] == 5
        assert min_max_rating["max"] == 20

        # Test with single value
        single_doc = [Document(content="Doc 5", meta={"single_value": 42})]
        document_store.write_documents(single_doc)
        min_max_single = document_store.get_field_min_max("meta.single_value")
        assert min_max_single["min"] == 42
        assert min_max_single["max"] == 42
                
        # Test with float values
        min_max_score = document_store.get_field_min_max("meta.rating")
        assert min_max_score["min"] == pytest.approx(5.2)
        assert min_max_score["max"] == pytest.approx(20.3)