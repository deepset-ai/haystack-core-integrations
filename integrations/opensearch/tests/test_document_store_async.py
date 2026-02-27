# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.opensearch.document_store import OpenSearchDocumentStore


@pytest.mark.integration
class TestDocumentStoreAsync:
    @pytest.mark.asyncio
    async def test_write_documents(self, document_store: OpenSearchDocumentStore):
        assert await document_store.write_documents_async([Document(id="1")], policy=DuplicatePolicy.OVERWRITE) == 1

    @pytest.mark.asyncio
    async def test_bm25_retrieval(self, document_store: OpenSearchDocumentStore, test_documents: list[Document]):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional", top_k=3)

        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_pagination(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """

        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("programming", top_k=11)

        assert len(res) == 11
        assert all("programming" in doc.content for doc in res)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_all_terms_must_match(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional Haskell", top_k=3, all_terms_must_match=True)

        assert len(res) == 1
        assert "Haskell is a functional programming language" in res[0].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_all_terms_must_match_false(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async("functional Haskell", top_k=10, all_terms_must_match=False)

        assert len(res) == 5
        assert all("functional" in doc.content for doc in res)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_filters(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)
        res = await document_store._bm25_retrieval_async(
            "programming",
            top_k=10,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )

        assert len(res) == 5
        retrieved_ids = sorted([doc.id for doc in res])
        assert retrieved_ids == ["1", "2", "3", "4", "5"]

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_custom_query(
        self, document_store: OpenSearchDocumentStore, test_documents: list[Document]
    ):
        document_store.write_documents(test_documents)

        custom_query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": {"match": {"content": "$query"}},
                            "filter": "$filters",
                        }
                    },
                    "field_value_factor": {
                        "field": "likes",
                        "factor": 0.1,
                        "modifier": "log1p",
                        "missing": 0,
                    },
                }
            }
        }
        res = await document_store._bm25_retrieval_async(
            "functional",
            top_k=3,
            custom_query=custom_query,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )
        assert len(res) == 3
        assert "1" == res[0].id
        assert "2" == res[1].id
        assert "3" == res[2].id

    @pytest.mark.asyncio
    async def test_embedding_retrieval(self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        results = await document_store_embedding_dim_4_no_emb_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_with_filters(
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

        results = await document_store_embedding_dim_4_no_emb_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3, filters=filters
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_with_custom_query(
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
                "bool": {
                    "must": [{"knn": {"embedding": {"vector": "$query_embedding", "k": 3}}}],
                    "filter": "$filters",
                }
            }
        }

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}

        results = await document_store_embedding_dim_4_no_emb_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1],
            top_k=1,
            filters=filters,
            custom_query=custom_query,
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_but_dont_return_embeddings_for_embedding_retrieval(
        self, document_store_embedding_dim_4_no_emb_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4_no_emb_returned.write_documents(docs)

        results = await document_store_embedding_dim_4_no_emb_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].embedding is None

    @pytest.mark.asyncio
    async def test_count_documents(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="test doc 1"),
                Document(content="test doc 2"),
                Document(content="test doc 3"),
            ]
        )
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_filter_documents(self, document_store: OpenSearchDocumentStore):
        filterable_docs = [
            Document(
                content="1",
                meta={
                    "number": -10,
                },
            ),
            Document(
                content="2",
                meta={
                    "number": 100,
                },
            ),
        ]
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )

        assert len(result) == 1
        assert result[0].content == "2"
        assert result[0].meta["number"] == 100

    @pytest.mark.asyncio
    async def test_count_documents_by_filter(self, document_store: OpenSearchDocumentStore):
        filterable_docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        await document_store.write_documents_async(filterable_docs)
        assert await document_store.count_documents_async() == 4

        count_a = await document_store.count_documents_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count_a == 3

        count_active = await document_store.count_documents_by_filter_async(
            filters={"field": "meta.status", "operator": "==", "value": "active"}
        )
        assert count_active == 3

        count_a_active = await document_store.count_documents_by_filter_async(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert count_a_active == 2

    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter(self, document_store: OpenSearchDocumentStore):
        filterable_docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2}),
        ]
        await document_store.write_documents_async(filterable_docs)
        assert await document_store.count_documents_async() == 5

        # count distinct values for all documents
        distinct_counts = await document_store.count_unique_metadata_by_filter_async(
            filters={}, metadata_fields=["category", "status", "priority"]
        )
        assert distinct_counts["category"] == 3  # A, B, C
        assert distinct_counts["status"] == 2  # active, inactive
        assert distinct_counts["priority"] == 3  # 1, 2, 3

        # count distinct values for documents with category="A"
        distinct_counts_a = await document_store.count_unique_metadata_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_a["category"] == 1  # Only A
        assert distinct_counts_a["status"] == 2  # active, inactive
        assert distinct_counts_a["priority"] == 2  # 1, 3

        # count distinct values for documents with status="active"
        distinct_counts_active = await document_store.count_unique_metadata_by_filter_async(
            filters={"field": "meta.status", "operator": "==", "value": "active"},
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_active["category"] == 3  # A, B, C
        assert distinct_counts_active["status"] == 1  # Only active
        assert distinct_counts_active["priority"] == 3  # 1, 2, 3

        # count distinct values with complex filter (category="A" AND status="active")
        distinct_counts_a_active = await document_store.count_unique_metadata_by_filter_async(
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
        distinct_counts_subset = await document_store.count_unique_metadata_by_filter_async(
            filters={}, metadata_fields=["category", "status"]
        )
        assert distinct_counts_subset["category"] == 3
        assert distinct_counts_subset["status"] == 2
        assert "priority" not in distinct_counts_subset

        # Test field name normalization (with "meta." prefix)
        distinct_counts_normalized = await document_store.count_unique_metadata_by_filter_async(
            filters={}, metadata_fields=["meta.category", "status", "meta.priority"]
        )
        assert distinct_counts_normalized["category"] == 3
        assert distinct_counts_normalized["status"] == 2
        assert distinct_counts_normalized["priority"] == 3

        # Test error handling when field doesn't exist
        with pytest.raises(ValueError, match="Fields not found in index mapping"):
            await document_store.count_unique_metadata_by_filter_async(
                filters={}, metadata_fields=["nonexistent_field"]
            )

    @pytest.mark.asyncio
    async def test_delete_documents(self, document_store: OpenSearchDocumentStore):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert document_store.count_documents() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0

    @pytest.mark.asyncio
    async def test_delete_all_documents_index_recreation(self, document_store: OpenSearchDocumentStore):
        # populate the index with some documents
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)

        # capture index structure before deletion
        assert document_store._client is not None
        index_info_before = document_store._client.indices.get(index=document_store._index)
        mappings_before = index_info_before[document_store._index]["mappings"]
        settings_before = index_info_before[document_store._index]["settings"]

        # delete all documents
        await document_store.delete_all_documents_async(recreate_index=True)
        assert await document_store.count_documents_async() == 0

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
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"

    @pytest.mark.asyncio
    async def test_delete_all_documents_no_index_recreation(self, document_store: OpenSearchDocumentStore):
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        await document_store.delete_all_documents_async(recreate_index=False, refresh=True)
        assert await document_store.count_documents_async() == 0

        new_doc = Document(id="3", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"

    @pytest.mark.asyncio
    async def test_write_with_routing(self, document_store: OpenSearchDocumentStore):
        """Test async writing documents with routing metadata"""
        docs = [
            Document(id="1", content="User A doc", meta={"_routing": "user_a", "category": "test"}),
            Document(id="2", content="User B doc", meta={"_routing": "user_b"}),
            Document(id="3", content="No routing"),
        ]

        written = await document_store.write_documents_async(docs)
        assert written == 3
        assert await document_store.count_documents_async() == 3

        # Verify _routing not stored in metadata
        retrieved = await document_store.filter_documents_async()
        retrieved_by_id = {doc.id: doc for doc in retrieved}

        # Check _routing is not stored for any document
        for doc in retrieved:
            assert "_routing" not in doc.meta

        assert retrieved_by_id["1"].meta["category"] == "test"
        assert retrieved_by_id["2"].meta == {}
        assert retrieved_by_id["3"].meta == {}

    @pytest.mark.asyncio
    async def test_delete_with_routing(self, document_store: OpenSearchDocumentStore):
        """Test async deleting documents with routing"""
        docs = [
            Document(id="1", content="Doc 1", meta={"_routing": "user_a"}),
            Document(id="2", content="Doc 2", meta={"_routing": "user_b"}),
            Document(id="3", content="Doc 3"),
        ]
        await document_store.write_documents_async(docs)

        routing_map = {"1": "user_a", "2": "user_b"}
        await document_store.delete_documents_async(["1", "2"], routing=routing_map)
        assert await document_store.count_documents_async() == 1

    async def test_delete_by_filter_async(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Delete documents with category="A"
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, refresh=True
        )
        assert deleted_count == 2
        assert await document_store.count_documents_async() == 1

        # Verify only category B remains
        remaining_docs = await document_store.filter_documents_async()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    async def test_update_by_filter_async(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Update status for category="A" documents
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            meta={"status": "published"},
            refresh=True,
        )
        assert updated_count == 2

        # Verify the updates
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["category"] == "A"
            assert doc.meta["status"] == "published"

        # Verify category B still has draft status
        draft_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "draft"}
        )
        assert len(draft_docs) == 1
        assert draft_docs[0].meta["category"] == "B"

    @pytest.mark.asyncio
    async def test_get_metadata_fields_info_async(self, document_store: OpenSearchDocumentStore):
        filterable_docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "inactive"}),
        ]
        await document_store.write_documents_async(filterable_docs)

        fields_info = await document_store.get_metadata_fields_info_async()

        # Verify that fields_info contains expected fields
        assert "category" in fields_info
        assert "status" in fields_info
        assert "priority" in fields_info

        assert fields_info["category"]["type"] == "keyword"
        assert fields_info["status"]["type"] == "keyword"
        assert fields_info["priority"]["type"] == "long"

    @pytest.mark.asyncio
    async def test_get_metadata_field_min_max_async(self, document_store: OpenSearchDocumentStore):
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
        await document_store.write_documents_async(docs)

        # Test with "meta." prefix for integer field
        min_max_priority = await document_store.get_metadata_field_min_max_async("meta.priority")
        assert min_max_priority["min"] == 1
        assert min_max_priority["max"] == 10

        # Test with "meta." prefix for another integer field
        min_max_rating = await document_store.get_metadata_field_min_max_async("meta.age")
        assert min_max_rating["min"] == 5
        assert min_max_rating["max"] == 20

        # Test with single value
        single_doc = [Document(content="Doc 5", meta={"single_value": 42})]
        await document_store.write_documents_async(single_doc)
        min_max_single = await document_store.get_metadata_field_min_max_async("meta.single_value")
        assert min_max_single["min"] == 42
        assert min_max_single["max"] == 42

        # Test with float values
        min_max_score = await document_store.get_metadata_field_min_max_async("meta.rating")
        assert min_max_score["min"] == pytest.approx(5.2)
        assert min_max_score["max"] == pytest.approx(20.3)

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async(self, document_store: OpenSearchDocumentStore):
        # Test with string values
        docs = [
            Document(content="Python programming", meta={"category": "A", "language": "Python"}),
            Document(content="Java programming", meta={"category": "B", "language": "Java"}),
            Document(content="Python scripting", meta={"category": "A", "language": "Python"}),
            Document(content="JavaScript development", meta={"category": "C", "language": "JavaScript"}),
            Document(content="Python data science", meta={"category": "A", "language": "Python"}),
            Document(content="Java backend", meta={"category": "B", "language": "Java"}),
        ]
        await document_store.write_documents_async(docs)

        # Test getting all unique values without search term
        unique_values, after_key = await document_store.get_metadata_field_unique_values_async(
            "meta.category", None, 10
        )
        assert set(unique_values) == {"A", "B", "C"}
        # after_key should be None when all results are returned
        assert after_key is None

        # Test with "meta." prefix
        unique_languages, _ = await document_store.get_metadata_field_unique_values_async("meta.language", None, 10)
        assert set(unique_languages) == {"Python", "Java", "JavaScript"}

        # Test pagination - first page
        unique_values_page1, after_key_page1 = await document_store.get_metadata_field_unique_values_async(
            "meta.category", None, 2
        )
        assert len(unique_values_page1) == 2
        assert all(val in ["A", "B", "C"] for val in unique_values_page1)
        # Should have an after_key for pagination
        assert after_key_page1 is not None

        # Test pagination - second page using after_key
        unique_values_page2, after_key_page2 = await document_store.get_metadata_field_unique_values_async(
            "meta.category", None, 2, after=after_key_page1
        )
        assert len(unique_values_page2) == 1
        assert unique_values_page2[0] in ["A", "B", "C"]
        # Should have no more results
        assert after_key_page2 is None

        # Test with search term - filter by content matching "Python"
        unique_values_filtered, _ = await document_store.get_metadata_field_unique_values_async(
            "meta.category", "Python", 10
        )
        assert set(unique_values_filtered) == {"A"}  # Only category A has documents with "Python" in content

        # Test with search term - filter by content matching "Java"
        unique_values_java, _ = await document_store.get_metadata_field_unique_values_async("meta.category", "Java", 10)
        assert set(unique_values_java) == {"B"}  # Only category B has documents with "Java" in content

        # Test with integer values
        int_docs = [
            Document(content="Doc 1", meta={"priority": 1}),
            Document(content="Doc 2", meta={"priority": 2}),
            Document(content="Doc 3", meta={"priority": 1}),
            Document(content="Doc 4", meta={"priority": 3}),
        ]
        await document_store.write_documents_async(int_docs)
        unique_priorities, _ = await document_store.get_metadata_field_unique_values_async("meta.priority", None, 10)
        assert set(unique_priorities) == {"1", "2", "3"}

        # Test with search term on integer field
        unique_priorities_filtered, _ = await document_store.get_metadata_field_unique_values_async(
            "meta.priority", "Doc 1", 10
        )
        assert set(unique_priorities_filtered) == {"1"}

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_search_async_fuzzy_mode(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search in fuzzy mode."""
        docs = [
            Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "Python", "status": "inactive", "priority": 3}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search for "Python" in category field
        result = await document_store._metadata_search_async(
            query="Python",
            fields=["category"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) >= 2  # At least 2 documents with category "Python"
        assert all(isinstance(row, dict) for row in result)
        assert all("category" in row for row in result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_search_async_strict_mode(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search in strict mode."""
        docs = [
            Document(content="Python programming", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "Java", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search for "Python" in category field with strict mode
        result = await document_store._metadata_search_async(
            query="Python",
            fields=["category"],
            mode="strict",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) >= 1  # At least 1 document with category "Python"
        assert all(isinstance(row, dict) for row in result)
        assert all("category" in row for row in result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_search_async_multiple_fields(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search across multiple fields."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Java", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search for "active" across both category and status fields
        result = await document_store._metadata_search_async(
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
    @pytest.mark.asyncio
    async def test_metadata_search_async_top_k(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search respects top_k parameter."""
        docs = [Document(content=f"Doc {i}", meta={"category": "Python", "index": i}) for i in range(15)]
        document_store.write_documents(docs, refresh=True)

        # Request top 5 results
        result = await document_store._metadata_search_async(
            query="Python",
            fields=["category"],
            mode="fuzzy",
            top_k=5,
        )

        assert isinstance(result, list)
        assert len(result) <= 5

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_search_async_comma_separated_query(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search with comma-separated query parts."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Java", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "Python", "status": "inactive", "priority": 3}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search for "Python, active" - should match documents with both
        result = await document_store._metadata_search_async(
            query="Python, active",
            fields=["category", "status"],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(row, dict) for row in result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_search_async_with_filters(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search with additional filters."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "Python", "status": "inactive", "priority": 2}),
            Document(content="Doc 3", meta={"category": "Java", "status": "active", "priority": 1}),
        ]
        document_store.write_documents(docs, refresh=True)

        # Search with filter for priority == 1
        filters = {"field": "priority", "operator": "==", "value": 1}
        result = await document_store._metadata_search_async(
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
    @pytest.mark.asyncio
    async def test_metadata_search_async_empty_fields(self, document_store: OpenSearchDocumentStore):
        """Test async metadata search with empty fields list returns empty result."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python"}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
            query="Python",
            fields=[],
            mode="fuzzy",
            top_k=10,
        )

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_search_async_deduplication(self, document_store: OpenSearchDocumentStore):
        """Test that async metadata search deduplicates results."""
        docs = [
            Document(content="Doc 1", meta={"category": "Python", "status": "active"}),
            Document(content="Doc 2", meta={"category": "Python", "status": "active"}),
        ]
        document_store.write_documents(docs, refresh=True)

        result = await document_store._metadata_search_async(
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

    @pytest.mark.asyncio
    async def test_query_sql(self, document_store: OpenSearchDocumentStore):
        docs = [
            Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
            Document(content="JavaScript development", meta={"category": "C", "status": "active", "priority": 1}),
        ]
        await document_store.write_documents_async(docs, refresh=True)

        # SQL query returns raw JSON response from OpenSearch SQL API
        sql_query = (
            f"SELECT content, category, status, priority FROM {document_store._index} "  # noqa: S608
            f"WHERE category = 'A' ORDER BY priority"
        )
        result = await document_store._query_sql_async(sql_query)

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
            await document_store._query_sql_async(invalid_query)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_sql_async_with_fetch_size(self, document_store: OpenSearchDocumentStore):
        """Test async SQL query with fetch_size parameter"""
        # Create multiple documents to test pagination
        docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(15)]
        await document_store.write_documents_async(docs, refresh=True)

        sql_query = (
            f"SELECT content, category, index FROM {document_store._index} "  # noqa: S608
            f"WHERE category = 'A' ORDER BY index"
        )

        # Test with fetch_size
        result = await document_store._query_sql_async(sql_query, fetch_size=5)

        # Should return raw JSON response (exact count depends on OpenSearch behavior)
        assert isinstance(result, dict)
        assert "schema" in result
        assert "datarows" in result
        assert "size" in result
        assert "status" in result
        assert [entry["name"] for entry in result["schema"]] == ["content", "category", "index"]
        assert len(result["datarows"]) > 0
        assert len(result["datarows"]) <= 5
        assert result.get("cursor") is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_sql_async_pagination_flow(self, document_store: OpenSearchDocumentStore):
        """Test async pagination flow with fetch_size"""
        # Create enough documents to require pagination
        docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(20)]
        await document_store.write_documents_async(docs, refresh=True)

        sql_query = (
            f"SELECT content, category, index FROM {document_store._index} "  # noqa: S608
            f"WHERE category = 'A' ORDER BY index"
        )

        # Query with small fetch_size to test pagination
        result = await document_store._query_sql_async(sql_query, fetch_size=10)
        assert isinstance(result, dict)
        assert "schema" in result
        assert "datarows" in result
        assert "size" in result
        assert "status" in result
        assert [entry["name"] for entry in result["schema"]] == ["content", "category", "index"]
        assert len(result["datarows"]) > 0
        assert len(result["datarows"]) <= 10

        # Verify all results contain expected row columns
        for row in result["datarows"]:
            assert len(row) == 3
