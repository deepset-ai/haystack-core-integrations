# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses.document import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


@pytest.mark.integration
class TestElasticsearchDocumentStoreAsync:
    @pytest.fixture
    async def document_store(self, request):
        """
        Basic fixture providing a document store instance for async tests
        """
        hosts = ["http://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = ElasticsearchDocumentStore(hosts=hosts, index=index)
        yield store
        store.client.options(ignore_status=[400, 404]).indices.delete(index=index)

        await store.async_client.close()

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store):
        docs = [Document(id="1", content="test")]
        assert await document_store.write_documents_async(docs) == 1
        assert await document_store.count_documents_async() == 1
        with pytest.raises(DocumentStoreError):
            await document_store.write_documents_async(docs, policy=DuplicatePolicy.FAIL)

    @pytest.mark.asyncio
    async def test_count_documents_async(self, document_store):
        docs = [
            Document(content="test doc 1"),
            Document(content="test doc 2"),
            Document(content="test doc 3"),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_delete_documents_async(self, document_store):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1
        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0

    @pytest.mark.asyncio
    async def test_filter_documents_async(self, document_store):
        filterable_docs = [
            Document(content="1", meta={"number": -10}),
            Document(content="2", meta={"number": 100}),
        ]
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "number", "operator": "==", "value": 100}
        )
        assert len(result) == 1
        assert result[0].meta["number"] == 100

    @pytest.mark.asyncio
    async def test_bm25_retrieval_async(self, document_store):
        docs = [
            Document(content="Haskell is a functional programming language"),
            Document(content="Python is an object oriented programming language"),
        ]
        await document_store.write_documents_async(docs)
        results = await document_store._bm25_retrieval_async("functional", top_k=1)
        assert len(results) == 1
        assert "functional" in results[0].content

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async(self, document_store):
        # init document store
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Less similar document", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        await document_store.write_documents_async(docs)

        # without num_candidates set to None
        results = await document_store._embedding_retrieval_async(query_embedding=[1.0, 1.0, 1.0, 1.0], top_k=1)
        assert len(results) == 1
        assert results[0].content == "Most similar document"

        # with num_candidates not None
        results = await document_store._embedding_retrieval_async(
            query_embedding=[1.0, 1.0, 1.0, 1.0], top_k=2, num_candidates=2
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document"

        # with an embedding containing None
        with pytest.raises(ValueError, match="query_embedding must be a non-empty list of floats"):
            _ = await document_store._embedding_retrieval_async(query_embedding=None, top_k=2)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_async_with_filters(self, document_store):
        docs = [
            Document(content="Haskell is a functional programming language", meta={"type": "functional"}),
            Document(content="Python is an object oriented programming language", meta={"type": "oop"}),
        ]
        await document_store.write_documents_async(docs)
        results = await document_store._bm25_retrieval_async(
            "programming", filters={"field": "type", "operator": "==", "value": "functional"}, top_k=1
        )
        assert len(results) == 1
        assert "functional" in results[0].content

        # test with scale_score=True
        results = await document_store._bm25_retrieval_async(
            "programming", filters={"field": "type", "operator": "==", "value": "functional"}, top_k=1, scale_score=True
        )
        assert len(results) == 1
        assert "functional" in results[0].content
        assert 0 <= results[0].score <= 1  # score should be between 0 and 1

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async_with_filters(self, document_store):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0], meta={"type": "similar"}),
            Document(content="Less similar document", embedding=[0.5, 0.5, 0.5, 0.5], meta={"type": "different"}),
        ]
        await document_store.write_documents_async(docs)
        results = await document_store._embedding_retrieval_async(
            query_embedding=[1.0, 1.0, 1.0, 1.0],
            filters={"field": "type", "operator": "==", "value": "similar"},
            top_k=1,
        )
        assert len(results) == 1
        assert results[0].content == "Most similar document"

    @pytest.mark.asyncio
    async def test_write_documents_async_invalid_document_type(self, document_store):
        """Test write_documents with invalid document type"""
        invalid_docs = [{"id": "1", "content": "test"}]  # Dictionary instead of Document object
        with pytest.raises(ValueError, match="param 'documents' must contain a list of objects of type Document"):
            await document_store.write_documents_async(invalid_docs)

    @pytest.mark.asyncio
    async def test_write_documents_async_with_sparse_embedding_warning(self, document_store, caplog):
        """Test write_documents with document containing sparse_embedding field"""
        doc = Document(id="1", content="test", sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.5]))

        await document_store.write_documents_async([doc])
        assert "but storing sparse embeddings in Elasticsearch is not currently supported." in caplog.text

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].id == "1"
        assert not hasattr(results[0], "sparse_embedding") or results[0].sparse_embedding is None

    @pytest.mark.asyncio
    async def test_delete_all_documents_async(self, document_store):
        docs = [
            Document(id="1", content="First document", meta={"category": "test"}),
            Document(id="2", content="Second document", meta={"category": "test"}),
            Document(id="3", content="Third document", meta={"category": "other"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # delete all documents
        await document_store.delete_all_documents_async(recreate_index=False)
        assert await document_store.count_documents_async() == 0

        # verify index still exists and can accept new documents and retrieve
        new_doc = Document(id="4", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].id == "4"
        assert results[0].content == "New document after delete all"

    @pytest.mark.asyncio
    async def test_delete_all_documents_async_index_recreation(self, document_store):
        # populate the index with some documents
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)

        # capture index structure before deletion
        assert document_store._async_client is not None
        index_info_before = await document_store._async_client.indices.get(index=document_store._index)
        mappings_before = index_info_before[document_store._index]["mappings"]
        settings_before = index_info_before[document_store._index]["settings"]

        # delete all documents with index recreation
        await document_store.delete_all_documents_async(recreate_index=True)
        assert await document_store.count_documents_async() == 0

        # verify index structure is preserved
        index_info_after = await document_store._async_client.indices.get(index=document_store._index)
        mappings_after = index_info_after[document_store._index]["mappings"]
        assert mappings_after == mappings_before, "delete_all_documents_async should preserve index mappings"

        settings_after = index_info_after[document_store._index]["settings"]
        settings_after["index"].pop("uuid", None)
        settings_after["index"].pop("creation_date", None)
        settings_before["index"].pop("uuid", None)
        settings_before["index"].pop("creation_date", None)
        assert settings_after == settings_before, "delete_all_documents_async should preserve index settings"

        # verify index can accept new documents and retrieve
        new_doc = Document(id="4", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].content == "New document after delete all"

    @pytest.mark.asyncio
    async def test_delete_all_documents_async_no_index_recreation(self, document_store):
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        await document_store.delete_all_documents_async(recreate_index=False, refresh=True)
        assert await document_store.count_documents_async() == 0

        new_doc = Document(id="3", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

    @pytest.mark.asyncio
    async def test_delete_by_filter_async(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Delete documents with category="A"
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "category", "operator": "==", "value": "A"}, refresh=True
        )

        assert deleted_count == 2
        assert await document_store.count_documents_async() == 1

        # Verify only category B remains
        remaining_docs = await document_store.filter_documents_async()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    @pytest.mark.asyncio
    async def test_update_by_filter_async(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Update status for category="A" documents
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "category", "operator": "==", "value": "A"},
            meta={"status": "published"},
            refresh=True,
        )
        assert updated_count == 2

        # Verify the updates
        published_docs = await document_store.filter_documents_async(
            filters={"field": "status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["category"] == "A"
            assert doc.meta["status"] == "published"

        # Verify category B still has draft status
        draft_docs = await document_store.filter_documents_async(
            filters={"field": "status", "operator": "==", "value": "draft"}
        )
        assert len(draft_docs) == 1
        assert draft_docs[0].meta["category"] == "B"

    @pytest.mark.asyncio
    async def test_count_documents_by_filter_async(self, document_store: ElasticsearchDocumentStore):
        filterable_docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        await document_store.write_documents_async(filterable_docs)
        assert await document_store.count_documents_async() == 4

        count_a = await document_store.count_documents_by_filter_async(
            filters={"field": "category", "operator": "==", "value": "A"}
        )
        assert count_a == 3

        count_active = await document_store.count_documents_by_filter_async(
            filters={"field": "status", "operator": "==", "value": "active"}
        )
        assert count_active == 3

        count_a_active = await document_store.count_documents_by_filter_async(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "category", "operator": "==", "value": "A"},
                    {"field": "status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert count_a_active == 2

    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async(self, document_store: ElasticsearchDocumentStore):
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
            filters={"field": "category", "operator": "==", "value": "A"},
            metadata_fields=["category", "status", "priority"],
        )
        assert distinct_counts_a["category"] == 1  # Only A
        assert distinct_counts_a["status"] == 2  # active, inactive
        assert distinct_counts_a["priority"] == 2  # 1, 3

        # count distinct values for documents with status="active"
        distinct_counts_active = await document_store.count_unique_metadata_by_filter_async(
            filters={"field": "status", "operator": "==", "value": "active"},
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
                    {"field": "category", "operator": "==", "value": "A"},
                    {"field": "status", "operator": "==", "value": "active"},
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
    async def test_get_metadata_fields_info_async(self, document_store: ElasticsearchDocumentStore):
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
    async def test_get_metadata_field_min_max_async(self, document_store: ElasticsearchDocumentStore):
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
    async def test_get_metadata_field_unique_values_async(self, document_store: ElasticsearchDocumentStore):
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
