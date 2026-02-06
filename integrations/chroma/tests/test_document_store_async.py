# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import operator
import sys
import uuid
from unittest import mock

import pytest
from haystack.dataclasses import Document
from haystack.testing.document_store import TEST_EMBEDDING_1

from haystack_integrations.document_stores.chroma import ChromaDocumentStore


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="We do not run the Chroma server on Windows and async is only supported with HTTP connections",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestDocumentStoreAsync:
    @pytest.fixture
    def document_store(self, embedding_function) -> ChromaDocumentStore:
        with mock.patch(
            "haystack_integrations.document_stores.chroma.document_store.get_embedding_function"
        ) as get_func:
            get_func.return_value = embedding_function
            return ChromaDocumentStore(
                embedding_function="test_function",
                collection_name=f"{uuid.uuid1()}-async",
                host="localhost",
                port=8000,
            )

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method.

        This can happen for example when the Document Store sets a score to returned Documents.
        Since we can't know what the score will be, we can't compare the Documents reliably.
        """
        received.sort(key=operator.attrgetter("id"))
        expected.sort(key=operator.attrgetter("id"))

        for doc_received, doc_expected in zip(received, expected, strict=True):
            assert doc_received.content == doc_expected.content
            assert doc_received.meta == doc_expected.meta

    async def test_write_documents_async(self, document_store: ChromaDocumentStore):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

    async def test_delete_documents_async(self, document_store: ChromaDocumentStore):
        """Test delete_documents() normal behaviour."""
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0

    async def test_count_empty_async(self, document_store: ChromaDocumentStore):
        """Test count is zero for an empty document store"""
        assert await document_store.count_documents_async() == 0

    async def test_count_not_empty_async(self, document_store: ChromaDocumentStore):
        """Test count is greater than zero if the document store contains documents"""
        await document_store.write_documents_async(
            [
                Document(content="test doc 1"),
                Document(content="test doc 2"),
                Document(content="test doc 3"),
            ]
        )
        assert await document_store.count_documents_async() == 3

    async def test_no_filters_async(self, document_store):
        """Test filter_documents() with empty filters"""
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [])
        self.assert_documents_are_equal(await document_store.filter_documents_async(filters={}), [])
        docs = [Document(content="test doc")]
        await document_store.write_documents_async(docs)
        self.assert_documents_are_equal(await document_store.filter_documents_async(), docs)
        self.assert_documents_are_equal(await document_store.filter_documents_async(filters={}), docs)

    async def test_comparison_equal_async(self, document_store, filterable_docs):
        """Test filter_documents() with == comparator"""
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") == 100])

    async def test_client_settings_applied_async(self):
        store = ChromaDocumentStore(
            host="localhost",
            port=8000,
            client_settings={"anonymized_telemetry": False},
            collection_name=f"{uuid.uuid1()}-async-settings",
        )
        await store._ensure_initialized_async()
        assert store._async_client.get_settings().anonymized_telemetry is False

    async def test_invalid_client_settings_async(self):
        store = ChromaDocumentStore(
            host="localhost",
            port=8000,
            client_settings={
                "invalid_setting_name": "some_value",
                "another_fake_setting": 123,
            },
            collection_name=f"{uuid.uuid1()}-async-invalid",
        )
        with pytest.raises(ValueError, match="Invalid client_settings"):
            await store._ensure_initialized_async()

    async def test_search_async(self):
        document_store = ChromaDocumentStore(host="localhost", port=8000, collection_name="my_custom_collection")

        documents = [
            Document(content="First document", meta={"author": "Author1"}),
            Document(content="Second document"),  # No metadata
            Document(content="Third document", meta={"author": "Author2"}),
            Document(content="Fourth document"),  # No metadata
        ]
        await document_store.write_documents_async(documents)
        result = await document_store.search_async(["Third"], top_k=1)

        # Assertions to verify correctness
        assert len(result) == 1
        doc = result[0][0]
        assert doc.content == "Third document"
        assert doc.meta == {"author": "Author2"}
        assert doc.embedding
        assert isinstance(doc.embedding, list)
        assert all(isinstance(el, float) for el in doc.embedding)

        # check that empty filters behave as no filters
        result_empty_filters = document_store.search(["Third"], filters={}, top_k=1)
        assert result == result_empty_filters

    @pytest.mark.asyncio
    async def test_delete_all_documents_index_recreation(self, document_store: ChromaDocumentStore):
        # write some documents
        docs = [
            Document(id="1", content="First document", meta={"category": "test"}),
            Document(id="2", content="Second document", meta={"category": "test"}),
            Document(id="3", content="Third document", meta={"category": "other"}),
        ]
        await document_store.write_documents_async(docs)

        # get the current document_store config
        config_before = await document_store._async_collection.get(document_store._collection_name)

        # delete all documents with recreating the index
        await document_store.delete_all_documents_async(recreate_index=True)
        assert await document_store.count_documents_async() == 0

        # assure that with the same config
        config_after = await document_store._async_collection.get(document_store._collection_name)

        assert config_before == config_after

        # ensure the collection still exists by writing documents again
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

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
        await document_store.delete_all_documents_async()
        assert await document_store.count_documents_async() == 0

        # verify index still exists and can accept new documents and retrieve
        new_doc = Document(id="4", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].id == "4"
        assert results[0].content == "New document after delete all"

    async def test_delete_by_filter_async(self, document_store: ChromaDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Delete documents with category="A"
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert deleted_count == 2
        assert await document_store.count_documents_async() == 1

        # Verify only category B remains
        remaining_docs = await document_store.filter_documents_async()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    async def test_delete_by_filter_async_no_matches(self, document_store: ChromaDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        # Try to delete documents with category="C" (no matches)
        deleted_count = await document_store.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "C"}
        )
        assert deleted_count == 0
        assert await document_store.count_documents_async() == 2

    async def test_update_by_filter_async(self, document_store: ChromaDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        # Update status for category="A" documents
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 2

        # Verify the updated documents have the new metadata
        published_docs = await document_store.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["status"] == "published"
            assert doc.meta["category"] == "A"

        # Verify documents with category="B" were not updated
        unpublished_docs = await document_store.filter_documents_async(
            filters={"field": "meta.category", "operator": "==", "value": "B"}
        )
        assert len(unpublished_docs) == 1
        assert unpublished_docs[0].meta["status"] == "draft"

    async def test_update_by_filter_async_no_matches(self, document_store: ChromaDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        # Try to update documents with category="C" (no matches)
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "C"}, meta={"status": "published"}
        )
        assert updated_count == 0
        assert await document_store.count_documents_async() == 2

    @pytest.mark.integration
    async def test_search_embeddings_async(self, document_store: ChromaDocumentStore):
        query_embedding = TEST_EMBEDDING_1
        documents = [
            Document(content="First document", embedding=TEST_EMBEDDING_1, meta={"author": "Author1"}),
            Document(content="Second document", embedding=[0.1] * len(TEST_EMBEDDING_1)),
            Document(content="Third document", embedding=TEST_EMBEDDING_1, meta={"author": "Author2"}),
        ]
        await document_store.write_documents_async(documents)
        result = await document_store.search_embeddings_async([query_embedding], top_k=2)

        # Assertions to verify correctness
        assert len(result) == 1
        assert len(result[0]) == 2
        # The documents with matching embeddings should be returned
        assert all(doc.embedding == pytest.approx(TEST_EMBEDDING_1) for doc in result[0])
        assert all(doc.score is not None for doc in result[0])

        # check that empty filters behave as no filters
        result_empty_filters = await document_store.search_embeddings_async([query_embedding], filters={}, top_k=2)
        assert len(result_empty_filters) == 1
        assert len(result_empty_filters[0]) == 2


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="We do not run the Chroma server on Windows and async is only supported with HTTP connections",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestMetadataOperationsAsync:
    """Test async metadata query operations for ChromaDocumentStore"""

    @pytest.fixture
    def document_store(self, embedding_function) -> ChromaDocumentStore:
        with mock.patch(
            "haystack_integrations.document_stores.chroma.document_store.get_embedding_function"
        ) as get_func:
            get_func.return_value = embedding_function
            return ChromaDocumentStore(
                embedding_function="test_function",
                collection_name=f"{uuid.uuid1()}-async",
                host="localhost",
                port=8000,
            )

    @pytest.fixture
    async def populated_store(self, document_store: ChromaDocumentStore) -> ChromaDocumentStore:
        """Fixture with pre-populated test documents with diverse metadata"""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1, "score": 0.9}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2, "score": 0.8}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1, "score": 0.7}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3, "score": 0.95}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2, "score": 0.6}),
            Document(content="Doc 6", meta={"category": "B", "status": "inactive", "priority": 1}),
        ]
        await document_store.write_documents_async(docs)
        return document_store

    async def test_count_documents_by_filter_async_simple(self, populated_store):
        """Test counting documents with simple filter"""
        count = await populated_store.count_documents_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count == 3

    async def test_count_documents_by_filter_async_compound(self, populated_store):
        """Test counting documents with compound filter"""
        count = await populated_store.count_documents_by_filter_async(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert count == 2

    async def test_count_unique_metadata_by_filter_async(self, populated_store):
        """Test counting unique metadata values"""
        counts = await populated_store.count_unique_metadata_by_filter_async({}, ["category", "status"])
        assert counts["category"] == 3  # A, B, C
        assert counts["status"] == 2  # active, inactive

    async def test_count_unique_metadata_by_filter_async_with_filter(self, populated_store):
        """Test counting unique metadata values with filter"""
        counts = await populated_store.count_unique_metadata_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, metadata_fields=["status"]
        )
        assert counts["status"] == 2  # active, inactive

    async def test_get_metadata_fields_info_async(self, populated_store):
        """Test getting metadata field information"""
        fields_info = await populated_store.get_metadata_fields_info_async()

        assert "category" in fields_info
        assert "status" in fields_info
        assert "priority" in fields_info
        assert "score" in fields_info

        # Check types
        assert fields_info["category"]["type"] == "keyword"
        assert fields_info["status"]["type"] == "keyword"
        assert fields_info["priority"]["type"] == "long"
        assert fields_info["score"]["type"] == "float"

    async def test_get_metadata_fields_info_async_empty_collection(self, document_store):
        """Test getting metadata field info from empty collection"""
        fields_info = await document_store.get_metadata_fields_info_async()
        assert fields_info == {}

    async def test_get_metadata_field_min_max_async_numeric(self, populated_store):
        """Test getting min/max values for numeric field"""
        min_max = await populated_store.get_metadata_field_min_max_async("priority")
        assert min_max["min"] == 1
        assert min_max["max"] == 3

    async def test_get_metadata_field_min_max_async_float(self, populated_store):
        """Test getting min/max values for float field"""
        min_max = await populated_store.get_metadata_field_min_max_async("score")
        assert min_max["min"] == 0.6
        assert min_max["max"] == 0.95

    async def test_get_metadata_field_min_max_async_string(self, populated_store):
        """Test getting min/max values for string field (alphabetical)"""
        min_max = await populated_store.get_metadata_field_min_max_async("category")
        assert min_max["min"] == "A"
        assert min_max["max"] == "C"

    async def test_get_metadata_field_min_max_async_missing_field(self, populated_store):
        """Test getting min/max for non-existent field"""
        min_max = await populated_store.get_metadata_field_min_max_async("nonexistent_field")
        assert min_max["min"] is None
        assert min_max["max"] is None

    async def test_get_metadata_field_unique_values_async_basic(self, populated_store):
        """Test getting unique values for metadata field"""
        values, total = await populated_store.get_metadata_field_unique_values_async("category", from_=0, size=10)
        assert sorted(values) == ["A", "B", "C"]
        assert total == 3

    async def test_get_metadata_field_unique_values_async_pagination(self, populated_store):
        """Test pagination of unique values"""
        # First page
        values_page1, total = await populated_store.get_metadata_field_unique_values_async("category", from_=0, size=2)
        assert len(values_page1) == 2
        assert total == 3

        # Second page
        values_page2, total = await populated_store.get_metadata_field_unique_values_async("category", from_=2, size=2)
        assert len(values_page2) == 1
        assert total == 3

        # Check all values are returned across pages
        all_values = values_page1 + values_page2
        assert sorted(all_values) == ["A", "B", "C"]

    async def test_get_metadata_field_unique_values_async_with_search_term(self, populated_store):
        """Test getting unique values filtered by search term"""
        # Search for documents containing "Doc 1"
        values, total = await populated_store.get_metadata_field_unique_values_async(
            "category", search_term="Doc 1", from_=0, size=10
        )
        assert values == ["A"]  # Only Doc 1 has category A
        assert total == 1

    async def test_get_metadata_field_unique_values_async_missing_field(self, populated_store):
        """Test getting unique values for non-existent field"""
        values, total = await populated_store.get_metadata_field_unique_values_async(
            "nonexistent_field", from_=0, size=10
        )
        assert values == []
        assert total == 0
