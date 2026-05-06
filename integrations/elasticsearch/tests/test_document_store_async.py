# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
import pytest_asyncio
from haystack.dataclasses.document import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


@pytest.mark.integration
class TestElasticsearchDocumentStoreAsync(
    CountDocumentsAsyncTest,
    WriteDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    FilterDocumentsAsyncTest,
    UpdateByFilterAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    @pytest_asyncio.fixture
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

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        # filter_documents_async() returns Documents with score populated; strip it before comparing
        received = [dataclasses.replace(doc, score=None) for doc in received]
        super().assert_documents_are_equal(received, expected)

    @pytest.mark.asyncio
    async def test_count_not_empty_async(self, document_store):
        # Override needed: base class uses @staticmethod which breaks fixture injection
        await document_store.write_documents_async(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store):
        docs = [Document(id="1", content="test")]
        assert await document_store.write_documents_async(docs) == 1
        assert await document_store.count_documents_async() == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs, policy=DuplicatePolicy.FAIL)

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
        assert "but `sparse_vector_field` is not configured" in caplog.text

        results = await document_store.filter_documents_async()
        assert len(results) == 1
        assert results[0].id == "1"
        assert not hasattr(results[0], "sparse_embedding") or results[0].sparse_embedding is None

    @pytest.mark.asyncio
    async def test_write_documents_async_with_sparse_vectors(self):
        """Test write_documents with document containing sparse_embedding field"""
        store = ElasticsearchDocumentStore(
            hosts=["http://localhost:9200"], index="test_async_sparse", sparse_vector_field="sparse_vec"
        )
        await store.async_client.options(ignore_status=[400, 404]).indices.delete(index="test_async_sparse")

        doc = Document(id="1", content="test", sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.5]))
        await store.write_documents_async([doc])

        # check ES natively
        raw_doc = await store.async_client.get(index="test_async_sparse", id="1")
        assert raw_doc["_source"]["sparse_vec"] == {"0": 0.5, "1": 0.5}

        # check retrieval
        results = await store.filter_documents_async()
        assert len(results) == 1
        assert results[0].sparse_embedding is not None
        assert results[0].sparse_embedding.indices == [0, 1]
        assert results[0].sparse_embedding.values == [0.5, 0.5]

        await store.async_client.indices.delete(index="test_async_sparse")

    @pytest.mark.asyncio
    async def test_write_documents_async_with_non_contiguous_sparse_indices(self):
        store = ElasticsearchDocumentStore(
            hosts=["http://localhost:9200"],
            index="test_async_sparse_noncontiguous",
            sparse_vector_field="sparse_vec",
        )
        await store.async_client.options(ignore_status=[400, 404]).indices.delete(
            index="test_async_sparse_noncontiguous"
        )

        doc = Document(
            id="1", content="test", sparse_embedding=SparseEmbedding(indices=[100, 5, 42], values=[0.1, 0.9, 0.5])
        )
        await store.write_documents_async([doc])

        results = await store.filter_documents_async()
        assert len(results) == 1
        assert results[0].sparse_embedding is not None
        assert results[0].sparse_embedding.indices == [5, 42, 100]
        assert results[0].sparse_embedding.values == [0.9, 0.5, 0.1]

        await store.async_client.indices.delete(index="test_async_sparse_noncontiguous")

    @pytest.mark.asyncio
    async def test_write_documents_async_mixed_sparse_and_non_sparse(self):
        store = ElasticsearchDocumentStore(
            hosts=["http://localhost:9200"], index="test_async_sparse_mixed", sparse_vector_field="sparse_vec"
        )
        await store.async_client.options(ignore_status=[400, 404]).indices.delete(index="test_async_sparse_mixed")

        docs = [
            Document(
                id="1", content="with sparse", sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.5, 0.5])
            ),
            Document(id="2", content="without sparse"),
        ]
        await store.write_documents_async(docs)

        results = sorted(await store.filter_documents_async(), key=lambda d: d.id)
        assert len(results) == 2
        assert results[0].sparse_embedding is not None
        assert results[0].sparse_embedding.indices == [0, 1]
        assert results[1].sparse_embedding is None

        await store.async_client.indices.delete(index="test_async_sparse_mixed")

    @pytest.mark.asyncio
    async def test_delete_all_documents_async_index_recreation(self, document_store):
        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)

        assert document_store._async_client is not None
        index_info_before = await document_store._async_client.indices.get(index=document_store._index)
        mappings_before = index_info_before[document_store._index]["mappings"]
        settings_before = index_info_before[document_store._index]["settings"]

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

        results = await document_store._bm25_retrieval_async(
            "programming",
            filters={"field": "type", "operator": "==", "value": "functional"},
            top_k=1,
            scale_score=True,
        )
        assert len(results) == 1
        assert "functional" in results[0].content
        assert 0 <= results[0].score <= 1

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
    async def test_sparse_vector_retrieval_async_requires_sparse_vector_field(self, document_store):
        with pytest.raises(ValueError, match="sparse_vector_field must be set for sparse vector retrieval"):
            await document_store._sparse_vector_retrieval_async(
                query_sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 1.0])
            )

    @pytest.mark.asyncio
    async def test_query_sql_async(self, document_store: ElasticsearchDocumentStore):
        docs = [
            Document(content="Python programming", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Java programming", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Python scripting", meta={"category": "A", "status": "inactive", "priority": 3}),
            Document(content="JavaScript development", meta={"category": "C", "status": "active", "priority": 1}),
        ]
        await document_store.write_documents_async(docs)

        # SQL query returns raw JSON response from Elasticsearch SQL API
        sql_query = (
            f'SELECT content, category, status, priority FROM "{document_store._index}" '  # noqa: S608
            f"WHERE category = 'A' ORDER BY priority"
        )
        result = await document_store._query_sql_async(sql_query)

        # Verify raw JSON response structure
        assert isinstance(result, dict)
        assert "columns" in result
        assert "rows" in result

        # Verify we got 2 rows (documents with category A)
        assert len(result["rows"]) == 2

        # Verify column structure
        column_names = [col["name"] for col in result["columns"]]
        assert "content" in column_names
        assert "category" in column_names

    @pytest.mark.asyncio
    async def test_query_sql_async_with_fetch_size(self, document_store: ElasticsearchDocumentStore):
        """Test async SQL query with fetch_size parameter"""
        docs = [Document(content=f"Document {i}", meta={"category": "A", "index": i}) for i in range(15)]
        await document_store.write_documents_async(docs)

        sql_query = (
            f'SELECT content, category FROM "{document_store._index}" '  # noqa: S608
            f"WHERE category = 'A'"
        )

        # Test with fetch_size
        result = await document_store._query_sql_async(sql_query, fetch_size=5)

        # Should return raw JSON response
        assert isinstance(result, dict)
        assert "columns" in result
        assert "rows" in result

    @pytest.mark.asyncio
    async def test_query_sql_async_error_handling(self, document_store: ElasticsearchDocumentStore):
        """Test error handling for invalid SQL queries"""
        invalid_query = "SELECT * FROM non_existent_index"
        with pytest.raises(DocumentStoreError, match="Failed to execute SQL query"):
            await document_store._query_sql_async(invalid_query)
