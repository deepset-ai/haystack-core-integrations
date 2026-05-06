# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.testing.document_store import create_filterable_docs
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
from numpy import array as np_array
from numpy import array_equal as np_array_equal
from numpy import float32 as np_float32

from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.document_stores.weaviate.document_store import DOCUMENT_COLLECTION_PROPERTIES


@pytest.mark.integration
class TestWeaviateDocumentStoreAsync(
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
    async def document_store(self, request) -> AsyncGenerator[WeaviateDocumentStore, None, None]:
        collection_settings = {
            "class": f"{request.node.name}",
            "invertedIndexConfig": {"indexNullState": True, "stopwords": {"preset": "none"}},
            "properties": [
                *DOCUMENT_COLLECTION_PROPERTIES,
                {"name": "name", "dataType": ["text"]},
                {"name": "page", "dataType": ["text"]},
                {"name": "chapter", "dataType": ["text"]},
                {"name": "category", "dataType": ["text"]},
                {"name": "status", "dataType": ["text"]},
                {"name": "number", "dataType": ["int"]},
                {"name": "date", "dataType": ["date"]},
                {"name": "no_embedding", "dataType": ["boolean"]},
                {"name": "priority", "dataType": ["int"]},
                {"name": "age", "dataType": ["int"]},
                {"name": "rating", "dataType": ["number"]},
                {"name": "year", "dataType": ["int"]},
            ],
        }
        store = WeaviateDocumentStore(
            url="http://localhost:8080",
            collection_settings=collection_settings,
        )
        yield store
        await (await store.async_client).collections.delete(collection_settings["class"])
        await store.close_async()

    @pytest.mark.asyncio
    async def test_get_metadata_fields_info_empty_collection_async(self, document_store):
        # Override: Weaviate derives field info from the collection schema, not from stored
        # documents. Schema-defined fields are visible even on an empty collection, so the
        # result is never {} as the standard interface expects.
        assert await document_store.count_documents_async() == 0
        fields_info = await document_store.get_metadata_fields_info_async()
        assert "category" in fields_info
        assert "status" in fields_info

    @pytest.fixture
    def filterable_docs(self) -> list[Document]:
        """
        Weaviate requires RFC 3339 date strings; the default fixture uses ISO 8601.
        """
        documents = create_filterable_docs()
        for i in range(len(documents)):
            if date := documents[i].meta.get("date"):
                documents[i].meta["date"] = f"{date}Z"
        return documents

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        assert len(received) == len(expected)
        received = sorted(received, key=lambda doc: doc.id)
        expected = sorted(expected, key=lambda doc: doc.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            received_doc_dict = dataclasses.replace(received_doc, score=None).to_dict(flatten=False)
            expected_doc_dict = expected_doc.to_dict(flatten=False)

            # Weaviate stores embeddings with lower precision floats so we handle that here.
            assert np_array_equal(
                np_array(received_doc_dict.pop("embedding", None), dtype=np_float32),
                np_array(expected_doc_dict.pop("embedding", None), dtype=np_float32),
                equal_nan=True,
            )

            received_meta = received_doc_dict.pop("meta", None)
            expected_meta = expected_doc_dict.pop("meta", None)

            assert received_doc_dict == expected_doc_dict

            # If a meta field is not set in a saved document, it will be None when retrieved
            # from Weaviate so we need to handle that.
            meta_keys = set(received_meta.keys()).union(set(expected_meta.keys()))
            for key in meta_keys:
                assert received_meta.get(key) == expected_meta.get(key)

    async def test_write_documents_async(self, document_store: WeaviateDocumentStore) -> None:
        # Override: mixin raises NotImplementedError and requires each store to define its own
        # default-policy behaviour. Weaviate's default overwrites existing documents.
        doc = Document(content="test doc")
        assert await document_store.write_documents_async([doc]) == 1
        assert await document_store.count_documents_async() == 1

        doc = dataclasses.replace(doc, content="test doc 2")
        assert await document_store.write_documents_async([doc]) == 1
        assert await document_store.count_documents_async() == 1

    async def test_count_not_empty_async(self, document_store: WeaviateDocumentStore) -> None:
        # Override: the mixin defines this without `self`, which breaks under asyncio_mode=auto.
        # Body is identical to the mixin's version.
        await document_store.write_documents_async(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_close_async(self, document_store: WeaviateDocumentStore) -> None:
        # Initialise client and collection
        assert await document_store.async_client is not None
        assert await document_store.async_collection is not None

        await document_store.close_async()

        assert document_store._async_client is None
        assert document_store._async_collection is None

        # Initialise client and collection, then test it stills works after reopening
        assert await document_store.async_client is not None
        assert await document_store.async_collection is not None

        assert await document_store.count_documents_async() == 0

    @pytest.mark.asyncio
    async def test_filter_documents_with_blob_data_async(
        self, document_store: WeaviateDocumentStore, test_files_path: Path
    ) -> None:
        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        doc = Document(content="test doc", blob=image)
        assert await document_store.write_documents_async([doc]) == 1

        docs = await document_store.filter_documents_async()

        assert len(docs) == 1
        assert docs[0].blob == image

    @pytest.mark.asyncio
    async def test_filter_documents_over_default_limit(self, document_store: WeaviateDocumentStore) -> None:
        docs = []
        for index in range(10000):
            docs.append(Document(content="This is some content", meta={"index": index}))
        await document_store.write_documents_async(docs)
        with pytest.raises(DocumentStoreError):
            await document_store.filter_documents_async(
                {"field": "content", "operator": "==", "value": "This is some content"}
            )

    @pytest.mark.asyncio
    async def test_write_documents_with_blob_data_async(
        self, document_store: WeaviateDocumentStore, test_files_path: Path
    ) -> None:
        image = ByteStream.from_file_path(test_files_path / "robot1.jpg", mime_type="image/jpeg")
        doc = Document(content="test doc", blob=image)
        assert await document_store.write_documents_async([doc]) == 1

    @pytest.mark.asyncio
    async def test_bm25_retrieval_async(self, document_store):
        await document_store.write_documents_async(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Python is an object oriented programming language"),
                Document(content="Rust is a systems programming language"),
            ]
        )

        result = await document_store._bm25_retrieval_async("functional Haskell", top_k=2)

        assert len(result) <= 2
        assert any("Haskell" in doc.content for doc in result)
        assert all(doc.score is not None and doc.score > 0.0 for doc in result)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_async_with_filters(self, document_store):
        await document_store.write_documents_async(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Python is an object oriented programming language"),
            ]
        )

        filters = {"field": "content", "operator": "==", "value": "Haskell"}
        result = await document_store._bm25_retrieval_async("functional", filters=filters)

        assert len(result) == 1
        assert "Haskell" in result[0].content

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async(self, document_store):
        await document_store.write_documents_async(
            [
                Document(content="The document", embedding=[1.0, 1.0, 1.0, 1.0]),
                Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
                Document(content="Yet another document", embedding=[0.00001, 0.00001, 0.00001, 0.00002]),
            ]
        )

        result = await document_store._embedding_retrieval_async(query_embedding=[1.0, 1.0, 1.0, 1.0], top_k=2)

        assert len(result) == 2
        assert result[0].content == "The document"
        assert result[0].score is not None and result[0].score > 0.0

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async_with_filters(self, document_store):
        await document_store.write_documents_async(
            [
                Document(content="The document I want", embedding=[1.0, 1.0, 1.0, 1.0]),
                Document(content="Another document", embedding=[0.8, 0.8, 0.8, 1.0]),
            ]
        )

        filters = {"field": "content", "operator": "==", "value": "The document I want"}
        result = await document_store._embedding_retrieval_async(query_embedding=[1.0, 1.0, 1.0, 1.0], filters=filters)

        assert len(result) == 1
        assert result[0].content == "The document I want"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async_distance_and_certainty_error(self, document_store):
        with pytest.raises(ValueError, match="Can't use 'distance' and 'certainty' parameters together"):
            await document_store._embedding_retrieval_async(
                query_embedding=[1.0, 1.0, 1.0, 1.0], distance=0.5, certainty=0.8
            )

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_async(self, document_store):
        await document_store.write_documents_async(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
                Document(content="Python is an object oriented language", embedding=[0.1, 0.2, 0.8, 0.9]),
            ]
        )

        result = await document_store._hybrid_retrieval_async(
            query="functional Haskell",
            query_embedding=[1.0, 0.8, 0.2, 0.1],
            top_k=2,
        )

        assert len(result) <= 2
        assert result[0].content == "Haskell is a functional programming language"
        assert result[0].score is not None and result[0].score > 0.0

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_async_with_filters(self, document_store):
        await document_store.write_documents_async(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Lisp is a functional programming language", embedding=[0.9, 0.7, 0.3, 0.2]),
            ]
        )

        filters = {"field": "content", "operator": "==", "value": "Haskell is a functional programming language"}
        result = await document_store._hybrid_retrieval_async(
            query="functional",
            query_embedding=[1.0, 0.8, 0.2, 0.1],
            filters=filters,
        )

        assert len(result) == 1
        assert result[0].content == "Haskell is a functional programming language"

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_async_with_alpha(self, document_store):
        await document_store.write_documents_async(
            [
                Document(content="Haskell is a functional programming language", embedding=[1.0, 0.8, 0.2, 0.1]),
                Document(content="Python is an object oriented language", embedding=[0.1, 0.2, 0.8, 0.9]),
            ]
        )

        # Test with alpha=0.0 (pure BM25)
        result_bm25 = await document_store._hybrid_retrieval_async(
            query="functional",
            query_embedding=[1.0, 0.8, 0.2, 0.1],
            alpha=0.0,
        )
        assert len(result_bm25) > 0
        assert result_bm25[0].score > 0.0

        # Test with alpha=1.0 (pure vector search)
        result_vector = await document_store._hybrid_retrieval_async(
            query="functional",
            query_embedding=[1.0, 0.8, 0.2, 0.1],
            alpha=1.0,
        )
        assert len(result_vector) > 0
        assert result_vector[0].score > 0.0

    @pytest.mark.asyncio
    async def test_update_by_filter_async_with_pagination(self, document_store, monkeypatch):
        # Reduce DEFAULT_QUERY_LIMIT to test pagination without creating 10000+ documents
        monkeypatch.setattr("haystack_integrations.document_stores.weaviate.document_store.DEFAULT_QUERY_LIMIT", 100)

        docs = []
        for index in range(250):
            docs.append(
                Document(
                    content="This is some content",
                    meta={"index": index, "status": "draft", "category": "test"},
                )
            )
        await document_store.write_documents_async(docs)

        # update all documents should trigger pagination (3 pages)
        updated_count = await document_store.update_by_filter_async(
            filters={"field": "category", "operator": "==", "value": "test"},
            meta={"status": "published"},
        )
        assert updated_count == 250

        published_docs = await document_store.filter_documents_async(
            filters={"field": "status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 250
        for doc in published_docs:
            assert doc.meta["category"] == "test"
            assert doc.meta["status"] == "published"
            assert "index" in doc.meta
            assert 0 <= doc.meta["index"] < 250

    @pytest.mark.asyncio
    async def test_get_metadata_field_min_max_async_unsupported_type(self, document_store):
        with pytest.raises(ValueError, match="doesn't support min/max aggregation"):
            await document_store.get_metadata_field_min_max_async("category")

    @pytest.mark.asyncio
    async def test_get_metadata_field_min_max_async_field_not_found(self, document_store):
        with pytest.raises(ValueError, match="not found in collection schema"):
            await document_store.get_metadata_field_min_max_async("nonexistent_field")

    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async_with_meta_prefix(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
            Document(content="Doc 2", meta={"category": "TypeB"}),
        ]
        await document_store.write_documents_async(docs)

        result = await document_store.count_unique_metadata_by_filter_async(
            filters={"field": "meta.category", "operator": "in", "value": ["TypeA", "TypeB"]},
            metadata_fields=["meta.category"],
        )
        assert result["category"] == 2

    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async_no_matches(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
        ]
        await document_store.write_documents_async(docs)

        result = await document_store.count_unique_metadata_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "NonExistent"},
            metadata_fields=["category"],
        )
        assert result["category"] == 0

    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async_field_not_found(self, document_store):
        with pytest.raises(ValueError, match="Fields not found in collection schema"):
            await document_store.count_unique_metadata_by_filter_async(
                filters={"field": "meta.category", "operator": "==", "value": "TypeA"},
                metadata_fields=["nonexistent_field"],
            )

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async_with_meta_prefix(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
            Document(content="Doc 2", meta={"category": "TypeB"}),
        ]
        await document_store.write_documents_async(docs)

        values, total_count = await document_store.get_metadata_field_unique_values_async("meta.category")
        assert total_count == 2
        assert set(values) == {"TypeA", "TypeB"}

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async_with_search_term(self, document_store):
        docs = [
            Document(content="Python programming language", meta={"category": "TypeA"}),
            Document(content="Java programming language", meta={"category": "TypeB"}),
            Document(content="Python is great", meta={"category": "TypeC"}),
            Document(content="JavaScript tutorial", meta={"category": "TypeD"}),
        ]
        await document_store.write_documents_async(docs)

        values, total_count = await document_store.get_metadata_field_unique_values_async(
            "category", search_term="Python"
        )
        assert total_count == 2
        assert set(values) == {"TypeA", "TypeC"}

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async_with_pagination(self, document_store):
        docs = [
            Document(content="Doc 1", meta={"category": "TypeA"}),
            Document(content="Doc 2", meta={"category": "TypeB"}),
            Document(content="Doc 3", meta={"category": "TypeC"}),
            Document(content="Doc 4", meta={"category": "TypeD"}),
            Document(content="Doc 5", meta={"category": "TypeE"}),
        ]
        await document_store.write_documents_async(docs)

        values, total_count = await document_store.get_metadata_field_unique_values_async("category", from_=0, size=2)
        assert total_count == 5
        assert len(values) == 2

        values2, total_count2 = await document_store.get_metadata_field_unique_values_async("category", from_=2, size=2)
        assert total_count2 == 5
        assert len(values2) == 2

        assert set(values).isdisjoint(set(values2))

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async_field_not_found(self, document_store):
        with pytest.raises(ValueError, match="not found in collection schema"):
            await document_store.get_metadata_field_unique_values_async("nonexistent_field")

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async_empty_result(self, document_store):
        values, total_count = await document_store.get_metadata_field_unique_values_async("category")
        assert total_count == 0
        assert values == []

    @pytest.mark.asyncio
    async def test_delete_all_documents_excessive_batch_size_async(
        self, document_store: WeaviateDocumentStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that the deletion is not complete if the batch size exceeds the QUERY_MAXIMUM_RESULTS."""
        # assume QUERY_MAXIMUM_RESULTS == 10000 with standard deployment
        docs = [Document(content=str(i)) for i in range(0, 10005)]
        assert await document_store.write_documents_async(docs) == 10005
        with caplog.at_level(logging.WARNING):
            await document_store.delete_all_documents_async(batch_size=20000)
        assert await document_store.count_documents_async() == 5
        assert "Not all documents have been deleted." in caplog.text
