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

from haystack_integrations.document_stores.chroma import ChromaDocumentStore


@pytest.mark.asyncio
class TestDocumentStoreAsyncUnit:
    async def test_ensure_initialized_async_requires_host_and_port(self):
        store = ChromaDocumentStore()
        with pytest.raises(ValueError, match="Async support"):
            await store._ensure_initialized_async()

    async def test_ensure_initialized_async_invalid_client_settings_raises(self):
        with mock.patch(
            "haystack_integrations.document_stores.chroma.document_store.Settings",
            side_effect=ValueError("bad setting"),
        ):
            store = ChromaDocumentStore(host="localhost", port=8000, client_settings={"foo": "bar"})
            with pytest.raises(ValueError, match="Invalid client_settings"):
                await store._ensure_initialized_async()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="We do not run the Chroma server on Windows and async is only supported with HTTP connections",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestDocumentStoreAsync(
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
):
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

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
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

    async def test_count_not_empty_async(self, document_store: ChromaDocumentStore):
        """Test count is greater than zero if the document store contains documents.

        Override: mixin method released with Haystack 2.28.0 is missing @staticmethod decorator.
        """
        await document_store.write_documents_async(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert await document_store.count_documents_async() == 3

    async def test_write_documents_async(self, document_store: ChromaDocumentStore):
        """Override: mixin's base test raises requires implementers to override."""
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

    # ── Chroma-specific tests (not covered by mixins) ──────────────────────

    async def test_client_settings_applied_async(self):
        store = ChromaDocumentStore(
            host="localhost",
            port=8000,
            client_settings={"anonymized_telemetry": False},
            collection_name=f"{uuid.uuid1()}-async-settings",
        )
        await store._ensure_initialized_async()
        assert store._async_client.get_settings().anonymized_telemetry is False

    async def test_search_async(self):
        document_store = ChromaDocumentStore(host="localhost", port=8000, collection_name="my_custom_collection")

        documents = [
            Document(content="First document", meta={"author": "Author1"}),
            Document(content="Second document"),
            Document(content="Third document", meta={"author": "Author2"}),
            Document(content="Fourth document"),
        ]
        await document_store.write_documents_async(documents)
        result = await document_store.search_async(["Third"], top_k=1)

        assert len(result) == 1
        doc = result[0][0]
        assert doc.content == "Third document"
        assert doc.meta == {"author": "Author2"}
        assert doc.embedding
        assert isinstance(doc.embedding, list)
        assert all(isinstance(el, float) for el in doc.embedding)

        result_empty_filters = document_store.search(["Third"], filters={}, top_k=1)
        assert result == result_empty_filters

    @pytest.mark.asyncio
    async def test_delete_all_documents_index_recreation(self, document_store: ChromaDocumentStore):
        docs = [
            Document(id="1", content="First document", meta={"category": "test"}),
            Document(id="2", content="Second document", meta={"category": "test"}),
            Document(id="3", content="Third document", meta={"category": "other"}),
        ]
        await document_store.write_documents_async(docs)

        config_before = await document_store._async_collection.get(document_store._collection_name)

        await document_store.delete_all_documents_async(recreate_index=True)
        assert await document_store.count_documents_async() == 0

        config_after = await document_store._async_collection.get(document_store._collection_name)
        assert config_before == config_after

        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

    async def test_search_embeddings_async(self, document_store: ChromaDocumentStore):
        query_embedding = TEST_EMBEDDING_1
        documents = [
            Document(content="First document", embedding=TEST_EMBEDDING_1, meta={"author": "Author1"}),
            Document(content="Second document", embedding=[0.1] * len(TEST_EMBEDDING_1)),
            Document(content="Third document", embedding=TEST_EMBEDDING_1, meta={"author": "Author2"}),
        ]
        await document_store.write_documents_async(documents)
        result = await document_store.search_embeddings_async([query_embedding], top_k=2)

        assert len(result) == 1
        assert len(result[0]) == 2
        assert all(doc.embedding == pytest.approx(TEST_EMBEDDING_1) for doc in result[0])
        assert all(doc.score is not None for doc in result[0])

        result_empty_filters = await document_store.search_embeddings_async([query_embedding], filters={}, top_k=2)
        assert len(result_empty_filters) == 1
        assert len(result_empty_filters[0]) == 2

    # ── Chroma-specific error cases for metadata operations ─────────────────

    async def test_get_metadata_field_min_max_async_string(self, document_store: ChromaDocumentStore):
        """Chroma-specific: min/max for string field returns alphabetical order."""
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "C"}),
        ]
        await document_store.write_documents_async(docs)
        min_max = await document_store.get_metadata_field_min_max_async("category")
        assert min_max["min"] == "A"
        assert min_max["max"] == "C"

    async def test_get_metadata_field_min_max_async_missing_field(self, document_store: ChromaDocumentStore):
        """Chroma-specific: min/max for non-existent field returns None."""
        docs = [Document(content="Doc 1", meta={"category": "A"})]
        await document_store.write_documents_async(docs)
        min_max = await document_store.get_metadata_field_min_max_async("nonexistent_field")
        assert min_max["min"] is None
        assert min_max["max"] is None

    async def test_get_metadata_field_unique_values_async_missing_field(self, document_store: ChromaDocumentStore):
        """Chroma-specific: unique values for non-existent field returns empty."""
        docs = [Document(content="Doc 1", meta={"category": "A"})]
        await document_store.write_documents_async(docs)
        values, total = await document_store.get_metadata_field_unique_values_async(
            "nonexistent_field", from_=0, size=10
        )
        assert values == []
        assert total == 0
