# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import operator
import sys
import uuid
from unittest import mock

import pytest
from haystack.dataclasses import Document

from haystack_integrations.document_stores.chroma import ChromaDocumentStore


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=("We do not run the Chroma server on Windows and async is only supported with HTTP connections"),
)
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

        for doc_received, doc_expected in zip(received, expected):
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

    @pytest.mark.integration
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
