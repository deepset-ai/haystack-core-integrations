# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import List
from unittest import mock

import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from haystack.dataclasses import Document
from haystack.testing.document_store import (
    TEST_EMBEDDING_1,
    TEST_EMBEDDING_2,
    _random_embeddings,
)

from haystack_integrations.document_stores.chroma import ChromaDocumentStore


class _TestEmbeddingFunction(EmbeddingFunction):
    """
    Chroma lets you provide custom functions to compute embeddings,
    we use this feature to provide a fake algorithm returning random
    vectors in unit tests.
    """

    def __call__(self, input: Documents) -> Embeddings:  # noqa - chroma will inspect the signature, it must match
        # embed the documents somehow
        return [np.random.default_rng().uniform(-1, 1, 768).tolist()]


class TestDocumentStoreAsync:
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self) -> ChromaDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        with mock.patch(
            "haystack_integrations.document_stores.chroma.document_store.get_embedding_function"
        ) as get_func:
            get_func.return_value = _TestEmbeddingFunction()
            return ChromaDocumentStore(
                embedding_function="test_function",
                collection_name=str(uuid.uuid1()),
                host="localhost",
                port=8000,
            )

    @pytest.fixture
    def filterable_docs(self) -> List[Document]:
        """
        This fixture has been copied from haystack/testing/document_store.py and modified to
        remove the documents that don't have textual content, as Chroma does not support writing them.
        """
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "100",
                        "chapter": "intro",
                        "number": 2,
                        "date": "1969-07-21T20:17:40",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "123",
                        "chapter": "abstract",
                        "number": -2,
                        "date": "1972-12-11T19:54:58",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Foobar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "90",
                        "chapter": "conclusion",
                        "number": -10,
                        "date": "1989-11-09T17:53:00",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"Document {i} without embedding",
                    meta={
                        "name": f"name_{i}",
                        "no_embedding": True,
                        "chapter": "conclusion",
                    },
                )
            )
            documents.append(
                Document(
                    content=f"Doc {i} with zeros emb",
                    meta={"name": "zeros_doc"},
                    embedding=TEST_EMBEDDING_1,
                )
            )
            documents.append(
                Document(
                    content=f"Doc {i} with ones emb",
                    meta={"name": "ones_doc"},
                    embedding=TEST_EMBEDDING_2,
                )
            )
        return documents

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store: ChromaDocumentStore):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

    @pytest.mark.asyncio
    async def test_delete_documents_async(self, document_store: ChromaDocumentStore):
        """Test delete_documents() normal behaviour."""
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0

    @pytest.mark.asyncio
    async def test_count_empty_async(self, document_store: ChromaDocumentStore):
        """Test count is zero for an empty document store"""
        assert await document_store.count_documents_async() == 0

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_no_filters_async(self, document_store):
        """Test filter_documents() with empty filters"""
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [])
        self.assert_documents_are_equal(await document_store.filter_documents_async(filters={}), [])
        docs = [Document(content="test doc")]
        await document_store.write_documents_async(docs)
        self.assert_documents_are_equal(await document_store.filter_documents_async(), docs)
        self.assert_documents_are_equal(await document_store.filter_documents_async(filters={}), docs)

    @pytest.mark.asyncio
    async def test_comparison_equal_async(self, document_store, filterable_docs):
        """Test filter_documents() with == comparator"""
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") == 100])

    @pytest.mark.asyncio
    async def test_search_async(self, document_store: ChromaDocumentStore):
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
        result_empty_filters = await document_store.search_async(["Third"], filters={}, top_k=1)
        assert result == result_empty_filters
