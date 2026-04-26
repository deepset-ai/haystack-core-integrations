# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import pytest_asyncio
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import SentenceWindowRetriever
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

from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set")
class TestDocumentStoreAsync(
    WriteDocumentsAsyncTest,
    CountDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    UpdateByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    @pytest_asyncio.fixture
    async def document_store(self, document_store_async: PineconeDocumentStore):
        return document_store_async

    @pytest.mark.asyncio
    async def test_count_not_empty_async(self, document_store: PineconeDocumentStore):
        # Override: haystack v2.28.0 is missing @staticmethod on this mixin method.
        await document_store.write_documents_async(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert await document_store.count_documents_async() == 3

    async def test_embedding_retrieval(self, document_store_async: PineconeDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = [-0.1] * 384 + [0.1] * 384

        docs = [
            Document(content="Most similar document", embedding=most_similar_embedding),
            Document(content="2nd best document", embedding=second_best_embedding),
            Document(content="Not very similar document", embedding=another_embedding),
        ]

        await document_store_async.write_documents_async(docs)

        results = await document_store_async._embedding_retrieval_async(
            query_embedding=query_embedding, top_k=2, filters={}
        )

        assert len(results) == 2

        # Pinecone does not seem to guarantee the order of the results
        assert "Most similar document" in [result.content for result in results]
        assert "2nd best document" in [result.content for result in results]

    async def test_close(self, document_store_async: PineconeDocumentStore):
        await document_store_async._initialize_async_index()
        assert document_store_async._async_index is not None

        await document_store_async.close_async()
        assert document_store_async._async_index is None

        await document_store_async._initialize_async_index()
        assert document_store_async._async_index is not None
        # test that the index is still usable after closing and reopening
        assert await document_store_async.count_documents_async() == 0

    async def test_sentence_window_retriever(self, document_store_async: PineconeDocumentStore):
        # indexing
        splitter = DocumentSplitter(split_length=10, split_overlap=5, split_by="word")
        text = (
            "Whose woods these are I think I know. His house is in the village though; He will not see me stopping "
            "here To watch his woods fill up with snow."
        )
        docs = splitter.run(documents=[Document(content=text)])

        for idx, doc in enumerate(docs["documents"]):
            if idx == 2:
                doc.embedding = [0.1] * 768
                continue
            doc.embedding = np.random.rand(768).tolist()
        await document_store_async.write_documents_async(docs["documents"])

        # query
        embedding_retriever = PineconeEmbeddingRetriever(document_store=document_store_async)
        query_embedding = [0.1] * 768
        retrieved_doc = await embedding_retriever.run_async(query_embedding=query_embedding, top_k=1, filters={})
        sentence_window_retriever = SentenceWindowRetriever(document_store=document_store_async, window_size=2)
        result = sentence_window_retriever.run(retrieved_documents=[retrieved_doc["documents"][0]])

        assert len(result["context_windows"]) == 1
