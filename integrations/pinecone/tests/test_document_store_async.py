# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import SentenceWindowRetriever

from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set")
class TestDocumentStoreAsync:
    async def test_write_documents(self, document_store_async: PineconeDocumentStore):
        docs = [Document(id="1")]

        assert await document_store_async.write_documents_async(docs) == 1

    async def test_write_documents_invalid_input(self, document_store_async: PineconeDocumentStore):
        """Test write_documents() fails when providing unexpected input."""
        with pytest.raises(ValueError):
            await document_store_async.write_documents_async(["not a document for sure"])  # type: ignore
        with pytest.raises(ValueError):
            await document_store_async.write_documents_async("not a list actually")  # type: ignore

    async def test_count_documents(self, document_store_async: PineconeDocumentStore):
        await document_store_async.write_documents_async(
            [
                Document(content="test doc 1"),
                Document(content="test doc 2"),
                Document(content="test doc 3"),
            ]
        )
        assert await document_store_async.count_documents_async() == 3

    async def test_filter_documents(self, document_store_async: PineconeDocumentStore):
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
        await document_store_async.write_documents_async(filterable_docs)
        result = await document_store_async.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )

        assert result == [d for d in filterable_docs if d.meta.get("number") == 100]

    async def test_delete_documents(self, document_store_async: PineconeDocumentStore):
        doc = Document(content="test doc")
        await document_store_async.write_documents_async([doc])
        assert await document_store_async.count_documents_async() == 1

        await document_store_async.delete_documents_async([doc.id])
        assert await document_store_async.count_documents_async() == 0

    async def test_delete_all_documents_async(self, document_store_async: PineconeDocumentStore):
        docs = [Document(content="first doc"), Document(content="second doc")]
        await document_store_async.write_documents_async(docs)
        assert await document_store_async.count_documents_async() == 2

        await document_store_async.delete_all_documents_async()
        assert await document_store_async.count_documents_async() == 0

    async def test_delete_all_documents_async_empty_collection(self, document_store_async: PineconeDocumentStore):
        assert await document_store_async.count_documents_async() == 0
        await document_store_async.delete_all_documents_async()
        assert await document_store_async.count_documents_async() == 0

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

    async def test_delete_by_filter_async(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        await document_store_async.write_documents_async(docs)

        # delete documents with category="A"
        deleted_count = await document_store_async.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert deleted_count == 2
        assert await document_store_async.count_documents_async() == 1

        # only category B remains
        remaining_docs = await document_store_async.filter_documents_async()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    async def test_delete_by_filter_async_no_matches(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
        ]
        await document_store_async.write_documents_async(docs)

        # delete documents with category="C" (no matches)
        deleted_count = await document_store_async.delete_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "C"}
        )
        assert deleted_count == 0
        assert await document_store_async.count_documents_async() == 2

    async def test_update_by_filter_async(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft"}),
        ]
        await document_store_async.write_documents_async(docs)
        assert await document_store_async.count_documents_async() == 3

        # update status for category="A" documents
        updated_count = await document_store_async.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 2

        published_docs = await document_store_async.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["category"] == "A"
            assert doc.meta["status"] == "published"

    async def test_update_by_filter_async_multiple_fields(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft", "priority": "low"}),
            Document(content="Doc 2", meta={"category": "B", "status": "draft", "priority": "low"}),
            Document(content="Doc 3", meta={"category": "A", "status": "draft", "priority": "low"}),
        ]
        await document_store_async.write_documents_async(docs)

        # update multiple fields for category="A" documents
        updated_count = await document_store_async.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            meta={"status": "published", "priority": "high"},
        )
        assert updated_count == 2

        # verify the updates
        published_docs = await document_store_async.filter_documents_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["category"] == "A"
            assert doc.meta["status"] == "published"
            assert doc.meta["priority"] == "high"

    async def test_update_by_filter_async_no_matches(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
        ]
        await document_store_async.write_documents_async(docs)

        # try to update documents with category="C" (no matches)
        updated_count = await document_store_async.update_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "C"}, meta={"status": "published"}
        )
        assert updated_count == 0
        assert await document_store_async.count_documents_async() == 2

    async def test_count_documents_by_filter_async(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
            Document(content="Doc 2", meta={"category": "B", "status": "published"}),
            Document(content="Doc 3", meta={"category": "A", "status": "published"}),
            Document(content="Doc 4", meta={"category": "A", "status": "draft"}),
        ]
        await document_store_async.write_documents_async(docs)

        # Count documents with category="A"
        count = await document_store_async.count_documents_by_filter_async(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count == 3

        # Count documents with status="published"
        count = await document_store_async.count_documents_by_filter_async(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert count == 2

    async def test_count_unique_metadata_by_filter_async(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "author": "Alice", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "author": "Bob", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "author": "Alice", "priority": 1}),
            Document(content="Doc 4", meta={"category": "C", "author": "Charlie", "priority": 3}),
        ]
        await document_store_async.write_documents_async(docs)

        # Count unique values without filter
        counts = await document_store_async.count_unique_metadata_by_filter_async(
            filters={}, metadata_fields=["category", "author"]
        )
        assert counts["category"] == 3  # A, B, C
        assert counts["author"] == 3  # Alice, Bob, Charlie

    async def test_get_metadata_fields_info_async(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(
                content="Doc 1",
                meta={"category": "A", "priority": 1, "is_published": True, "tags": ["tag1", "tag2"]},
            ),
            Document(content="Doc 2", meta={"category": "B", "priority": 2, "is_published": False}),
        ]
        await document_store_async.write_documents_async(docs)

        field_info = await document_store_async.get_metadata_fields_info_async()

        # Check content field
        assert "content" in field_info
        assert field_info["content"]["type"] == "text"

        # Check metadata fields
        assert field_info["category"]["type"] == "keyword"
        assert field_info["priority"]["type"] == "long"
        assert field_info["is_published"]["type"] == "boolean"
        assert field_info["tags"]["type"] == "keyword"

    async def test_get_metadata_field_min_max_async(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"priority": 1, "score": 85.5, "active": True, "category": "Zebra"}),
            Document(content="Doc 2", meta={"priority": 5, "score": 92.3, "active": False, "category": "Alpha"}),
            Document(content="Doc 3", meta={"priority": 3, "score": 78.9, "active": True, "category": "Beta"}),
        ]
        await document_store_async.write_documents_async(docs)

        # Get min/max for numeric field (int)
        min_max = await document_store_async.get_metadata_field_min_max_async("priority")
        assert min_max["min"] == 1
        assert min_max["max"] == 5

        # Get min/max for numeric field (float)
        min_max = await document_store_async.get_metadata_field_min_max_async("score")
        assert min_max["min"] == 78.9
        assert min_max["max"] == 92.3

        # Get min/max for boolean field
        min_max = await document_store_async.get_metadata_field_min_max_async("active")
        assert min_max["min"] is False
        assert min_max["max"] is True

        # Get min/max for string field (alphabetical ordering)
        min_max = await document_store_async.get_metadata_field_min_max_async("category")
        assert min_max["min"] == "Alpha"
        assert min_max["max"] == "Zebra"

    async def test_get_metadata_field_unique_values_async(self, document_store_async: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha"}),
            Document(content="Doc 2", meta={"category": "Beta"}),
            Document(content="Doc 3", meta={"category": "Gamma"}),
            Document(content="Doc 4", meta={"category": "Alpha"}),
            Document(content="Doc 5", meta={"category": "Delta"}),
        ]
        await document_store_async.write_documents_async(docs)

        # Get all unique values
        values, total = await document_store_async.get_metadata_field_unique_values_async("category", size=10)
        assert total == 4  # Alpha, Beta, Delta, Gamma
        assert len(values) == 4
        assert set(values) == {"Alpha", "Beta", "Delta", "Gamma"}

        # Test pagination
        values, total = await document_store_async.get_metadata_field_unique_values_async("category", from_=0, size=2)
        assert total == 4
        assert len(values) == 2  # First 2 values

        # Test search term
        values, total = await document_store_async.get_metadata_field_unique_values_async(
            "category", search_term="ta", size=10
        )
        assert total == 2  # Beta and Delta contain "ta"
        assert set(values) == {"Beta", "Delta"}
