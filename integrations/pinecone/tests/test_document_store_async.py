import os
import time

import numpy as np
import pytest
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import SentenceWindowRetriever
from pinecone import PineconeAsyncio

from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set")
async def test_serverless_index_creation_from_scratch(sleep_time):
    # we use a fixed index name to avoid hitting the limit of Pinecone's free tier (max 5 indexes)
    # the index name is defined in the test matrix of the GitHub Actions workflow
    # the default value is provided for local testing
    index_name = os.environ.get("INDEX_NAME", "serverless-test-index")

    client = PineconeAsyncio(api_key=os.environ["PINECONE_API_KEY"])
    try:
        await client.delete_index(name=index_name)
    except Exception:  # noqa S110
        pass

    time.sleep(sleep_time)

    ds = PineconeDocumentStore(
        index=index_name,
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
    )
    # Trigger the connection
    _ = await ds._initialize_async_index()

    index_description = await client.describe_index(name=index_name)
    assert index_description["name"] == index_name
    assert index_description["dimension"] == 30
    assert index_description["metric"] == "euclidean"
    assert index_description["spec"]["serverless"]["region"] == "us-east-1"
    assert index_description["spec"]["serverless"]["cloud"] == "aws"

    try:
        await client.delete_index(name=index_name)
    except Exception:  # noqa S110
        pass


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


    async def test_embedding_retrieval(self, document_store_async: PineconeDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = np.random.rand(768).tolist()

        docs = [
            Document(content="Most similar document", embedding=most_similar_embedding),
            Document(content="2nd best document", embedding=second_best_embedding),
            Document(content="Not very similar document", embedding=another_embedding),
        ]

        await document_store_async.write_documents_async(docs)

        results = await document_store_async._embedding_retrieval_async(query_embedding=query_embedding, top_k=2, filters={})
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

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
