# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document, Pipeline
from haystack.components.rankers import MaxSimRanker
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.qdrant import QdrantMultiVectorRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


def test_maxsim_matches_local_ranker_and_pipeline_round_trip(tmp_path):
    store = QdrantDocumentStore(path=str(tmp_path), multi_vector_embedding_dim=2, progress_bar=False)
    documents = [
        Document(id="best", meta={"group": "a"}, multi_vector_embedding=[[1.0, 0.0], [0.0, 1.0]]),
        Document(id="partial", meta={"group": "b"}, multi_vector_embedding=[[0.5, 0.1]]),
        Document(id="negative", meta={"group": "a"}, multi_vector_embedding=[[-1.0, -2.0]]),
    ]
    query = [[1.0, 0.0], [0.0, 1.0]]
    pipeline = Pipeline()
    pipeline.add_component("retriever", QdrantMultiVectorRetriever(document_store=store, top_k=2))
    try:
        store.write_documents(documents=documents)
        serialized = pipeline.dumps()
    finally:
        store.close()
    restored = Pipeline.loads(serialized)
    retriever = restored.get_component("retriever")
    try:
        result = restored.run({"retriever": {"query_embedding": query, "top_k": 3}})["retriever"]["documents"]
        expected = MaxSimRanker().run(query_embedding=query, documents=documents)["documents"]
        assert [doc.id for doc in result] == [doc.id for doc in expected]
        assert [doc.score for doc in result] == pytest.approx([doc.score for doc in expected])
        assert all(doc.multi_vector_embedding is None for doc in result)
        filtered = retriever.run(
            query_embedding=query,
            filters={"field": "meta.group", "operator": "==", "value": "b"},
            return_embedding=True,
        )["documents"]
        assert len(filtered) == 1
        assert filtered[0].id == "partial"
        assert filtered[0].multi_vector_embedding == [[0.5, 0.1]]
    finally:
        retriever.close()


@pytest.mark.asyncio
async def test_async_maxsim_and_filter_merge():
    store = QdrantDocumentStore(location=":memory:", multi_vector_embedding_dim=2, progress_bar=False)
    retriever = QdrantMultiVectorRetriever(
        document_store=store,
        filters={"field": "meta.group", "operator": "==", "value": "a"},
        filter_policy=FilterPolicy.MERGE,
        return_embedding=True,
    )
    documents = [
        Document(id="a", meta={"group": "a", "keep": True}, multi_vector_embedding=[[1.0, 0.0]]),
        Document(id="b", meta={"group": "b", "keep": True}, multi_vector_embedding=[[2.0, 0.0]]),
        Document(id="c", meta={"group": "a", "keep": False}, multi_vector_embedding=[[3.0, 0.0]]),
    ]
    try:
        await store.write_documents_async(documents=documents)
        result = await retriever.run_async(
            query_embedding=[[1.0, 0.0]], filters={"field": "meta.keep", "operator": "==", "value": True}
        )
        assert [(doc.id, doc.score) for doc in result["documents"]] == [("a", 1.0)]
        assert result["documents"][0].multi_vector_embedding == [[1.0, 0.0]]
        assert len(await store.filter_documents_async()) == 3
    finally:
        await retriever.close_async()
