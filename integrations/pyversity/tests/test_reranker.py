# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from haystack import Document, Pipeline
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pyversity import Strategy

from haystack_integrations.components.rankers.pyversity import PyversityReranker

EMBEDDING_DIM = 16
RNG = np.random.default_rng(42)


@pytest.fixture()
def documents():
    docs = []
    for i in range(20):
        vec = RNG.standard_normal(EMBEDDING_DIM)
        vec /= np.linalg.norm(vec)
        docs.append(Document(content=f"Document {i}", embedding=vec.tolist()))
    return docs


@pytest.fixture()
def query_embedding():
    vec = RNG.standard_normal(EMBEDDING_DIM)
    return (vec / np.linalg.norm(vec)).tolist()


@pytest.fixture()
def pipeline(documents, request):
    strategy = getattr(request, "param", Strategy.MMR)
    store = InMemoryDocumentStore()
    store.write_documents(documents)

    p = Pipeline()
    p.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(document_store=store, top_k=15, return_embedding=True),
    )
    p.add_component("reranker", PyversityReranker(k=5, strategy=strategy, diversity=0.7))
    p.connect("retriever.documents", "reranker.documents")
    return p


class TestPyversityReranker:
    def test_init_defaults(self):
        reranker = PyversityReranker(k=5)
        assert reranker._k == 5
        assert reranker._strategy == Strategy.DPP
        assert reranker._diversity == 0.5

    def test_init_custom_params(self):
        reranker = PyversityReranker(k=10, strategy=Strategy.MMR, diversity=0.3)
        assert reranker._k == 10
        assert reranker._strategy == Strategy.MMR
        assert reranker._diversity == 0.3

    def test_init_invalid_k_zero(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            PyversityReranker(k=0)

    def test_init_invalid_k_negative(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            PyversityReranker(k=-1)

    def test_init_invalid_diversity_above_one(self):
        with pytest.raises(ValueError, match="diversity must be in"):
            PyversityReranker(k=5, diversity=1.1)

    def test_init_invalid_diversity_below_zero(self):
        with pytest.raises(ValueError, match="diversity must be in"):
            PyversityReranker(k=5, diversity=-0.1)

    def test_run_empty_documents(self):
        reranker = PyversityReranker(k=5)
        result = reranker.run(documents=[])
        assert result == {"documents": []}

    def test_run_skips_docs_without_score(self):
        rng = np.random.default_rng(0)
        docs = [
            Document(content="has score and embedding", score=0.9, embedding=rng.standard_normal(4).tolist()),
            Document(content="missing score", embedding=rng.standard_normal(4).tolist()),
        ]
        reranker = PyversityReranker(k=5)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) == 1

    def test_run_skips_docs_without_embedding(self):
        docs = [
            Document(content="has score and embedding", score=0.9, embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="missing embedding", score=0.5),
        ]
        reranker = PyversityReranker(k=5)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) == 1

    def test_run_all_docs_invalid_returns_empty(self):
        docs = [
            Document(content="no score", embedding=[0.1, 0.2]),
            Document(content="no embedding", score=0.5),
        ]
        reranker = PyversityReranker(k=5)
        result = reranker.run(documents=docs)
        assert result == {"documents": []}

    def test_run_returns_at_most_k_documents(self):
        rng = np.random.default_rng(1)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(10)
        ]
        reranker = PyversityReranker(k=3)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) <= 3

    def test_run_returns_fewer_than_k_if_not_enough_valid(self):
        rng = np.random.default_rng(2)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(2)
        ]
        reranker = PyversityReranker(k=5)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) == 2


@pytest.mark.parametrize(
    "pipeline",
    [Strategy.DPP, Strategy.MMR, Strategy.MSD, Strategy.SSD, Strategy.COVER],
    indirect=True,
)
def test_pipeline_returns_k_documents(pipeline, query_embedding):
    result = pipeline.run({"retriever": {"query_embedding": query_embedding}})
    assert len(result["reranker"]["documents"]) == 5
