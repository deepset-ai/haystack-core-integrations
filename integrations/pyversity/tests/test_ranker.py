# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from haystack import Document, Pipeline
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pyversity import Strategy

from haystack_integrations.components.rankers.pyversity import PyversityRanker

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
    p.add_component("reranker", PyversityRanker(top_k=5, strategy=strategy, diversity=0.7))
    p.connect("retriever.documents", "reranker.documents")
    return p


class TestPyversityRanker:
    def test_init_defaults(self):
        reranker = PyversityRanker()
        assert reranker.top_k is None
        assert reranker.strategy == Strategy.DPP
        assert reranker.diversity == 0.5

    def test_init_top_k_none(self):
        reranker = PyversityRanker(top_k=None)
        assert reranker.top_k is None

    def test_init_explicit_top_k(self):
        reranker = PyversityRanker(top_k=5)
        assert reranker.top_k == 5

    def test_init_custom_params(self):
        reranker = PyversityRanker(top_k=10, strategy=Strategy.MMR, diversity=0.3)
        assert reranker.top_k == 10
        assert reranker.strategy == Strategy.MMR
        assert reranker.diversity == 0.3

    def test_init_invalid_k_zero(self):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            PyversityRanker(top_k=0)

    def test_init_invalid_k_negative(self):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            PyversityRanker(top_k=-1)

    def test_init_invalid_diversity_above_one(self):
        with pytest.raises(ValueError, match="diversity must be in"):
            PyversityRanker(top_k=5, diversity=1.1)

    def test_init_invalid_diversity_below_zero(self):
        with pytest.raises(ValueError, match="diversity must be in"):
            PyversityRanker(top_k=5, diversity=-0.1)

    def test_run_empty_documents(self):
        reranker = PyversityRanker(top_k=5)
        result = reranker.run(documents=[])
        assert result == {"documents": []}

    def test_run_skips_docs_without_score(self):
        rng = np.random.default_rng(0)
        docs = [
            Document(content="has score and embedding", score=0.9, embedding=rng.standard_normal(4).tolist()),
            Document(content="missing score", embedding=rng.standard_normal(4).tolist()),
        ]
        reranker = PyversityRanker(top_k=5)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) == 1

    def test_run_skips_docs_without_embedding(self):
        docs = [
            Document(content="has score and embedding", score=0.9, embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="missing embedding", score=0.5),
        ]
        reranker = PyversityRanker(top_k=5)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) == 1

    def test_run_all_docs_invalid_returns_empty(self):
        docs = [
            Document(content="no score", embedding=[0.1, 0.2]),
            Document(content="no embedding", score=0.5),
        ]
        reranker = PyversityRanker(top_k=5)
        result = reranker.run(documents=docs)
        assert result == {"documents": []}

    def test_run_returns_at_most_k_documents(self):
        rng = np.random.default_rng(1)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(10)
        ]
        reranker = PyversityRanker(top_k=3)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) <= 3

    def test_run_returns_all_documents_when_top_k_is_none(self):
        rng = np.random.default_rng(3)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(10)
        ]
        reranker = PyversityRanker(top_k=None)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) == 10

    def test_run_returns_fewer_than_k_if_not_enough_valid(self):
        rng = np.random.default_rng(2)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(2)
        ]
        reranker = PyversityRanker(top_k=5)
        result = reranker.run(documents=docs)
        assert len(result["documents"]) == 2

    def test_run_sets_selection_scores_on_output(self):
        rng = np.random.default_rng(7)
        original_scores = [float(i) / 10 for i in range(1, 6)]
        docs = [
            Document(content=f"doc {i}", score=s, embedding=rng.standard_normal(8).tolist())
            for i, s in enumerate(original_scores)
        ]
        reranker = PyversityRanker(top_k=3)
        result = reranker.run(documents=docs)
        for doc in result["documents"]:
            assert doc.score is not None
            assert isinstance(doc.score, float)

    def test_run_does_not_mutate_input_documents(self):
        rng = np.random.default_rng(8)
        original_scores = [float(i) / 10 for i in range(1, 6)]
        docs = [
            Document(content=f"doc {i}", score=s, embedding=rng.standard_normal(8).tolist())
            for i, s in enumerate(original_scores)
        ]
        reranker = PyversityRanker(top_k=3)
        reranker.run(documents=docs)
        for doc, original_score in zip(docs, original_scores, strict=False):  # type: ignore[call-overload]
            assert doc.score == original_score

    def test_to_dict_top_k_none(self):
        reranker = PyversityRanker(top_k=None)
        data = reranker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.pyversity.ranker.PyversityRanker",
            "init_parameters": {
                "top_k": None,
                "strategy": "dpp",
                "diversity": 0.5,
            },
        }

    def test_to_dict_defaults(self):
        reranker = PyversityRanker(top_k=5)
        data = reranker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.pyversity.ranker.PyversityRanker",
            "init_parameters": {
                "top_k": 5,
                "strategy": "dpp",
                "diversity": 0.5,
            },
        }

    def test_to_dict_custom_params(self):
        reranker = PyversityRanker(top_k=10, strategy=Strategy.MMR, diversity=0.3)
        data = reranker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.pyversity.ranker.PyversityRanker",
            "init_parameters": {
                "top_k": 10,
                "strategy": "mmr",
                "diversity": 0.3,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.rankers.pyversity.ranker.PyversityRanker",
            "init_parameters": {
                "top_k": 7,
                "strategy": "mmr",
                "diversity": 0.8,
            },
        }
        reranker = PyversityRanker.from_dict(data)
        assert reranker.top_k == 7
        assert reranker.strategy == Strategy.MMR
        assert reranker.diversity == 0.8

    def test_from_dict_defaults(self):
        data = {
            "type": "haystack_integrations.components.rankers.pyversity.ranker.PyversityRanker",
            "init_parameters": {"top_k": 3},
        }
        reranker = PyversityRanker.from_dict(data)
        assert reranker.top_k == 3
        assert reranker.strategy == Strategy.DPP
        assert reranker.diversity == 0.5

    def test_to_dict_from_dict_roundtrip(self):
        original = PyversityRanker(top_k=4, strategy=Strategy.SSD, diversity=0.2)
        restored = PyversityRanker.from_dict(original.to_dict())
        assert restored.top_k == original.top_k
        assert restored.strategy == original.strategy
        assert restored.diversity == original.diversity

    def test_to_dict_from_dict_roundtrip_top_k_none(self):
        original = PyversityRanker(top_k=None)
        restored = PyversityRanker.from_dict(original.to_dict())
        assert restored.top_k is None

    def test_run_top_k_runtime_overrides_init(self):
        rng = np.random.default_rng(10)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(10)
        ]
        reranker = PyversityRanker(top_k=10)
        result = reranker.run(documents=docs, top_k=3)
        assert len(result["documents"]) == 3

    def test_run_top_k_runtime_none_falls_back_to_init(self):
        rng = np.random.default_rng(11)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(10)
        ]
        reranker = PyversityRanker(top_k=3)
        result = reranker.run(documents=docs, top_k=None)
        assert len(result["documents"]) == 3

    def test_run_top_k_runtime_invalid_zero(self):
        reranker = PyversityRanker()
        docs = [Document(content="doc", score=0.5, embedding=[0.1, 0.2, 0.3, 0.4])]
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            reranker.run(documents=docs, top_k=0)

    def test_run_top_k_runtime_invalid_negative(self):
        reranker = PyversityRanker()
        docs = [Document(content="doc", score=0.5, embedding=[0.1, 0.2, 0.3, 0.4])]
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            reranker.run(documents=docs, top_k=-1)

    def test_run_top_k_runtime_with_init_top_k_none(self):
        rng = np.random.default_rng(12)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(10)
        ]
        reranker = PyversityRanker(top_k=None)
        result = reranker.run(documents=docs, top_k=5)
        assert len(result["documents"]) == 5

    def test_run_diversity_runtime_none_falls_back_to_init(self):
        rng = np.random.default_rng(13)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(5)
        ]
        reranker = PyversityRanker(diversity=0.9)
        result_default = reranker.run(documents=docs)
        result_none = reranker.run(documents=docs, diversity=None)
        assert [d.score for d in result_default["documents"]] == [d.score for d in result_none["documents"]]

    def test_run_diversity_runtime_invalid_above_one(self):
        reranker = PyversityRanker()
        docs = [Document(content="doc", score=0.5, embedding=[0.1, 0.2, 0.3, 0.4])]
        with pytest.raises(ValueError, match="diversity must be in"):
            reranker.run(documents=docs, diversity=1.1)

    def test_run_diversity_runtime_invalid_below_zero(self):
        reranker = PyversityRanker()
        docs = [Document(content="doc", score=0.5, embedding=[0.1, 0.2, 0.3, 0.4])]
        with pytest.raises(ValueError, match="diversity must be in"):
            reranker.run(documents=docs, diversity=-0.1)

    def test_run_both_params_overridden_at_runtime(self):
        rng = np.random.default_rng(14)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(10)
        ]
        reranker = PyversityRanker(top_k=10, diversity=0.9)
        result = reranker.run(documents=docs, top_k=4, diversity=0.1)
        assert len(result["documents"]) == 4

    def test_run_strategy_runtime_overrides_init(self):
        rng = np.random.default_rng(15)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(5)
        ]
        reranker = PyversityRanker(strategy=Strategy.DPP)
        result = reranker.run(documents=docs, strategy=Strategy.MMR)
        assert len(result["documents"]) == 5

    def test_run_strategy_runtime_none_falls_back_to_init(self):
        rng = np.random.default_rng(16)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(5)
        ]
        reranker = PyversityRanker(strategy=Strategy.MMR)
        result_default = reranker.run(documents=docs)
        result_none = reranker.run(documents=docs, strategy=None)
        assert [d.score for d in result_default["documents"]] == [d.score for d in result_none["documents"]]

    def test_run_all_params_overridden_at_runtime(self):
        rng = np.random.default_rng(17)
        docs = [
            Document(content=f"doc {i}", score=float(i) / 10, embedding=rng.standard_normal(8).tolist())
            for i in range(10)
        ]
        reranker = PyversityRanker(top_k=10, strategy=Strategy.DPP, diversity=0.9)
        result = reranker.run(documents=docs, top_k=4, strategy=Strategy.MMR, diversity=0.1)
        assert len(result["documents"]) == 4


@pytest.mark.parametrize(
    "pipeline",
    [Strategy.DPP, Strategy.MMR, Strategy.MSD, Strategy.SSD, Strategy.COVER],
    indirect=True,
)
def test_pipeline_returns_k_documents(pipeline, query_embedding):
    result = pipeline.run({"retriever": {"query_embedding": query_embedding}})
    assert len(result["reranker"]["documents"]) == 5
