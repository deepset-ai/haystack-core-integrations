import numpy as np
import pytest
from haystack import Document
from integrations.fastembed.haystack_integrations.components.rankers.fastembed import (
    FastembedColbertReranker,
)
from integrations.fastembed.haystack_integrations.components.rankers.fastembed.colbert_reranker import (
    _maxsim_score,
    _l2_normalize_rows,
)


def test_symbol_imports():
    rr = FastembedColbertReranker()
    assert hasattr(rr, "run")


def test_l2_normalize_rows_shapes_and_zeros():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    y = _l2_normalize_rows(x)
    # First row becomes unit norm, second row stays zeros (safe divide)
    assert y.shape == x.shape
    assert np.isclose(np.linalg.norm(y[0]), 1.0, atol=1e-6)
    assert np.allclose(y[1], np.zeros_like(y[1]))


def test_maxsim_score_simple_cosine():
    # Query has two tokens; doc has three tokens
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # e1, e2
    d = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)  # e1, mid, e2

    # With cosine + normalization, MaxSim per q-token is 1.0 for both -> sum = 2.0
    score = _maxsim_score(q, d, similarity="cosine", normalize=True)
    assert np.isclose(score, 2.0, atol=1e-6)


def test_run_no_documents_returns_empty():
    rr = FastembedColbertReranker()
    result = rr.run(query="test", documents=[])
    assert result["documents"] == []


def test_run_topk_slices_without_fastembed(monkeypatch):
    """
    We monkeypatch the encoder methods to avoid requiring FastEmbed models in CI.
    This also checks sorting by score and top_k behavior deterministically.
    """
    rr = FastembedColbertReranker()

    # Fake "ready" state (skip warm_up + fastembed import)
    rr._ready = True
    rr._encoder = object()

    # Query encoding -> 2x2 (arbitrary but fixed)
    def fake_encode_query(text: str):
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # Doc encodings -> create three docs with different alignments
    def fake_encode_docs(texts):
        mats = []
        for t in texts:
            if "best" in t:
                mats.append(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))  # high
            elif "ok" in t:
                mats.append(np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32))  # medium
            else:
                mats.append(np.array([[1.0, 0.0]], dtype=np.float32))  # lower
        return mats

    rr._encode_query = fake_encode_query  # type: ignore
    rr._encode_docs_batched = fake_encode_docs  # type: ignore

    docs = [
        Document(content="this is lower"),
        Document(content="this is ok"),
        Document(content="this is best"),
    ]
    out = rr.run(query="q", documents=docs, top_k=2)
    ranked = out["documents"]

    # Expect "best" first, then "ok"
    assert len(ranked) == 2
    assert "best" in ranked[0].content
    assert "ok" in ranked[1].content
    assert ranked[0].score >= ranked[1].score


def test_param_validation():
    with pytest.raises(ValueError):
        FastembedColbertReranker(batch_size=0)
    with pytest.raises(ValueError):
        FastembedColbertReranker(similarity="euclid")  # unsupported


def test_topk_validation():
    rr = FastembedColbertReranker()
    with pytest.raises(ValueError):
        rr.run(query="q", documents=[Document(content="x")], top_k=-1)


def test_stable_tie_break(monkeypatch):
    rr = FastembedColbertReranker()
    rr._ready = True
    rr._encoder = object()

    def fake_q(_):  # same query vectors
        return np.eye(2, dtype=np.float32)

    def fake_docs(texts):
        # craft docs with identical scores → tie
        mat = np.eye(2, dtype=np.float32)
        return [mat for _ in texts]

    rr._encode_query = fake_q  # type: ignore
    rr._encode_docs_batched = fake_docs  # type: ignore

    docs = [Document(content=f"doc{i}") for i in range(4)]
    out = rr.run(query="q", documents=docs)
    ranked = out["documents"]
    # Ensure length preserved and deterministic order
    assert [d.content for d in ranked] == ["doc3", "doc2", "doc1", "doc0"]  # due to tie-break we defined
