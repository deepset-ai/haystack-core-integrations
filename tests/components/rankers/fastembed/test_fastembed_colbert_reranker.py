from haystack import Document
from integrations.fastembed.haystack_integrations.components.rankers.fastembed import (
    FastembedColbertReranker,
)


def test_symbol_imports():
    # Just ensures the class is importable before we add real tests
    rr = FastembedColbertReranker()
    assert hasattr(rr, "run")


def test_run_no_documents_returns_empty():
    rr = FastembedColbertReranker()
    result = rr.run(query="test", documents=[])
    assert result["documents"] == []


def test_run_sets_default_scores_and_preserves_length():
    docs = [Document(content="a"), Document(content="b")]
    rr = FastembedColbertReranker()
    out = rr.run(query="q", documents=docs)
    assert len(out["documents"]) == 2
    assert all(d.score is not None for d in out["documents"])