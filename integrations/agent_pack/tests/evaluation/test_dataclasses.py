import pytest

from haystack_integrations.evaluation import EvalMetrics, ModelTokenUsage, RetrievalEvalCase


class TestRetrievalEvalCase:
    def test_init_without_evidence(self):
        with pytest.raises(ValueError, match="needs evidence to score against"):
            RetrievalEvalCase(question="q", evidence={})

    def test_expected_document_ids(self):
        eval_case = RetrievalEvalCase(question="q", evidence={"a": "a quote", "b": "another quote"})
        assert eval_case.expected_document_ids == frozenset({"a", "b"})

    @pytest.mark.parametrize(
        ("ranked", "k", "found", "recall", "precision"),
        [
            pytest.param(["a", "x", "y", "b"], 2, {"a"}, 0.5, 0.5, id="counts_only_the_first_k"),
            pytest.param(["a", "x", "y", "b"], None, {"a", "b"}, 1.0, 0.5, id="no_cutoff_scores_everything"),
            pytest.param(["a", "a", "b"], 2, {"a", "b"}, 1.0, 1.0, id="duplicates_do_not_fill_the_cutoff"),
            pytest.param([], None, set(), 0.0, 0.0, id="an_empty_run_scores_zero"),
        ],
    )
    def test_scoring(self, ranked: list[str], k: int | None, found: set[str], recall: float, precision: float):
        """A pipeline is measured on what it put at the top, so a needed document ranked below k earns nothing."""
        eval_case = RetrievalEvalCase(question="q", evidence={"a": "first quote", "b": "second quote"})
        assert eval_case.found_at(document_ids=ranked, k=k) == frozenset(found)
        assert eval_case.recall_at(document_ids=ranked, k=k) == recall
        assert eval_case.precision_at(document_ids=ranked, k=k) == precision

    def test_serialization_roundtrip(self):
        eval_case = RetrievalEvalCase(question="q", evidence={"b": "second", "a": "first"}, min_precision=0.5)
        assert RetrievalEvalCase.from_dict(data=eval_case.to_dict()) == eval_case
        # Evidence is ordered, so the same set is identified the same way whichever order it was built in.
        assert list(eval_case.to_dict()["evidence"]) == ["a", "b"]


class TestEvalMetrics:
    @pytest.mark.parametrize("quality", [-0.01, 1.01])
    def test_init_invalid_quality(self, quality: float):
        """Every harness evaluator must use the shared normalized quality scale."""
        with pytest.raises(ValueError, match="quality must be between"):
            EvalMetrics(quality=quality, latency_ms=1.0)

    def test_serialization_roundtrip(self):
        metrics = EvalMetrics(
            quality=0.75,
            latency_ms=12.5,
            model_usage={"model": ModelTokenUsage(input_tokens=100, output_tokens=20)},
            details={"mean_recall": 0.5},
        )
        assert EvalMetrics.from_dict(data=metrics.to_dict()) == metrics
