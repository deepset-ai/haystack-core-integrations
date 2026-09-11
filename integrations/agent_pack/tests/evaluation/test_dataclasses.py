import pytest

from haystack_integrations.evaluation import EvaluationMetrics, ModelTokenUsage, RetrievalEvalCase


class TestRetrievalEvalCase:
    def test_init_without_evidence_raises_error(self):
        with pytest.raises(ValueError, match="needs evidence to score against"):
            RetrievalEvalCase(question="q", evidence={})

    def test_expected_document_ids_are_the_evidence_keys(self):
        eval_case = RetrievalEvalCase(question="q", evidence={"a": "a quote", "b": "another quote"})

        assert eval_case.expected_document_ids == frozenset({"a", "b"})

    def test_scoring_counts_only_the_first_k(self):
        """A pipeline is measured on what it put at the top, so a needed document ranked below k earns nothing."""
        eval_case = RetrievalEvalCase(question="q", evidence={"a": "first quote", "b": "second quote"})
        ranked = ["a", "filler", "filler2", "b"]

        assert eval_case.found_at(document_ids=ranked, k=2) == frozenset({"a"})
        assert eval_case.recall_at(document_ids=ranked, k=2) == 0.5
        assert eval_case.recall_at(document_ids=ranked) == 1.0
        # Precision is over what was scored, not over everything returned.
        assert eval_case.precision_at(document_ids=ranked, k=2) == 0.5
        assert eval_case.precision_at(document_ids=ranked) == 0.5

    def test_scoring_ignores_duplicates_when_applying_the_cutoff(self):
        eval_case = RetrievalEvalCase(question="q", evidence={"b": "a quote"})
        ranked = ["a", "a", "b"]

        assert eval_case.recall_at(document_ids=ranked, k=2) == 1.0
        assert eval_case.precision_at(document_ids=ranked, k=2) == 0.5

    def test_scoring_an_empty_run_is_zero(self):
        eval_case = RetrievalEvalCase(question="q", evidence={"a": "a quote"})

        assert eval_case.recall_at(document_ids=[]) == 0.0
        assert eval_case.precision_at(document_ids=[]) == 0.0

    def test_serialization_roundtrip(self):
        eval_case = RetrievalEvalCase(question="q", evidence={"b": "second", "a": "first"}, min_precision=0.5)

        assert RetrievalEvalCase.from_dict(data=eval_case.to_dict()) == eval_case
        # Evidence is ordered, so the same set is identified the same way whichever order it was built in.
        assert list(eval_case.to_dict()["evidence"]) == ["a", "b"]


class TestEvaluationMetrics:
    @pytest.mark.parametrize("quality", [-0.01, 1.01])
    def test_init_quality_outside_the_normalized_range_raises_error(self, quality: float):
        """Every harness evaluator must use the shared normalized quality scale."""
        with pytest.raises(ValueError, match="quality must be between"):
            EvaluationMetrics(quality=quality, latency_ms=1.0)

    def test_serialization_roundtrip(self):
        metrics = EvaluationMetrics(
            quality=0.75,
            latency_ms=12.5,
            model_usage={"model": ModelTokenUsage(input_tokens=100, output_tokens=20)},
            details={"mean_recall": 0.5},
        )

        assert EvaluationMetrics.from_dict(data=metrics.to_dict()) == metrics
