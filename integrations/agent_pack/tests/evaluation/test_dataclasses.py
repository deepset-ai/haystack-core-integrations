import pytest

from haystack_integrations.evaluation import EvaluationMetrics, ModelTokenUsage, RetrievalEvalCase


def test_the_documents_an_answer_needs_are_the_ones_its_evidence_is_in():
    case = RetrievalEvalCase(question="q", evidence={"a": "a quote", "b": "another quote"})

    assert case.expected_document_ids == frozenset({"a", "b"})


def test_only_the_documents_above_the_cutoff_count():
    """A pipeline is measured on what it put at the top, so a needed document ranked below k earns nothing."""
    eval_case = RetrievalEvalCase(question="q", evidence={"a": "first quote", "b": "second quote"})
    ranked = ["a", "filler", "filler2", "b"]

    assert eval_case.found_at(document_ids=ranked, k=2) == frozenset({"a"})
    assert eval_case.recall_at(document_ids=ranked, k=2) == 0.5
    assert eval_case.recall_at(document_ids=ranked) == 1.0
    # Precision is over what was scored, not over everything returned.
    assert eval_case.precision_at(document_ids=ranked, k=2) == 0.5
    assert eval_case.precision_at(document_ids=ranked) == 0.5


def test_a_document_returned_twice_does_not_occupy_the_cutoff_twice():
    eval_case = RetrievalEvalCase(question="q", evidence={"b": "a quote"})
    ranked = ["a", "a", "b"]

    assert eval_case.recall_at(document_ids=ranked, k=2) == 1.0
    assert eval_case.precision_at(document_ids=ranked, k=2) == 0.5


def test_scoring_a_run_that_returned_nothing_is_zero_rather_than_an_error():
    eval_case = RetrievalEvalCase(question="q", evidence={"a": "a quote"})

    assert eval_case.recall_at(document_ids=[]) == 0.0
    assert eval_case.precision_at(document_ids=[]) == 0.0


def test_an_eval_case_without_evidence_has_nothing_to_score_against():
    with pytest.raises(ValueError, match="needs evidence to score against"):
        RetrievalEvalCase(question="q", evidence={})


def test_evaluation_metrics_roundtrip() -> None:
    """Shared harness measurements retain raw usage and evaluator-specific details when serialized."""
    metrics = EvaluationMetrics(
        quality=0.75,
        latency_ms=12.5,
        model_usage={"model": ModelTokenUsage(input_tokens=100, output_tokens=20)},
        details={"mean_recall": 0.5},
    )

    assert EvaluationMetrics.from_dict(data=metrics.to_dict()) == metrics


@pytest.mark.parametrize("quality", [-0.01, 1.01])
def test_evaluation_metrics_reject_quality_outside_normalized_range(quality: float) -> None:
    """Every harness evaluator must use the shared normalized quality scale."""
    with pytest.raises(ValueError, match="quality must be between"):
        EvaluationMetrics(quality=quality, latency_ms=1.0)
