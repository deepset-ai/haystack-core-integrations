import pytest

from haystack_integrations.agent_pack.dataclasses import EvaluationMetrics, ModelTokenUsage, RunRecord


def test_run_fingerprint_depends_on_content_not_storage_identity() -> None:
    """Run content, rather than its storage identifier, determines its experiment fingerprint."""
    first = RunRecord(run_id="first", inputs={"messages": []}, outputs={"answer": "same"})
    second = RunRecord(run_id="second", inputs=first.inputs, outputs=first.outputs)
    changed = RunRecord(run_id="first", inputs=first.inputs, outputs={"answer": "changed"})

    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != changed.fingerprint()


def test_evaluation_metrics_roundtrip() -> None:
    """Shared harness measurements retain raw usage and evaluator-specific details when serialized."""
    metrics = EvaluationMetrics(
        quality=0.75,
        latency_ms=12.5,
        model_usage={"model": ModelTokenUsage(input_tokens=100, output_tokens=20)},
        details={"validated": True},
    )

    assert EvaluationMetrics.from_dict(data=metrics.to_dict()) == metrics


@pytest.mark.parametrize("quality", [-0.01, 1.01])
def test_evaluation_metrics_reject_quality_outside_normalized_range(quality: float) -> None:
    """Every harness evaluator must use the shared normalized quality scale."""
    with pytest.raises(ValueError, match="quality must be between"):
        EvaluationMetrics(quality=quality, latency_ms=1.0)
