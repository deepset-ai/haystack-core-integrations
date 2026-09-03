import pytest

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics, ModelTokenUsage


def test_run_fingerprint_depends_on_content_not_storage_identity() -> None:
    """Run content, rather than its storage identifier, determines its experiment fingerprint."""
    first = AgentRunRecord(run_id="first", inputs={"messages": []}, outputs={"answer": "same"})
    second = AgentRunRecord(run_id="second", inputs=first.inputs, outputs=first.outputs)
    changed = AgentRunRecord(run_id="first", inputs=first.inputs, outputs={"answer": "changed"})

    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != changed.fingerprint()


def test_evaluation_metrics_roundtrip() -> None:
    """Shared harness measurements retain raw usage and evaluator-specific details when serialized."""
    metrics = EvaluationMetrics(
        quality=0.75,
        latency_ms=12.5,
        model_usage={"model": ModelTokenUsage(input_tokens=100, output_tokens=20)},
        quality_lower_bound=0.5,
        details={"validated": True},
    )

    assert EvaluationMetrics.from_dict(data=metrics.to_dict()) == metrics


@pytest.mark.parametrize("quality", [-0.01, 1.01])
def test_evaluation_metrics_reject_quality_outside_normalized_range(quality: float) -> None:
    """Every harness evaluator must use the shared normalized quality scale."""
    with pytest.raises(ValueError, match="quality must be between"):
        EvaluationMetrics(quality=quality, latency_ms=1.0)


def test_evaluation_metrics_reject_lower_bound_above_quality() -> None:
    """A conservative quality estimate cannot exceed the aggregate quality it bounds."""
    with pytest.raises(ValueError, match="quality_lower_bound"):
        EvaluationMetrics(quality=0.5, quality_lower_bound=0.6, latency_ms=1.0)
