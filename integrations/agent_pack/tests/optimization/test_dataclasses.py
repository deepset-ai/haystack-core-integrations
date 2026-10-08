import pytest

from haystack_integrations.agent_pack.evaluation.dataclasses import EvalMetrics
from haystack_integrations.agent_pack.optimization import (
    CandidateOutcome,
    OptimizationObjectives,
)


def measured(quality, all_tokens_reported=True, durations=None, **fields):
    """A measurement from a stub harness that reports its quality under `quality`."""
    return EvalMetrics(
        durations=durations or [], all_tokens_reported=all_tokens_reported, details={"quality": quality}, **fields
    )


class TestOptimizationObjectives:
    def test_init_negative_quality_loss(self):
        """A negative tolerance would put the floor above the reference itself."""
        with pytest.raises(ValueError, match="max_quality_loss"):
            OptimizationObjectives(quality_metric="mean_recall_at_k", max_quality_loss=-0.1)

    def test_get_quality(self):
        metrics = EvalMetrics(durations=[1], all_tokens_reported=True, details={"mean_recall_at_k": 0.75})
        assert OptimizationObjectives(quality_metric="mean_recall_at_k").get_quality(metrics=metrics) == 0.75

    def test_get_quality_missing_key(self):
        """A misspelled metric names what the harness did report, so the fix is obvious."""
        metrics = EvalMetrics(durations=[1], all_tokens_reported=True, details={"mean_recall_at_k": 0.75})
        with pytest.raises(ValueError, match="mean_recall_at_k"):
            OptimizationObjectives(quality_metric="recall").get_quality(metrics=metrics)

    @pytest.mark.parametrize(
        ("candidate", "max_quality_loss", "expected"),
        [
            (measured(quality=0.9), 0.0, ()),
            # Within floating-point noise of the floor counts as reaching it
            (measured(quality=0.1 + 0.2), 0.0, ()),
            (measured(quality=0.2), 0.0, ("quality_below_floor:0.3000",)),
            # The tolerance lowers the floor to 0.15
            (measured(quality=0.2), 0.15, ()),
            (measured(quality=0.9, all_tokens_reported=False), 0.0, ("usage_incomplete",)),
        ],
    )
    def test_find_failed_gates(self, candidate, max_quality_loss, expected):
        objectives = OptimizationObjectives(quality_metric="quality", max_quality_loss=max_quality_loss)
        assert objectives.find_failed_gates(metrics=candidate, baseline=measured(quality=0.3)) == expected

    @pytest.mark.parametrize(
        ("primary", "expected"),
        [("cost", (2.0, 3.0)), ("duration", (3.0, 2.0)), ("quality", (-0.5, 2.0))],
    )
    def test_sort_key(self, primary, expected):
        objectives = OptimizationObjectives(quality_metric="quality", primary=primary)
        assert objectives.sort_key(metrics=measured(quality=0.5, durations=[1.0, 3.0, 5.0]), cost=2.0) == expected

    def test_sort_key_missing_cost_and_duration(self):
        objectives = OptimizationObjectives(quality_metric="quality")
        assert objectives.sort_key(metrics=measured(quality=0.5), cost=None) == (float("inf"), float("inf"))


class TestCandidateOutcome:
    def test_to_dict(self):
        metrics = measured(quality=0.5, durations=[1.0])
        candidate = CandidateOutcome(candidate_id="c1", metrics=metrics, cost=0.25, gate_failures=("usage_incomplete",))
        assert candidate.to_dict() == {
            "candidate_id": "c1",
            "parent_id": None,
            "rationale": "",
            "diff": "",
            "metrics": metrics.to_dict(),
            "cost": 0.25,
            "failure": None,
            "gate_failures": ["usage_incomplete"],
        }
