# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from math import isclose
from statistics import median
from typing import Any, Literal

from haystack_integrations.agent_pack.evaluation.dataclasses import EvalMetrics


@dataclass(kw_only=True)
class OptimizationObjectives:
    """
    Which measurement the experiment treats as quality, the gates a candidate must clear, and what it is ranked on.

    :param quality_metric: The key in `EvalMetrics.details` treated as quality, where higher is better. Which keys
        exist depends on the harness evaluator measuring the candidates.
    :param max_quality_loss: How far below the reference's quality metric a candidate may fall and still clear the
        gates.
    :param primary: What candidates are ranked by. "cost" and "duration" are minimized among candidates that clear
        the quality gates, where "duration" is the median of `EvalMetrics.durations`. "quality" is maximized
        directly, with cost breaking ties.
    """

    quality_metric: str
    max_quality_loss: float = 0.0
    primary: Literal["cost", "duration", "quality"] = "cost"

    def __post_init__(self) -> None:
        """Reject a negative tolerance, which would put the floor above the reference itself."""
        if self.max_quality_loss < 0:
            msg = "max_quality_loss must not be negative."
            raise ValueError(msg)

    def get_quality(self, metrics: EvalMetrics) -> float:
        """
        Read the quality metric from one measurement.

        :param metrics: A harness evaluator's measurement.
        :returns: The value the harness reported under `quality_metric`.
        :raises ValueError: If the harness reported no such key.
        """
        if self.quality_metric not in metrics.details:
            msg = f"The harness reported no {self.quality_metric!r}; it reported {sorted(metrics.details)}."
            raise ValueError(msg)
        return float(metrics.details[self.quality_metric])

    def find_failed_gates(self, metrics: EvalMetrics, baseline: EvalMetrics) -> tuple[str, ...]:
        """
        Check a candidate's measurement against the hard gates.

        :param metrics: The candidate's measurement.
        :param baseline: The reference's measurement.
        :returns: The gates the candidate failed, for example `("quality_below_floor:0.8000",)`. An empty tuple is
            returned when it cleared them all.
        """
        failures: list[str] = []
        floor = self.get_quality(metrics=baseline) - self.max_quality_loss
        quality = self.get_quality(metrics=metrics)
        # A candidate within floating-point noise of the floor counts as reaching it
        if quality < floor and not isclose(quality, floor, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(f"quality_below_floor:{floor:.4f}")

        # A model call that reported no usage usually means a component failed without raising
        if not metrics.all_tokens_reported:
            failures.append("usage_incomplete")
        return tuple(failures)

    def sort_key(self, metrics: EvalMetrics, cost: float | None) -> tuple[float, float]:
        """
        Build the sort key for one measurement, lower being better.

        :param metrics: A measurement.
        :param cost: Its cost, or `None` when it could not be priced.
        :returns: `(cost, median duration)` for "cost", `(median duration, cost)` for "duration", and
            `(-quality, cost)` for "quality". A missing cost or duration sorts as infinity.
        """
        known_cost = cost if cost is not None else float("inf")
        if self.primary == "quality":
            return (-self.get_quality(metrics=metrics), known_cost)
        duration = median(metrics.durations) if metrics.durations else float("inf")
        if self.primary == "duration":
            return (duration, known_cost)
        return (known_cost, duration)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(obj=self)


@dataclass
class CandidateConfiguration:
    """
    One configuration the optimizer submitted through a `ConfigurationEditorToolset`.

    :param candidate_id: Digest of the parsed YAML, which ignores formatting.
    :param parent_id: Identifier of the snapshot the edits started from, the reference's ID when that is the
        reference.
    :param yaml: The submitted configuration as Pipeline YAML.
    :param rationale: The hypothesis the optimizer gave when submitting it.
    :param diff: Unified diff of the parent snapshot against the submitted YAML.
    """

    candidate_id: str
    parent_id: str
    yaml: str
    rationale: str
    diff: str


@dataclass(kw_only=True)
class CandidateOutcome:
    """
    One measured candidate, as the optimizer is shown it on later turns.

    :param candidate_id: Identifier of the candidate configuration.
    :param parent_id: Identifier of the configuration it was edited from, or None when that is the reference.
    :param rationale: Why the optimizer proposed it.
    :param diff: Its change against the parent, as a unified diff.
    :param metrics: Its measurement, or None when it could not be measured.
    :param cost: Its cost, or None when it could not be priced.
    :param failure: Why it could not be measured.
    :param gate_failures: The gates it failed, for example `("usage_incomplete",)`. Empty when it cleared them all.
    """

    candidate_id: str
    parent_id: str | None = None
    rationale: str = ""
    diff: str = ""
    metrics: EvalMetrics | None = None
    cost: float | None = None
    failure: str | None = None
    gate_failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "rationale": self.rationale,
            "diff": self.diff,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "cost": self.cost,
            "failure": self.failure,
            "gate_failures": list(self.gate_failures),
        }
