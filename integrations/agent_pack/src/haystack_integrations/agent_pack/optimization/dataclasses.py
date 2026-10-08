# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from math import isclose
from pathlib import Path
from statistics import median
from typing import Any, Literal

from haystack_integrations.agent_pack.evaluation.dataclasses import EvalMetrics, ModelTokenUsage
from haystack_integrations.agent_pack.optimization.utils import content_digest


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
    One configuration the optimizer submitted with `submit_candidate`.

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


@dataclass(frozen=True)
class ConfigurationDraft:
    """
    The YAML the optimizer is editing during one proposal turn.

    :param yaml: The current YAML.
    :param parent_id: The configuration the edits started from.
    :param validated_revision: The revision that last passed `validate_config`, or `None` when the current YAML has
        not.
    """

    yaml: str
    parent_id: str
    validated_revision: str | None = None

    @property
    def revision(self) -> str:
        """The current YAML's revision, which every edit has to name."""
        return content_digest(payload=self.yaml)


@dataclass(frozen=True)
class KnownConfigurations:
    """
    Every configuration measured before a proposal turn, which the optimizer can restore and may not submit again.

    :param reference_id: The reference's ID, which the optimizer restores as `"reference"`.
    :param yaml_by_id: The YAML of the reference and of every earlier candidate, keyed by ID.
    """

    reference_id: str
    yaml_by_id: dict[str, str] = field(default_factory=dict)


@dataclass(kw_only=True)
class ProposalResult:
    """
    What one optimizer turn produced.

    :param candidate: The submitted configuration, or `None` when the optimizer finished or ran out of steps.
    :param finish_reason: Why the optimizer ended the search, when it called `finish`.
    :param validation_failures: Every draft that failed `validate_config` during the turn, as `{revision, error}`.
    """

    candidate: CandidateConfiguration | None
    finish_reason: str | None = None
    validation_failures: list[dict[str, str]] = field(default_factory=list)


@dataclass(kw_only=True)
class CandidateEvaluation:
    """
    Journaled measurement or failure for one complete YAML configuration.

    :param measurement_context: Fingerprint of what makes two measurements comparable.
    :param run_id: The experiment run this record belongs to.
    :param candidate_id: Identifier of the measured configuration.
    :param configuration: The submitted configuration, or None for the reference and for drafts that failed
        validation.
    :param metrics: The evaluator's measurement, or None when the configuration could not be measured.
    :param cost: What `metrics.model_usage` costs at the experiment's prices, or None when it was not measured or
        uses a model without a known price.
    :param failure: Why the configuration could not be measured.
    """

    measurement_context: str
    run_id: str
    candidate_id: str
    configuration: CandidateConfiguration | None
    metrics: EvalMetrics | None
    cost: float | None = None
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible journal record."""
        return {
            "measurement_context": self.measurement_context,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "configuration": asdict(self.configuration) if self.configuration is not None else None,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "cost": self.cost,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateEvaluation":
        """Restore a journal record."""
        metrics = data.get("metrics")
        return cls(
            measurement_context=data["measurement_context"],
            run_id=data["run_id"],
            candidate_id=data["candidate_id"],
            configuration=CandidateConfiguration(**data["configuration"]) if data.get("configuration") else None,
            metrics=EvalMetrics.from_dict(data=metrics) if metrics is not None else None,
            cost=data.get("cost"),
            failure=data.get("failure"),
        )


@dataclass(kw_only=True)
class CandidateProgress:
    """
    One measured candidate, passed to `on_candidate` as soon as it has been measured.

    :param position: Which candidate this is, counting from one.
    :param total: How many candidates the experiment may measure in all.
    :param evaluation: The measurement and its cost, or the failure that replaced them.
    :param gate_failures: The hard gates this candidate missed, empty when it cleared them all.
    :param baseline: The reference measurement, so a caller can report a change without holding state.
    :param baseline_cost: The reference's cost, or None when it could not be priced.
    :param is_best: Whether this candidate now leads the eligible ones.
    """

    position: int
    total: int
    evaluation: CandidateEvaluation
    gate_failures: tuple[str, ...]
    baseline: EvalMetrics
    baseline_cost: float | None
    is_best: bool


@dataclass(kw_only=True)
class ExperimentRecommendation:
    """The measured configuration that passed its gates and outranked the reference."""

    configuration: CandidateConfiguration
    evaluation: CandidateEvaluation
    reasons: tuple[str, ...] = ()


@dataclass(kw_only=True)
class ExperimentResult:
    """Baseline, candidate outcomes, and the optional best recommendation."""

    baseline: EvalMetrics
    baseline_cost: float | None
    candidates: tuple[CandidateEvaluation, ...]
    recommendation: ExperimentRecommendation | None
    measurement_context: str
    run_id: str
    artifact_directory: Path
    gate_failures: dict[str, tuple[str, ...]] = field(default_factory=dict)
    optimizer_usage: dict[str, ModelTokenUsage] = field(default_factory=dict)
    optimizer_cost: float | None = None
