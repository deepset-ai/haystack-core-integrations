# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Measurements, objectives, and outcomes of a harness optimization experiment."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from haystack.components.agents import Agent

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe


@dataclass(frozen=True, kw_only=True)
class EvaluationMetrics:
    """
    Comparable quality, cost, and latency metrics for one harness.

    :param quality: Aggregate quality score in `[0, 1]`.
    :param cost: Total model cost for the evaluated cases, in currency units.
    :param latency_ms: Wall-clock latency for the evaluated cases.
    :param quality_lower_bound: Optional pessimistic quality estimate. Quality gates use this when it is present, so
        an evaluator that repeats each case can require the *lower* end of the observed range to clear the floor
        rather than the mean of a single noisy sample.
    :param details: Free-form evaluator output recorded in the experiment journal.
    """

    quality: float
    cost: float
    latency_ms: float
    quality_lower_bound: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def gating_quality(self) -> float:
        """
        Return the quality value that gates compare against.

        :returns: The lower bound when the evaluator reported one, otherwise the aggregate quality.
        """
        return self.quality if self.quality_lower_bound is None else self.quality_lower_bound

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the EvaluationMetrics into a dictionary.

        :returns: A dictionary with one key per field.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationMetrics":
        """
        Create a new EvaluationMetrics object from a dictionary.

        :param data: The dictionary to build the metrics from.
        :returns: The created object.
        """
        lower_bound = data.get("quality_lower_bound")
        return cls(
            quality=float(data["quality"]),
            cost=float(data["cost"]),
            latency_ms=float(data["latency_ms"]),
            quality_lower_bound=None if lower_bound is None else float(lower_bound),
            details=data.get("details") or {},
        )


@dataclass(frozen=True, kw_only=True)
class OptimizationObjectives:
    """
    Experiment gates and ranking preferences.

    :param min_quality: Absolute quality floor every candidate must clear.
    :param max_quality_loss: How far below the reference's quality a candidate may fall.
    :param primary: Which measurement candidates are ranked on first.
    """

    min_quality: float = 0.0
    max_quality_loss: float = 0.0
    primary: Literal["cost", "latency"] = "cost"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_quality <= 1.0:
            msg = "min_quality must be between 0 and 1."
            raise ValueError(msg)
        if self.max_quality_loss < 0:
            msg = "max_quality_loss cannot be negative."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the OptimizationObjectives into a dictionary.

        :returns: A dictionary with one key per field.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationObjectives":
        """
        Create a new OptimizationObjectives object from a dictionary.

        :param data: The dictionary to build the objectives from.
        :returns: The created object.
        """
        return cls(**data)


@dataclass(frozen=True, kw_only=True)
class CandidateEvaluation:
    """
    Journaled outcome for one recipe.

    Only measurements are recorded. Whether a candidate passes the experiment's gates is recomputed on every run, so
    changing the objectives re-ranks journaled results instead of replaying a stale verdict.

    :param candidate_id: Identifier covering both the recipe and the configuration it was measured under.
    :param configuration_hash: Hash of the experiment configuration the measurement belongs to.
    :param recipe: The serialized transformation that produced the candidate.
    :param metrics: What the evaluator measured, or None if the evaluation failed.
    :param failure: The error that ended the evaluation, if it failed.
    """

    candidate_id: str
    configuration_hash: str
    recipe: dict[str, Any]
    metrics: EvaluationMetrics | None
    failure: str | None = None

    @property
    def succeeded(self) -> bool:
        """
        Return whether this candidate produced metrics.

        :returns: True if the evaluation completed and recorded measurements.
        """
        return self.failure is None and self.metrics is not None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the CandidateEvaluation into a dictionary.

        :returns: A dictionary with one key per field, with the nested metrics and validation as dictionaries.
        """
        return {
            "candidate_id": self.candidate_id,
            "configuration_hash": self.configuration_hash,
            "recipe": self.recipe,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateEvaluation":
        """
        Create a new CandidateEvaluation object from a dictionary.

        :param data: The dictionary to build the evaluation from.
        :returns: The created object.
        """
        metrics = data.get("metrics")
        return cls(
            candidate_id=data["candidate_id"],
            configuration_hash=data.get("configuration_hash", ""),
            recipe=data["recipe"],
            metrics=EvaluationMetrics.from_dict(data=metrics) if metrics is not None else None,
            failure=data.get("failure"),
        )


@dataclass(frozen=True, kw_only=True)
class ExperimentRecommendation:
    """
    A candidate that met the hard gates and outranked the reference harness.

    :param recipe: The transformation to apply.
    :param evaluation: Its recorded measurements.
    :param reasons: Why this candidate is recommended. `cost_improvement` / `latency_improvement` mean it beat the
        reference on the primary objective. `quality_unvalidated` means quality was scored against cases derived from
        reference traces rather than labelled expectations, so the comparison only shows parity with the incumbent,
        not correctness.
    """

    recipe: CandidateRecipe
    evaluation: CandidateEvaluation
    reasons: tuple[str, ...] = ()

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Explicitly materialize the recommended candidate for human inspection or approval.

        :param reference: The champion harness the recipe applies to.
        :param assets: The approved model and tool allowlist, re-checked before the Agent is handed back.
        :returns: The recommended candidate Agent.
        """
        return self.recipe.materialize(reference=reference, assets=assets)


@dataclass(frozen=True, kw_only=True)
class ExperimentResult:
    """
    Baseline, candidate outcomes, and optional recommendation.

    :param baseline: What the reference harness measured.
    :param candidates: Every candidate's recorded outcome.
    :param recommendation: The candidate worth approving, if any.
    :param configuration_hash: Hash of the configuration these measurements belong to.
    :param gate_failures: Per candidate ID, the hard gates it missed. An empty tuple means it was eligible.
    """

    baseline: EvaluationMetrics
    candidates: tuple[CandidateEvaluation, ...]
    recommendation: ExperimentRecommendation | None
    configuration_hash: str = ""
    gate_failures: dict[str, tuple[str, ...]] = field(default_factory=dict)
