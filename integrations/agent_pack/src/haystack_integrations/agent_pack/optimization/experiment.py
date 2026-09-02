# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Iterative measurement and recommendation of Agent harness candidates."""

import hashlib
import json
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from haystack import logging
from haystack.components.agents import Agent
from pydantic import ValidationError

from haystack_integrations.agent_pack.optimization.models import (
    ApprovedAssetCatalog,
    EvaluationMetrics,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.proposer import RecipeProposer
from haystack_integrations.agent_pack.optimization.recipes import (
    CandidateRecipe,
    is_obvious_noop,
    parse_proposal,
    recipe_fingerprint,
)
from haystack_integrations.agent_pack.runs import AgentRunRecord, RunSelection, RunSource

logger = logging.getLogger(__name__)
_MAX_STALLED_PROPOSALS = 3


class HarnessEvaluator(Protocol):
    """Measure one materialized Agent over a fixed set of reference runs."""

    def evaluate(self, agent: Agent, reference_runs: list[AgentRunRecord]) -> EvaluationMetrics:
        """Return raw quality, latency, and model-usage measurements."""
        ...


@dataclass(frozen=True, kw_only=True)
class CandidateEvaluation:
    """Journaled raw measurement or failure for one resolved recipe."""

    measurement_context: str
    candidate_id: str
    recipe: dict[str, Any]
    metrics: EvaluationMetrics | None
    failure: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether evaluation produced metrics."""
        return self.failure is None and self.metrics is not None

    def price(self, assets: ApprovedAssetCatalog) -> "CandidateEvaluation":
        """Apply current prices without changing the journaled raw measurement."""
        return replace(self, metrics=self.metrics.price(assets) if self.metrics is not None else None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible journal record."""
        return {
            "measurement_context": self.measurement_context,
            "candidate_id": self.candidate_id,
            "recipe": self.recipe,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateEvaluation":
        """Restore a journal record."""
        metrics = data.get("metrics")
        return cls(
            measurement_context=data["measurement_context"],
            candidate_id=data["candidate_id"],
            recipe=data["recipe"],
            metrics=EvaluationMetrics.from_dict(metrics) if metrics is not None else None,
            failure=data.get("failure"),
        )


class ExperimentJournal:
    """Append-only JSON-lines persistence for raw experiment measurements."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: dict[str, CandidateEvaluation] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = CandidateEvaluation.from_dict(json.loads(line))
                    self._records[record.candidate_id] = record

    def get(self, candidate_id: str) -> CandidateEvaluation | None:
        """Return a completed measurement; failed attempts remain retryable."""
        with self._lock:
            record = self._records.get(candidate_id)
        return record if record is not None and record.succeeded else None

    def completed(self, measurement_context: str) -> list[CandidateEvaluation]:
        """Return successful candidate measurements from this context, excluding its baseline."""
        with self._lock:
            return [
                record
                for record in self._records.values()
                if record.measurement_context == measurement_context
                and record.succeeded
                and record.recipe.get("kind") != "reference"
            ]

    def append(self, evaluation: CandidateEvaluation) -> None:
        """Persist an outcome unless a successful measurement already exists."""
        with self._lock:
            existing = self._records.get(evaluation.candidate_id)
            if existing is not None and existing.succeeded:
                return
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(evaluation.to_dict(), sort_keys=True) + "\n")
            self._records[evaluation.candidate_id] = evaluation


@dataclass(frozen=True, kw_only=True)
class ExperimentRecommendation:
    """The measured candidate that passed its gates and outranked the reference."""

    recipe: CandidateRecipe
    evaluation: CandidateEvaluation
    reasons: tuple[str, ...] = ()

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Rebuild the recommendation for inspection or approval."""
        return self.recipe.materialize(reference, assets)


@dataclass(frozen=True, kw_only=True)
class ExperimentResult:
    """Priced baseline, candidate outcomes, and the optional best recommendation."""

    baseline: EvaluationMetrics
    candidates: tuple[CandidateEvaluation, ...]
    recommendation: ExperimentRecommendation | None
    measurement_context: str
    gate_failures: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def configuration_hash(self) -> str:
        """Compatibility alias for the measurement context identifier."""
        return self.measurement_context


def _stable_serialization(value: Any) -> Any:
    """Remove only the process-local identity of an InMemoryDocumentStore from serialized Agent data."""
    if isinstance(value, list):
        return [_stable_serialization(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: _stable_serialization(item) for key, item in value.items()}
    type_name = normalized.get("type")
    if isinstance(type_name, str) and type_name.endswith("InMemoryDocumentStore"):
        parameters = normalized.get("init_parameters")
        if isinstance(parameters, dict):
            parameters.pop("index", None)
    return normalized


class HarnessOptimizationExperiment:
    """Let an optimizer Agent choose, observe, and refine a bounded sequence of harness experiments."""

    def __init__(
        self,
        *,
        reference: Agent,
        run_source: RunSource,
        evaluator: HarnessEvaluator,
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
        journal: ExperimentJournal,
        proposer: RecipeProposer,
        run_selection: RunSelection | None = None,
        configuration_key: str | None = None,
        max_iterations: int = 8,
    ) -> None:
        self.reference = reference
        self.run_source = run_source
        self.evaluator = evaluator
        self.assets = assets
        self.objectives = objectives
        self.journal = journal
        self.proposer = proposer
        self.run_selection = run_selection
        self.configuration_key = configuration_key
        self.max_iterations = max_iterations

    def run(self) -> ExperimentResult:
        """Measure the baseline, then iteratively evaluate recipes selected from observed outcomes."""
        reference_runs = self.run_source.list(self.run_selection)
        if not reference_runs:
            msg = "The selected run source contains no successful reference runs."
            raise ValueError(msg)

        context = self._measurement_context(reference_runs)
        baseline = self._baseline(context, reference_runs).price(self.assets)
        outcomes: list[CandidateEvaluation] = []
        recipe_by_id: dict[str, CandidateRecipe] = {}
        history: list[dict[str, Any]] = []
        seen: set[str] = set()

        for prior in self.journal.completed(context):
            try:
                recipe = parse_proposal({"recipe": prior.recipe}, self.assets)
            except (ValueError, ValidationError):
                continue
            if recipe is None:
                continue
            fingerprint = recipe_fingerprint(recipe, self.assets)
            expected_id = hashlib.sha256(f"{context}:{fingerprint}".encode()).hexdigest()
            if prior.candidate_id != expected_id:
                continue
            seen.add(fingerprint)
            priced = prior.price(self.assets)
            outcomes.append(priced)
            recipe_by_id[priced.candidate_id] = recipe
            history.append(self._history_entry(priced, baseline))

        stalled = 0
        while len(outcomes) < self.max_iterations and stalled < _MAX_STALLED_PROPOSALS:
            recipe = self.proposer.propose(
                reference=self.reference,
                reference_runs=reference_runs,
                assets=self.assets,
                objectives=self.objectives,
                baseline=baseline,
                history=history,
            )
            if recipe is None:
                break

            fingerprint = recipe_fingerprint(recipe, self.assets)
            if fingerprint in seen or is_obvious_noop(recipe, self.reference):
                stalled += 1
                history.append({"recipe": recipe.model_dump(), "status": "rejected", "reason": "duplicate_or_no_op"})
                continue
            stalled = 0
            seen.add(fingerprint)
            candidate_id = hashlib.sha256(f"{context}:{fingerprint}".encode()).hexdigest()
            raw = self.journal.get(candidate_id)
            if raw is None:
                raw = self._evaluate_candidate(recipe, candidate_id, context, reference_runs)
                self.journal.append(raw)
            priced = raw.price(self.assets)
            outcomes.append(priced)
            recipe_by_id[candidate_id] = recipe
            history.append(self._history_entry(priced, baseline))

        gate_failures = {outcome.candidate_id: self._gate_failures(outcome, baseline) for outcome in outcomes}
        eligible = [outcome for outcome in outcomes if not gate_failures[outcome.candidate_id]]
        eligible.sort(key=self._candidate_rank)
        recommendation = None
        for candidate in eligible:
            if (reasons := self._recommendation_reasons(candidate, baseline)) is not None:
                recommendation = ExperimentRecommendation(
                    recipe=recipe_by_id[candidate.candidate_id], evaluation=candidate, reasons=reasons
                )
                break
        return ExperimentResult(
            baseline=baseline,
            candidates=tuple(outcomes),
            recommendation=recommendation,
            measurement_context=context,
            gate_failures=gate_failures,
        )

    def _measurement_context(self, reference_runs: list[AgentRunRecord]) -> str:
        try:
            reference = _stable_serialization(self.reference.to_dict())
        except Exception:
            reference = {"type": f"{type(self.reference).__module__}.{type(self.reference).__qualname__}"}
        evaluator: dict[str, Any] = {"type": f"{type(self.evaluator).__module__}.{type(self.evaluator).__qualname__}"}
        if callable(fingerprint := getattr(self.evaluator, "fingerprint", None)):
            evaluator["configuration"] = fingerprint()
        payload = {
            "reference": reference,
            "evaluator": evaluator,
            "runs": sorted(record.fingerprint() for record in reference_runs),
            "configuration_key": self.configuration_key,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _baseline(self, context: str, reference_runs: list[AgentRunRecord]) -> EvaluationMetrics:
        candidate_id = f"baseline:{context}"
        if (prior := self.journal.get(candidate_id)) is not None and prior.metrics is not None:
            return prior.metrics
        metrics = self.evaluator.evaluate(self.reference, reference_runs)
        self.journal.append(
            CandidateEvaluation(
                measurement_context=context,
                candidate_id=candidate_id,
                recipe={"kind": "reference"},
                metrics=metrics,
            )
        )
        return metrics

    def _evaluate_candidate(
        self,
        recipe: CandidateRecipe,
        candidate_id: str,
        context: str,
        reference_runs: list[AgentRunRecord],
    ) -> CandidateEvaluation:
        try:
            candidate = recipe.materialize(self.reference, self.assets)
            metrics = self.evaluator.evaluate(candidate, reference_runs)
            return CandidateEvaluation(
                measurement_context=context,
                candidate_id=candidate_id,
                recipe=recipe.model_dump(),
                metrics=metrics,
            )
        except Exception as error:
            logger.warning("Candidate evaluation failed for {recipe}: {error}", recipe=recipe.model_dump(), error=error)
            return CandidateEvaluation(
                measurement_context=context,
                candidate_id=candidate_id,
                recipe=recipe.model_dump(),
                metrics=None,
                failure="".join(traceback.format_exception_only(type(error), error)).strip(),
            )

    def _history_entry(self, candidate: CandidateEvaluation, baseline: EvaluationMetrics) -> dict[str, Any]:
        return {
            "recipe": candidate.recipe,
            "status": "measured" if candidate.metrics is not None else "failed",
            "metrics": candidate.metrics.to_dict() if candidate.metrics is not None else None,
            "failure": candidate.failure,
            "gate_failures": self._gate_failures(candidate, baseline),
        }

    def _gate_failures(self, candidate: CandidateEvaluation, baseline: EvaluationMetrics) -> tuple[str, ...]:
        if candidate.metrics is None:
            return ("evaluation_failed",)
        floor = max(self.objectives.min_quality, baseline.gating_quality - self.objectives.max_quality_loss)
        return (f"quality_below_floor:{floor:.4f}",) if candidate.metrics.gating_quality < floor else ()

    def _rank(self, metrics: EvaluationMetrics) -> tuple[float, float]:
        if metrics.cost is None:
            msg = "Metrics must be priced before ranking."
            raise ValueError(msg)
        return (
            (metrics.latency_ms, metrics.cost)
            if self.objectives.primary == "latency"
            else (
                metrics.cost,
                metrics.latency_ms,
            )
        )

    def _candidate_rank(self, candidate: CandidateEvaluation) -> tuple[float, float]:
        return self._rank(candidate.metrics) if candidate.metrics is not None else (float("inf"), float("inf"))

    def _recommendation_reasons(
        self, candidate: CandidateEvaluation, baseline: EvaluationMetrics
    ) -> tuple[str, ...] | None:
        if candidate.metrics is None or self._rank(candidate.metrics) >= self._rank(baseline):
            return None
        reasons: list[str] = []
        if candidate.metrics.details.get("validated") is False:
            reasons.append("quality_unvalidated")
        if candidate.metrics.quality_lower_bound is None:
            reasons.append("single_sample")
        reasons.append("cost_improvement" if self.objectives.primary == "cost" else "latency_improvement")
        return tuple(reasons)


__all__ = [
    "CandidateEvaluation",
    "ExperimentJournal",
    "ExperimentRecommendation",
    "ExperimentResult",
    "HarnessEvaluator",
    "HarnessOptimizationExperiment",
]
