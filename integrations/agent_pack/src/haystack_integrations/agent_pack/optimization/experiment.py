# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Iterative measurement and recommendation of Agent configuration candidates."""

import hashlib
import json
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any

from haystack import logging
from haystack.components.agents import Agent
from pydantic import ValidationError

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics
from haystack_integrations.agent_pack.harness_evaluator import HarnessEvaluator
from haystack_integrations.agent_pack.local_run_store import LocalRunStore
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.mutations import (
    AgentMutation,
    materialize_mutation,
    mutation_fingerprint,
)
from haystack_integrations.agent_pack.optimization.proposer import MutationProposer

logger = logging.getLogger(__name__)
_MAX_STALLED_PROPOSALS = 3


@dataclass(frozen=True, kw_only=True)
class CandidateEvaluation:
    """Journaled raw measurement or failure for one resolved configuration mutation."""

    measurement_context: str
    candidate_id: str
    mutation: dict[str, Any] | None
    metrics: EvaluationMetrics | None
    failure: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether evaluation produced metrics."""
        return self.failure is None and self.metrics is not None

    def price(self, pricing: ModelPriceCatalog) -> "CandidateEvaluation":
        """Apply current prices without changing the journaled raw measurement."""
        return replace(self, metrics=pricing.price(metrics=self.metrics) if self.metrics is not None else None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible journal record."""
        return {
            "measurement_context": self.measurement_context,
            "candidate_id": self.candidate_id,
            "mutation": self.mutation,
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
            mutation=data.get("mutation"),
            metrics=EvaluationMetrics.from_dict(data=metrics) if metrics is not None else None,
            failure=data.get("failure"),
        )


class ExperimentJournal:
    """Append-only JSON-lines persistence for raw experiment measurements."""

    def __init__(self, path: str | Path) -> None:
        """
        Load existing measurements from an append-only JSON-lines journal.

        :param path: Path to the journal file. Parent directories are created automatically, and an existing journal
            is loaded so completed measurements can be reused.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: dict[str, CandidateEvaluation] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = CandidateEvaluation.from_dict(data=json.loads(line))
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
                and record.mutation is not None
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
    """The measured configuration that passed its gates and outranked the reference."""

    mutation: AgentMutation
    evaluation: CandidateEvaluation
    reasons: tuple[str, ...] = ()

    def materialize(self, reference: Agent) -> Agent:
        """Rebuild the recommendation for inspection or approval."""
        candidate, _ = materialize_mutation(reference=reference, mutation=self.mutation)
        return candidate


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
    """Remove only process-local InMemoryDocumentStore identity from serialized Agent data."""
    if isinstance(value, list):
        return [_stable_serialization(value=item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: _stable_serialization(value=item) for key, item in value.items()}
    type_name = normalized.get("type")
    if isinstance(type_name, str) and type_name.endswith("InMemoryDocumentStore"):
        parameters = normalized.get("init_parameters")
        if isinstance(parameters, dict):
            parameters.pop("index", None)
    return normalized


def _configuration_fingerprint(serialized_agent: dict[str, Any]) -> str:
    """Fingerprint the stable form of a complete resulting Agent configuration."""
    payload = json.dumps(_stable_serialization(value=serialized_agent), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class HarnessOptimizationExperiment:
    """Let an optimizer Agent choose, observe, and refine a bounded sequence of configuration experiments."""

    def __init__(
        self,
        reference: Agent,
        run_store: LocalRunStore,
        evaluator: HarnessEvaluator,
        pricing: ModelPriceCatalog,
        objectives: OptimizationObjectives,
        journal: ExperimentJournal,
        proposer: MutationProposer,
        run_ids: frozenset[str] | None = None,
        configuration_key: str | None = None,
        max_iterations: int = 8,
    ) -> None:
        """
        Configure an iterative, journaled harness optimization run.

        :param reference: The unchanged Agent used as the baseline and as the source configuration for every
            candidate mutation.
        :param run_store: Local store of successful Agent runs whose inputs are replayed during evaluation.
        :param evaluator: Evaluator that measures the reference and each materialized candidate against the selected
            runs.
        :param pricing: Model prices used to calculate candidate costs and rank cost optimizations. Prices do not
            restrict which models the optimizer may choose.
        :param objectives: Quality gates and primary measurement used to rank eligible candidates.
        :param journal: Persistent measurement journal used to resume compatible experiments without repeating work.
        :param proposer: Strategy that chooses each next mutation after observing prior outcomes.
        :param run_ids: Optional identifiers selecting which records to load from `run_store`.
        :param configuration_key: Optional caller-supplied identifier for external measurement inputs, such as a
            corpus or harness version, that cannot be inferred from the serialized Agent and evaluator.
        :param max_iterations: Maximum number of candidate outcomes included in the experiment, counting compatible
            completed measurements loaded from the journal.
        """
        self.reference = reference
        self.run_store = run_store
        self.evaluator = evaluator
        self.pricing = pricing
        self.objectives = objectives
        self.journal = journal
        self.proposer = proposer
        self.run_ids = run_ids
        self.configuration_key = configuration_key
        self.max_iterations = max_iterations

    def run(self) -> ExperimentResult:
        """Measure the baseline, then iteratively evaluate structured configuration mutations."""
        reference_runs = self.run_store.list(run_ids=self.run_ids)
        if not reference_runs:
            msg = "The selected run store contains no successful reference runs."
            raise ValueError(msg)

        context = self._measurement_context(reference_runs=reference_runs)
        baseline = self.pricing.price(metrics=self._baseline(context=context, reference_runs=reference_runs))
        if self.objectives.primary == "cost" and baseline.cost is None:
            msg = "The reference Agent uses an unpriced model, so cost cannot be the primary objective."
            raise ValueError(msg)
        outcomes: list[CandidateEvaluation] = []
        mutation_by_id: dict[str, AgentMutation] = {}
        history: list[dict[str, Any]] = []
        seen: set[str] = set()

        for prior in self.journal.completed(measurement_context=context):
            try:
                mutation = AgentMutation.model_validate(prior.mutation)
                _, serialized = materialize_mutation(reference=self.reference, mutation=mutation)
            except (ValueError, ValidationError):
                continue
            fingerprint = _configuration_fingerprint(serialized_agent=serialized)
            expected_id = hashlib.sha256(f"{context}:{fingerprint}".encode()).hexdigest()
            if prior.candidate_id != expected_id:
                continue
            seen.add(fingerprint)
            priced = prior.price(pricing=self.pricing)
            outcomes.append(priced)
            mutation_by_id[priced.candidate_id] = mutation
            history.append(self._history_entry(candidate=priced, baseline=baseline))

        reference_fingerprint = _configuration_fingerprint(serialized_agent=self.reference.to_dict())
        stalled = 0
        while len(outcomes) < self.max_iterations and stalled < _MAX_STALLED_PROPOSALS:
            proposed = self.proposer.propose(
                reference=self.reference,
                reference_runs=reference_runs,
                pricing=self.pricing,
                objectives=self.objectives,
                baseline=baseline,
                history=history,
            )
            if proposed is None:
                break

            try:
                candidate_agent, serialized = materialize_mutation(reference=self.reference, mutation=proposed)
                fingerprint = _configuration_fingerprint(serialized_agent=serialized)
            except Exception as error:
                stalled = 0
                invalid_fingerprint = mutation_fingerprint(mutation=proposed)
                candidate_id = hashlib.sha256(f"{context}:invalid:{invalid_fingerprint}".encode()).hexdigest()
                failure = "".join(traceback.format_exception_only(type(error), error)).strip()
                outcome = CandidateEvaluation(
                    measurement_context=context,
                    candidate_id=candidate_id,
                    mutation=proposed.model_dump(),
                    metrics=None,
                    failure=failure,
                )
                outcomes.append(outcome)
                history.append(self._history_entry(candidate=outcome, baseline=baseline))
                logger.warning(
                    "Candidate materialization failed for {mutation}: {error}", mutation=proposed, error=error
                )
                continue

            if fingerprint == reference_fingerprint or fingerprint in seen:
                stalled += 1
                history.append(
                    {"mutation": proposed.model_dump(), "status": "rejected", "reason": "duplicate_or_no_op"}
                )
                continue
            stalled = 0
            seen.add(fingerprint)
            candidate_id = hashlib.sha256(f"{context}:{fingerprint}".encode()).hexdigest()
            raw = self.journal.get(candidate_id=candidate_id)
            if raw is None:
                raw = self._evaluate_candidate(
                    candidate=candidate_agent,
                    mutation=proposed,
                    candidate_id=candidate_id,
                    context=context,
                    reference_runs=reference_runs,
                )
                self.journal.append(evaluation=raw)
            priced = raw.price(pricing=self.pricing)
            outcomes.append(priced)
            mutation_by_id[candidate_id] = proposed
            history.append(self._history_entry(candidate=priced, baseline=baseline))

        gate_failures = {
            outcome.candidate_id: self._gate_failures(candidate=outcome, baseline=baseline) for outcome in outcomes
        }
        eligible = [outcome for outcome in outcomes if not gate_failures[outcome.candidate_id]]
        eligible.sort(key=self._candidate_rank)
        recommendation = None
        for eligible_candidate in eligible:
            if (reasons := self._recommendation_reasons(candidate=eligible_candidate, baseline=baseline)) is not None:
                recommendation = ExperimentRecommendation(
                    mutation=mutation_by_id[eligible_candidate.candidate_id],
                    evaluation=eligible_candidate,
                    reasons=reasons,
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
        """Fingerprint every input that changes a raw Agent measurement."""
        try:
            reference = _stable_serialization(value=self.reference.to_dict())
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
        """Load or measure the reference Agent for the current context."""
        candidate_id = f"baseline:{context}"
        if (prior := self.journal.get(candidate_id=candidate_id)) is not None and prior.metrics is not None:
            return prior.metrics
        metrics = self.evaluator.evaluate(agent=self.reference, reference_runs=reference_runs)
        self.journal.append(
            evaluation=CandidateEvaluation(
                measurement_context=context,
                candidate_id=candidate_id,
                mutation=None,
                metrics=metrics,
            )
        )
        return metrics

    def _evaluate_candidate(
        self,
        candidate: Agent,
        mutation: AgentMutation,
        candidate_id: str,
        context: str,
        reference_runs: list[AgentRunRecord],
    ) -> CandidateEvaluation:
        """Measure one materialized candidate, converting failures into journal records."""
        try:
            metrics = self.evaluator.evaluate(agent=candidate, reference_runs=reference_runs)
            return CandidateEvaluation(
                measurement_context=context,
                candidate_id=candidate_id,
                mutation=mutation.model_dump(),
                metrics=metrics,
            )
        except Exception as error:
            logger.warning("Candidate evaluation failed for {mutation}: {error}", mutation=mutation, error=error)
            return CandidateEvaluation(
                measurement_context=context,
                candidate_id=candidate_id,
                mutation=mutation.model_dump(),
                metrics=None,
                failure="".join(traceback.format_exception_only(type(error), error)).strip(),
            )

    def _history_entry(self, candidate: CandidateEvaluation, baseline: EvaluationMetrics) -> dict[str, Any]:
        """Describe one outcome for the optimizer Agent's next decision."""
        return {
            "mutation": candidate.mutation,
            "status": "measured" if candidate.metrics is not None else "failed",
            "metrics": candidate.metrics.to_dict() if candidate.metrics is not None else None,
            "failure": candidate.failure,
            "gate_failures": self._gate_failures(candidate=candidate, baseline=baseline),
        }

    def _gate_failures(self, candidate: CandidateEvaluation, baseline: EvaluationMetrics) -> tuple[str, ...]:
        """Return the hard gates a candidate failed."""
        if candidate.metrics is None:
            return ("evaluation_failed",)
        failures: list[str] = []
        baseline_quality = self._gating_quality(metrics=baseline)
        candidate_quality = self._gating_quality(metrics=candidate.metrics)
        floor = max(self.objectives.min_quality, baseline_quality - self.objectives.max_quality_loss)
        if candidate_quality < floor:
            failures.append(f"quality_below_floor:{floor:.4f}")
        if self.objectives.primary == "cost" and candidate.metrics.cost is None:
            failures.append("cost_unavailable")
        return tuple(failures)

    def _rank(self, metrics: EvaluationMetrics) -> tuple[float, float]:
        """Return the objective-dependent ordering key for priced metrics."""
        cost = metrics.cost if metrics.cost is not None else float("inf")
        return (metrics.latency_ms, cost) if self.objectives.primary == "latency" else (cost, metrics.latency_ms)

    @staticmethod
    def _gating_quality(metrics: EvaluationMetrics) -> float:
        """Return the conservative quality value used by optimization gates."""
        return metrics.quality if metrics.quality_lower_bound is None else metrics.quality_lower_bound

    def _candidate_rank(self, candidate: CandidateEvaluation) -> tuple[float, float]:
        """Return a sortable rank that places failed candidates last."""
        return self._rank(metrics=candidate.metrics) if candidate.metrics is not None else (float("inf"), float("inf"))

    def _recommendation_reasons(
        self, candidate: CandidateEvaluation, baseline: EvaluationMetrics
    ) -> tuple[str, ...] | None:
        """Explain an improvement or return ``None`` when the candidate does not beat the baseline."""
        if candidate.metrics is None or self._rank(metrics=candidate.metrics) >= self._rank(metrics=baseline):
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
    "HarnessOptimizationExperiment",
]
