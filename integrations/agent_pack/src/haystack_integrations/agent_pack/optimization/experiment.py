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

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics
from haystack_integrations.agent_pack.harness_evaluator import HarnessEvaluator
from haystack_integrations.agent_pack.local_run_store import LocalRunStore
from haystack_integrations.agent_pack.optimization.agent import propose_mutation
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.mutations import (
    AgentMutation,
    apply_mutation,
    rebuild_agent,
)
from haystack_integrations.agent_pack.run_digest import RunDigestPolicy

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


@dataclass(frozen=True, kw_only=True)
class ExperimentRecommendation:
    """The measured configuration that passed its gates and outranked the reference."""

    mutation: AgentMutation
    evaluation: CandidateEvaluation
    reasons: tuple[str, ...] = ()


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


class ExperimentJournal:
    """Append-only JSON-lines record of raw experiment measurements."""

    def __init__(self, path: str | Path) -> None:
        """
        Open an append-only JSON-lines journal.

        The journal records what was measured; it is not read back to measure less. Every experiment measures its
        own reference and its own candidates, so a run's result never depends on what an earlier one happened to
        write, and the file stays a plain log that can be read after the fact.

        :param path: Path to the journal file. Parent directories are created automatically.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def append(self, evaluation: CandidateEvaluation) -> None:
        """
        Record one outcome.

        :param evaluation: The raw measurement or failure to record.
        """
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evaluation.to_dict(), sort_keys=True) + "\n")


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
        optimizer_agent: Agent,
        run_ids: frozenset[str] | None = None,
        configuration_key: str | None = None,
        max_iterations: int = 8,
        digest_policy: RunDigestPolicy | None = None,
        history_digest_window: int = 2,
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
        :param journal: Record of every raw measurement the experiment takes.
        :param optimizer_agent: Agent that chooses each next mutation after observing prior outcomes.
        :param run_ids: Optional identifiers selecting which records to load from `run_store`.
        :param configuration_key: Optional caller-supplied identifier for external measurement inputs, such as a
            corpus or harness version, that cannot be inferred from the serialized Agent and evaluator.
        :param max_iterations: Maximum number of candidate outcomes included in the experiment.
        :param digest_policy: Caps applied when compressing reference runs into the evidence the optimizer reads.
        :param history_digest_window: How many of the most recent outcomes keep their tool traces when the history
            is sent to the optimizer.
        """
        self.reference = reference
        self.run_store = run_store
        self.evaluator = evaluator
        self.pricing = pricing
        self.objectives = objectives
        self.journal = journal
        self.optimizer_agent = optimizer_agent
        self.run_ids = run_ids
        self.configuration_key = configuration_key
        self.max_iterations = max_iterations
        self.digest_policy = digest_policy
        self.history_digest_window = history_digest_window

    def run(self) -> ExperimentResult:
        """Measure the baseline, then iteratively evaluate structured configuration mutations."""
        reference_runs = self.run_store.list(run_ids=self.run_ids)
        if not reference_runs:
            msg = "The selected run store contains no successful reference runs."
            raise ValueError(msg)
        try:
            serialized_reference = self.reference.to_dict()
        except Exception as error:
            msg = f"{type(self.reference).__name__} cannot be serialized and optimized: {error}"
            raise ValueError(msg) from error

        context = self._measurement_context(serialized_reference=serialized_reference, reference_runs=reference_runs)
        logger.info(
            "measuring the reference over {runs} recorded runs, then up to {total} candidates",
            runs=len(reference_runs),
            total=self.max_iterations,
        )
        baseline = self.pricing.price(metrics=self._baseline(context=context, reference_runs=reference_runs))
        if self.objectives.primary == "cost" and baseline.cost is None:
            msg = "The reference Agent uses an unpriced model, so cost cannot be the primary objective."
            raise ValueError(msg)
        outcomes: list[CandidateEvaluation] = []
        mutation_by_id: dict[str, AgentMutation] = {}
        history: list[dict[str, Any]] = []
        seen: set[str] = set()

        reference_fingerprint = _configuration_fingerprint(serialized_agent=serialized_reference)
        stalled = 0
        while len(outcomes) < self.max_iterations and stalled < _MAX_STALLED_PROPOSALS:
            proposed = propose_mutation(
                optimizer_agent=self.optimizer_agent,
                reference=self.reference,
                reference_runs=reference_runs,
                pricing=self.pricing,
                objectives=self.objectives,
                baseline=baseline,
                history=history,
                digest_policy=self.digest_policy,
                history_digest_window=self.history_digest_window,
            )
            if proposed is None:
                break

            try:
                serialized = apply_mutation(serialized_agent=serialized_reference, mutation=proposed)
                fingerprint = _configuration_fingerprint(serialized_agent=serialized)
            except Exception as error:
                stalled = 0
                candidate_id = hashlib.sha256(f"{context}:invalid:{proposed.fingerprint()}".encode()).hexdigest()
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
                    "Candidate configuration could not be mutated for {mutation}: {error}",
                    mutation=proposed,
                    error=error,
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
            raw = self._evaluate_candidate(
                serialized_candidate=serialized,
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

    def _measurement_context(self, serialized_reference: dict[str, Any], reference_runs: list[AgentRunRecord]) -> str:
        """Fingerprint every input that changes a raw Agent measurement."""
        evaluator: dict[str, Any] = {"type": f"{type(self.evaluator).__module__}.{type(self.evaluator).__qualname__}"}
        if callable(fingerprint := getattr(self.evaluator, "fingerprint", None)):
            evaluator["configuration"] = fingerprint()
        payload = {
            "reference": _stable_serialization(value=serialized_reference),
            "evaluator": evaluator,
            "runs": sorted(record.fingerprint() for record in reference_runs),
            "configuration_key": self.configuration_key,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _baseline(self, context: str, reference_runs: list[AgentRunRecord]) -> EvaluationMetrics:
        """Load or measure the reference Agent for the current context."""
        candidate_id = f"baseline:{context}"
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
        serialized_candidate: dict[str, Any],
        mutation: AgentMutation,
        candidate_id: str,
        context: str,
        reference_runs: list[AgentRunRecord],
    ) -> CandidateEvaluation:
        """Rebuild and measure one candidate configuration, converting failures into journal records."""
        try:
            candidate = rebuild_agent(serialized_agent=serialized_candidate)
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
        baseline_quality = baseline.quality
        candidate_quality = candidate.metrics.quality
        floor = max(self.objectives.min_quality, baseline_quality - self.objectives.max_quality_loss)
        if candidate_quality < floor:
            failures.append(f"quality_below_floor:{floor:.4f}")
        if self.objectives.primary == "cost" and candidate.metrics.cost is None:
            failures.append("cost_unavailable")
        return tuple(failures)

    def _rank(self, metrics: EvaluationMetrics) -> tuple[float, float]:
        """Return the objective-dependent ordering key for priced metrics, lower being better."""
        cost = metrics.cost if metrics.cost is not None else float("inf")
        if self.objectives.primary == "quality":
            # Negated so that more quality sorts first, with cost deciding between equally good answers. Ranking
            # on quality needs no threshold, and cannot rank a configuration that answers worse above one that
            # answers better however cheap it is.
            return (-metrics.quality, cost)
        if self.objectives.primary == "latency":
            return (metrics.latency_ms, cost)
        return (cost, metrics.latency_ms)

    def _candidate_rank(self, candidate: CandidateEvaluation) -> tuple[float, float]:
        """Return a sortable rank that places failed candidates last."""
        return self._rank(metrics=candidate.metrics) if candidate.metrics is not None else (float("inf"), float("inf"))

    def _recommendation_reasons(
        self, candidate: CandidateEvaluation, baseline: EvaluationMetrics
    ) -> tuple[str, ...] | None:
        """Explain an improvement or return `None` when the candidate does not beat the baseline."""
        if candidate.metrics is None or self._rank(metrics=candidate.metrics) >= self._rank(metrics=baseline):
            return None
        reasons: list[str] = []
        if candidate.metrics.details.get("validated") is False:
            reasons.append("quality_unvalidated")
        reasons.append(f"{self.objectives.primary}_improvement")
        return tuple(reasons)
