# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Iterative measurement and recommendation of Agent configuration candidates."""

import json
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from haystack import logging
from haystack.components.agents import Agent

from haystack_integrations.agent_pack.dataclasses import EvaluationMetrics, content_digest
from haystack_integrations.agent_pack.harness_evaluator import HarnessEvaluator
from haystack_integrations.agent_pack.local_run_store import LocalRunStore
from haystack_integrations.agent_pack.optimization.agent import propose_candidate
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.workspace import (
    CandidateConfiguration,
    ConfigurationWorkspace,
    configuration_id,
    dump_agent,
    load_agent,
)
from haystack_integrations.agent_pack.run_digest import RunDigestPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class CandidateEvaluation:
    """Journaled raw measurement or failure for one complete YAML configuration."""

    measurement_context: str
    run_id: str
    candidate_id: str
    configuration: CandidateConfiguration | None
    metrics: EvaluationMetrics | None
    failure: str | None = None

    def price(self, pricing: ModelPriceCatalog) -> "CandidateEvaluation":
        """Apply current prices without changing the journaled raw measurement."""
        return replace(self, metrics=pricing.price(metrics=self.metrics) if self.metrics is not None else None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible journal record."""
        return {
            "measurement_context": self.measurement_context,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "configuration": asdict(self.configuration) if self.configuration is not None else None,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
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
            metrics=EvaluationMetrics.from_dict(data=metrics) if metrics is not None else None,
            failure=data.get("failure"),
        )


@dataclass(frozen=True, kw_only=True)
class ExperimentRecommendation:
    """The measured configuration that passed its gates and outranked the reference."""

    configuration: CandidateConfiguration
    evaluation: CandidateEvaluation
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ExperimentResult:
    """Priced baseline, candidate outcomes, and the optional best recommendation."""

    baseline: EvaluationMetrics
    candidates: tuple[CandidateEvaluation, ...]
    recommendation: ExperimentRecommendation | None
    measurement_context: str
    run_id: str
    artifact_directory: Path
    gate_failures: dict[str, tuple[str, ...]] = field(default_factory=dict)


class ExperimentJournal:
    """Append-only JSON-lines record of raw experiment measurements, one file per experiment."""

    def __init__(self, directory: str | Path) -> None:
        """
        Open a directory of experiment journals.

        The journal records what was measured; it is not read back to measure less. Every experiment measures its
        own reference and its own candidates, so a run's result never depends on what an earlier one wrote.

        Each invocation gets a unique run ID and journal, even when its measurement context matches an earlier run.

        :param directory: Directory to write journals into. Created automatically.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def path_for(self, run_id: str) -> Path:
        """
        Return the journal file holding one experiment's measurements.

        :param run_id: Unique identifier of one experiment invocation.
        :returns: Path to that experiment's journal.
        """
        return self.directory / f"{run_id}.jsonl"

    def append(self, evaluation: CandidateEvaluation) -> None:
        """
        Record one outcome, in the journal of the experiment it belongs to.

        :param evaluation: The raw measurement or failure to record.
        """
        # Keys are written in the order `to_dict` builds them rather than sorted, so each line opens with the
        # context and the candidate it describes.
        with (
            self._lock,
            self.path_for(run_id=evaluation.run_id).open("a", encoding="utf-8") as stream,
        ):
            stream.write(json.dumps(evaluation.to_dict()) + "\n")


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
        history_digest_window: int = 1,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Configure an iterative, journaled harness optimization run.

        :param reference: Agent used to generate the initial pipeline YAML and baseline.
        :param run_store: Local store of successful Agent runs whose inputs are replayed during evaluation.
        :param evaluator: Evaluator that measures the reference and each materialized candidate against the selected
            runs.
        :param pricing: Model prices used to calculate candidate costs and rank cost optimizations. Prices do not
            restrict which models the optimizer may choose.
        :param objectives: Quality gates and primary measurement used to rank eligible candidates.
        :param journal: Where every raw measurement the experiment takes is recorded.
        :param optimizer_agent: Agent that edits YAML after observing prior outcomes.
        :param run_ids: Optional identifiers selecting which records to load from `run_store`.
        :param configuration_key: Optional caller-supplied identifier for external measurement inputs, such as a
            corpus or harness version, that cannot be inferred from the serialized Agent and evaluator.
        :param max_iterations: Maximum number of candidate outcomes included in the experiment.
        :param digest_policy: Caps applied when compressing reference runs into the evidence the optimizer reads.
        :param history_digest_window: How many recent outcomes retain detailed traces in optimizer context.
        :param config_path: Optional editable YAML draft, created if absent. Defaults to the artifact directory.
        """
        if max_iterations < 0:
            msg = "max_iterations must be nonnegative."
            raise ValueError(msg)
        self.config_path = config_path
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
        """Measure a baseline and a bounded number of validated YAML candidates."""
        reference_runs = self.run_store.list(run_ids=self.run_ids)
        if not reference_runs:
            msg = "The selected run store contains no successful reference runs."
            raise ValueError(msg)
        reference_yaml = dump_agent(self.reference)
        payload = {
            "reference": configuration_id(reference_yaml),
            "runs": sorted(record.fingerprint() for record in reference_runs),
            "evaluator": type(self.evaluator).__qualname__,
            "configuration_key": self.configuration_key,
        }
        if callable(fingerprint := getattr(self.evaluator, "fingerprint", None)):
            payload["evaluator_configuration"] = fingerprint()
        context = content_digest(json.dumps(payload, sort_keys=True, default=str))
        run_id = uuid4().hex[:12]
        artifacts = self.journal.directory / run_id
        artifacts.mkdir()
        (artifacts / "reference.yaml").write_text(reference_yaml, encoding="utf-8")
        validator = getattr(self.evaluator, "validate_agent", None)
        workspace = ConfigurationWorkspace(
            self.config_path or artifacts / "candidate.yaml",
            reference_yaml,
            validator=validator if callable(validator) else None,
        )
        baseline_agent = load_agent(reference_yaml)
        try:
            baseline_raw = self.evaluator.evaluate(agent=baseline_agent, reference_runs=reference_runs)
        finally:
            baseline_agent.close()
        self.journal.append(
            CandidateEvaluation(
                measurement_context=context,
                run_id=run_id,
                candidate_id=workspace.reference_id,
                configuration=None,
                metrics=baseline_raw,
            )
        )
        baseline = self.pricing.price(baseline_raw)
        if self.objectives.primary == "cost" and baseline.cost is None:
            msg = "The reference Agent has unavailable cost; supply pricing and complete usage."
            raise ValueError(msg)
        outcomes: list[CandidateEvaluation] = []
        history: list[dict[str, Any]] = []
        while len(outcomes) < self.max_iterations:
            proposed = propose_candidate(
                optimizer_agent=self.optimizer_agent,
                workspace=workspace,
                reference=self.reference,
                reference_runs=reference_runs,
                pricing=self.pricing,
                objectives=self.objectives,
                baseline=baseline,
                history=history,
                digest_policy=self.digest_policy,
                history_digest_window=self.history_digest_window,
            )
            for failure in workspace.validation_failures:
                self.journal.append(
                    CandidateEvaluation(
                        measurement_context=context,
                        run_id=run_id,
                        candidate_id=failure["revision"],
                        configuration=None,
                        metrics=None,
                        failure=failure["error"],
                    )
                )
            workspace.validation_failures.clear()
            if proposed is None:
                break
            (artifacts / f"{proposed.candidate_id}.yaml").write_text(proposed.yaml, encoding="utf-8")
            try:
                candidate = load_agent(proposed.yaml)
                try:
                    metrics = self.evaluator.evaluate(agent=candidate, reference_runs=reference_runs)
                finally:
                    candidate.close()
                raw = CandidateEvaluation(
                    measurement_context=context,
                    run_id=run_id,
                    candidate_id=proposed.candidate_id,
                    configuration=proposed,
                    metrics=metrics,
                )
            except Exception as error:
                raw = CandidateEvaluation(
                    measurement_context=context,
                    run_id=run_id,
                    candidate_id=proposed.candidate_id,
                    configuration=proposed,
                    metrics=None,
                    failure="".join(traceback.format_exception_only(type(error), error)).strip(),
                )
            self.journal.append(raw)
            priced = raw.price(self.pricing)
            outcomes.append(priced)
            history.append(
                {
                    "candidate_id": proposed.candidate_id,
                    "parent_id": proposed.parent_id,
                    "rationale": proposed.rationale,
                    "diff": proposed.diff,
                    "metrics": priced.metrics.to_dict() if priced.metrics is not None else None,
                    "failure": priced.failure,
                    "gate_failures": self._gate_failures(priced, baseline),
                }
            )
            logger.info(
                "candidate {position}/{total}: {candidate_id}, gates={gates}",
                position=len(outcomes),
                total=self.max_iterations,
                candidate_id=proposed.candidate_id,
                gates=history[-1]["gate_failures"],
            )
        gates = {outcome.candidate_id: self._gate_failures(outcome, baseline) for outcome in outcomes}
        eligible = sorted(
            (outcome for outcome in outcomes if not gates[outcome.candidate_id]), key=self._candidate_rank
        )
        recommendation = None
        for evaluated in eligible:
            reasons = self._recommendation_reasons(evaluated, baseline)
            if reasons is not None and evaluated.configuration is not None:
                recommendation = ExperimentRecommendation(
                    configuration=evaluated.configuration,
                    evaluation=evaluated,
                    reasons=reasons,
                )
                (artifacts / "recommended.yaml").write_text(evaluated.configuration.yaml, encoding="utf-8")
                break
        (artifacts / "context.json").write_text(
            json.dumps({"measurement_context": context, "run_id": run_id, **payload}, indent=2, default=str),
            encoding="utf-8",
        )
        return ExperimentResult(
            baseline=baseline,
            candidates=tuple(outcomes),
            recommendation=recommendation,
            measurement_context=context,
            run_id=run_id,
            artifact_directory=artifacts,
            gate_failures=gates,
        )

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
