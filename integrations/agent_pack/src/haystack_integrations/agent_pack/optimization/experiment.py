# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from math import isclose
from pathlib import Path
from threading import RLock
from typing import Any

from haystack import Pipeline, logging, tracing
from haystack.components.agents import Agent

from haystack_integrations.agent_pack.optimization.agent import propose_candidate
from haystack_integrations.agent_pack.optimization.models import (
    ModelPriceCatalog,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.workspace import (
    CandidateConfiguration,
    ConfigurationWorkspace,
    content_digest,
    dump_agent,
    dump_pipeline,
    load_agent,
    load_pipeline,
)
from haystack_integrations.evaluation.dataclasses import (
    EvalMetrics,
    ModelTokenUsage,
)
from haystack_integrations.evaluation.harness_evaluator import HarnessEvaluator
from haystack_integrations.evaluation.tracer import EVAL_CASE_SPAN, HarnessSpan, HarnessTracer

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CandidateEvaluation:
    """Journaled raw measurement or failure for one complete YAML configuration."""

    measurement_context: str
    run_id: str
    candidate_id: str
    configuration: CandidateConfiguration | None
    metrics: EvalMetrics | None
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
            metrics=EvalMetrics.from_dict(data=metrics) if metrics is not None else None,
            failure=data.get("failure"),
        )


@dataclass(kw_only=True)
class ExperimentRecommendation:
    """The measured configuration that passed its gates and outranked the reference."""

    configuration: CandidateConfiguration
    evaluation: CandidateEvaluation
    reasons: tuple[str, ...] = ()


@dataclass(kw_only=True)
class ExperimentResult:
    """Priced baseline, candidate outcomes, and the optional best recommendation."""

    baseline: EvalMetrics
    candidates: tuple[CandidateEvaluation, ...]
    recommendation: ExperimentRecommendation | None
    measurement_context: str
    run_id: str
    artifact_directory: Path
    gate_failures: dict[str, tuple[str, ...]] = field(default_factory=dict)
    optimizer_usage: dict[str, ModelTokenUsage] = field(default_factory=dict)
    optimizer_cost: float | None = None


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

    def claim_run(self) -> tuple[str, Path]:
        """
        Allocate the next run identifier and create the directory its artifacts go in.

        Numbered rather than random so a directory of experiments reads in the order they were run. The number is
        claimed by creating the directory, which either succeeds or tells us somebody else has that number: two
        experiments pointed at one journal directory would otherwise both count the same existing runs and pick
        the same next one.

        :returns: The run identifier and its artifact directory.
        """
        with self._lock:
            taken = [
                int(match.group(1))
                for path in self.directory.iterdir()
                if (match := re.fullmatch(r"run-(\d+)(?:\.jsonl)?", path.name))
            ]
            number = max(taken, default=0) + 1
            while True:
                artifacts = self.directory / f"run-{number}"
                try:
                    artifacts.mkdir()
                except FileExistsError:
                    number += 1
                    continue
                return f"run-{number}", artifacts

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


def _fingerprint_eval_cases(eval_cases: list[Any]) -> list[dict[str, Any]]:
    """
    Describe the evaluation set, so two measurements are comparable only when it matches.

    :param eval_cases: The labelled expectations every configuration is measured against.
    :returns: Every eval case as a dictionary, ordered by question.
    """
    return sorted((eval_case.to_dict() for eval_case in eval_cases), key=lambda entry: str(entry["question"]))


def _serialization(reference: Agent | Pipeline) -> tuple[Callable[[Any], str], Callable[[str], Agent | Pipeline]]:
    """
    Pick the pair that round-trips this reference through YAML.

    :param reference: What the experiment optimizes.
    :returns: The dump and load functions for it.
    """
    if isinstance(reference, Pipeline):
        return dump_pipeline, load_pipeline
    return dump_agent, load_agent


class HarnessOptimizationExperiment:
    """Let an optimizer Agent choose, observe, and refine a bounded sequence of configuration experiments."""

    def __init__(
        self,
        reference: Agent | Pipeline,
        evaluator: HarnessEvaluator,
        eval_cases: list[Any],
        pricing: ModelPriceCatalog,
        objectives: OptimizationObjectives,
        journal: ExperimentJournal,
        optimizer_agent: Agent,
        configuration_key: str | None = None,
        max_iterations: int = 8,
        history_digest_window: int = 1,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Configure an iterative, journaled harness optimization run.

        :param reference: Agent or Pipeline used to generate the initial pipeline YAML and baseline. An Agent is
            serialized wrapped in a one-component Pipeline; a Pipeline is serialized as itself.
        :param eval_cases: The labelled expectations every configuration is measured against.
        :param evaluator: Evaluator that measures the reference and each materialized candidate against the
            eval cases.
        :param pricing: Model prices used to calculate candidate costs and rank cost optimizations. Prices do not
            restrict which models the optimizer may choose.
        :param objectives: Quality gates and primary measurement used to rank eligible candidates.
        :param journal: Where every raw measurement the experiment takes is recorded.
        :param optimizer_agent: Agent that edits YAML after observing prior outcomes.
        :param configuration_key: Optional caller-supplied identifier for external measurement inputs, such as a
            corpus or harness version, that cannot be inferred from the serialized Agent and evaluator.
        :param max_iterations: Maximum number of candidate outcomes included in the experiment.
        :param history_digest_window: How many recent outcomes retain detailed traces in optimizer context.
        :param config_path: Optional editable YAML draft, created if absent. Defaults to the artifact directory.
        """
        if max_iterations < 0:
            msg = "max_iterations must be nonnegative."
            raise ValueError(msg)
        self.config_path = config_path
        self.reference = reference
        self.evaluator = evaluator
        self.eval_cases = list(eval_cases)
        self.pricing = pricing
        self.objectives = objectives
        self.journal = journal
        self.optimizer_agent = optimizer_agent
        self.configuration_key = configuration_key
        self.max_iterations = max_iterations
        self.history_digest_window = history_digest_window

    def run(self) -> ExperimentResult:
        """Measure a baseline and a bounded number of validated YAML candidates."""
        dump, load = _serialization(reference=self.reference)
        reference_yaml = dump(self.reference)

        # Fingerprint what makes two measurements comparable: the questions, the yardstick, and the corpus. Not
        # the reference configuration, whose serialization carries incidental values such as an in-memory store's
        # generated index; it is recorded separately as `reference.yaml` and as the baseline's candidate ID.
        payload = {
            "evaluator": type(self.evaluator).__qualname__,
            "eval_cases": _fingerprint_eval_cases(eval_cases=self.eval_cases),
            "configuration_key": self.configuration_key,
        }
        context = content_digest(json.dumps(payload, sort_keys=True, default=str))

        # Claim a run directory and open the single file the optimizer is allowed to edit.
        run_id, artifacts = self.journal.claim_run()
        (artifacts / "reference.yaml").write_text(reference_yaml, encoding="utf-8")
        validator = getattr(self.evaluator, "validate", None)
        draft = artifacts / "candidate.yaml"
        workspace = ConfigurationWorkspace(
            self.config_path or draft,
            reference_yaml,
            validator=validator if callable(validator) else None,
            loader=load,
        )

        # Measure the reference, which every candidate is ranked and gated against.
        ref_target = load(reference_yaml)
        try:
            ref_eval_metrics = self.evaluator.evaluate(target=ref_target, eval_cases=self.eval_cases)
        finally:
            ref_target.close()
        self.journal.append(
            CandidateEvaluation(
                measurement_context=context,
                run_id=run_id,
                candidate_id=workspace.reference_id,
                configuration=None,
                metrics=ref_eval_metrics,
            )
        )
        # Adds cost to the measurement, derived from the model usage the evaluator recorded.
        ref_eval_metrics = self.pricing.price(ref_eval_metrics)
        if self.objectives.primary == "cost" and ref_eval_metrics.cost is None:
            msg = "The reference Agent has unavailable cost; supply pricing and complete usage."
            raise ValueError(msg)

        # Search: propose one candidate, measure it, and feed the outcome into the next proposal.
        outcomes: list[CandidateEvaluation] = []
        history: list[dict[str, Any]] = []
        best_id: str | None = None
        optimizer_usage: dict[str, ModelTokenUsage] = {}
        optimizer_tracer = HarnessTracer()
        while len(outcomes) < self.max_iterations:
            # Measured like a candidate's calls, so the cost of searching is reported beside what it found.
            with optimizer_tracer.activate(), tracing.tracer.trace(EVAL_CASE_SPAN) as turn_span:
                proposed = propose_candidate(
                    optimizer_agent=self.optimizer_agent,
                    workspace=workspace,
                    reference=self.reference,
                    pricing=self.pricing,
                    objectives=self.objectives,
                    baseline=ref_eval_metrics,
                    history=history,
                    history_digest_window=self.history_digest_window,
                    remaining_evaluations=self.max_iterations - len(outcomes),
                    base_id=best_id,
                )
            # An empty summary when a HarnessTracer was not the active tracer.
            collected = turn_span.collected if isinstance(turn_span, HarnessSpan) else None
            for model, tokens in (collected.summarize().models if collected is not None else {}).items():
                current = optimizer_usage.get(model, ModelTokenUsage())
                optimizer_usage[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + tokens.input_tokens,
                    output_tokens=current.output_tokens + tokens.output_tokens,
                )
            # Drafts that failed validation cost editing steps rather than evaluations, and are journalled so
            # the run shows what the optimizer had to repair.
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
                logger.info(
                    "optimizer ended the search with {remaining} of {total} evaluations unused: {reason}",
                    remaining=self.max_iterations - len(outcomes),
                    total=self.max_iterations,
                    reason=workspace.finish_reason or "no reason recorded",
                )
                break

            # Measure the proposal, recording a failure to build or run it rather than ending the experiment.
            (artifacts / f"{proposed.candidate_id}.yaml").write_text(proposed.yaml, encoding="utf-8")
            try:
                candidate = load(proposed.yaml)
                try:
                    metrics = self.evaluator.evaluate(target=candidate, eval_cases=self.eval_cases)
                finally:
                    candidate.close()
                unpriced = CandidateEvaluation(
                    measurement_context=context,
                    run_id=run_id,
                    candidate_id=proposed.candidate_id,
                    configuration=proposed,
                    metrics=metrics,
                )
            except Exception as error:
                unpriced = CandidateEvaluation(
                    measurement_context=context,
                    run_id=run_id,
                    candidate_id=proposed.candidate_id,
                    configuration=proposed,
                    metrics=None,
                    failure="".join(traceback.format_exception_only(type(error), error)).strip(),
                )
            self.journal.append(unpriced)
            priced = unpriced.price(self.pricing)
            outcomes.append(priced)
            history.append(
                {
                    "candidate_id": proposed.candidate_id,
                    "parent_id": proposed.parent_id,
                    "rationale": proposed.rationale,
                    "diff": proposed.diff,
                    "metrics": priced.metrics.to_dict() if priced.metrics is not None else None,
                    "failure": priced.failure,
                    "gate_failures": self._gate_failures(candidate=priced, baseline=ref_eval_metrics),
                }
            )
            # The next turn edits the best candidate so far, so a regression is not inherited by what follows.
            eligible = [
                outcome for outcome in outcomes if not self._gate_failures(candidate=outcome, baseline=ref_eval_metrics)
            ]
            best_id = min(eligible, key=self._candidate_rank).candidate_id if eligible else None

            logger.info(
                "candidate {position}/{total}: {candidate_id}, gates={gates}",
                position=len(outcomes),
                total=self.max_iterations,
                candidate_id=proposed.candidate_id,
                gates=history[-1]["gate_failures"],
            )

        # Recommend the best candidate that clears every gate and actually beats the reference.
        gates = {
            outcome.candidate_id: self._gate_failures(candidate=outcome, baseline=ref_eval_metrics)
            for outcome in outcomes
        }
        eligible = sorted(
            (outcome for outcome in outcomes if not gates[outcome.candidate_id]), key=self._candidate_rank
        )
        recommendation = None
        for evaluated in eligible:
            reasons = self._recommendation_reasons(candidate=evaluated, baseline=ref_eval_metrics)
            if reasons is not None and evaluated.configuration is not None:
                recommendation = ExperimentRecommendation(
                    configuration=evaluated.configuration,
                    evaluation=evaluated,
                    reasons=reasons,
                )
                (artifacts / "recommended.yaml").write_text(evaluated.configuration.yaml, encoding="utf-8")
                break

        # The optimizer is not a candidate, so its spend is priced and reported but never ranked or gated.
        optimizer_cost = self.pricing.cost_of(model_usage=optimizer_usage)
        logger.info(
            "optimizer spend across {turns} turns: {usage} ({cost})",
            turns=len(outcomes),
            usage={model: asdict(obj=tokens) for model, tokens in optimizer_usage.items()},
            cost="unpriced" if optimizer_cost is None else f"${optimizer_cost:.6f}",
        )

        # One file per run naming what its measurements can be compared against.
        (artifacts / "context.json").write_text(
            json.dumps(
                {
                    "measurement_context": context,
                    "run_id": run_id,
                    **payload,
                    "optimizer_usage": {model: asdict(obj=tokens) for model, tokens in optimizer_usage.items()},
                    "optimizer_cost": optimizer_cost,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # A caller-supplied path is theirs to keep; the internal draft is not worth leaving behind.
        if self.config_path is None:
            draft.unlink(missing_ok=True)

        return ExperimentResult(
            baseline=ref_eval_metrics,
            candidates=tuple(outcomes),
            recommendation=recommendation,
            measurement_context=context,
            run_id=run_id,
            artifact_directory=artifacts,
            gate_failures=gates,
            optimizer_usage=optimizer_usage,
            optimizer_cost=optimizer_cost,
        )

    def _gate_failures(self, candidate: CandidateEvaluation, baseline: EvalMetrics) -> tuple[str, ...]:
        """Return the hard gates a candidate failed."""
        if candidate.metrics is None:
            return ("evaluation_failed",)

        failures: list[str] = []
        baseline_quality = baseline.quality
        candidate_quality = candidate.metrics.quality
        floor = max(self.objectives.min_quality, baseline_quality - self.objectives.max_quality_loss)
        # We check isclose to avoid floating-point precision issues where a candidate is effectively equal to the floor
        if candidate_quality < floor and not isclose(candidate_quality, floor, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(f"quality_below_floor:{floor:.4f}")

        # Usage a harness could not account for means a model call was observed and produced nothing measurable,
        # which is what a silently swallowed component failure looks like from here. Such a candidate is not
        # describing its own behaviour, whatever it scored, so it cannot win on any objective.
        if candidate.metrics.details.get("all_tokens_reported") is False:
            failures.append("usage_incomplete")
        elif self.objectives.primary == "cost" and candidate.metrics.cost is None:
            failures.append("cost_unavailable")
        return tuple(failures)

    def _rank(self, metrics: EvalMetrics) -> tuple[float, float]:
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

    def _recommendation_reasons(self, candidate: CandidateEvaluation, baseline: EvalMetrics) -> tuple[str, ...] | None:
        """Explain an improvement or return `None` when the candidate does not beat the baseline."""
        if candidate.metrics is None or self._rank(metrics=candidate.metrics) >= self._rank(metrics=baseline):
            return None
        return (f"{self.objectives.primary}_improvement",)
