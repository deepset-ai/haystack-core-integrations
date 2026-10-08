# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from haystack import Pipeline, logging, tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat.types import ChatGenerator

from haystack_integrations.agent_pack.evaluation.dataclasses import (
    EvalMetrics,
    ModelPrice,
    ModelTokenUsage,
    cost_of_model_usage,
)
from haystack_integrations.agent_pack.evaluation.harness_evaluator import HarnessEvaluator
from haystack_integrations.agent_pack.evaluation.tracer import EVAL_CASE_SPAN, HarnessSpan, HarnessTracer
from haystack_integrations.agent_pack.optimization.agent import create_harness_optimizer_agent, propose_candidate
from haystack_integrations.agent_pack.optimization.dataclasses import (
    CandidateConfiguration,
    CandidateOutcome,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.utils import (
    _configuration_id,
    content_digest,
    dump_agent,
    dump_pipeline,
    load_agent,
    load_pipeline,
)

logger = logging.getLogger(__name__)


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
    evaluation: "CandidateEvaluation"
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


class ExperimentJournal:
    """Append-only JSON-lines record of raw experiment measurements, one file per experiment."""

    def __init__(self, directory: str | Path) -> None:
        """
        Open a directory of experiment journals.

        Each run gets its own run ID and journal file, even when its measurement context matches an earlier run.

        :param directory: Directory to write journals into. Created automatically.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def claim_run(self) -> tuple[str, Path]:
        """
        Allocate the next run identifier and create the directory its artifacts go in.

        Runs are numbered in the order they start. A number is claimed by creating its directory, so two
        experiments sharing one journal directory never get the same number.

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
        # Keys keep the `to_dict` order, so each line starts with the context and the candidate
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


def _describe_setting(value: Any) -> Any:
    """
    Reduce one evaluator setting to plain JSON that comes out the same on every run.

    :param value: An attribute the evaluator was configured with.
    :returns: The value as JSON, with sets and mapping keys in a fixed order, and any other object named by its type
        so a memory address never reaches the fingerprint.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _describe_setting(value=asdict(value))
    if isinstance(value, dict):
        return dict(sorted((_setting_key(key=key), _describe_setting(value=item)) for key, item in value.items()))
    if isinstance(value, (set, frozenset)):
        return sorted((_describe_setting(value=item) for item in value), key=json.dumps)
    if isinstance(value, (list, tuple)):
        return [_describe_setting(value=item) for item in value]
    return type(value).__qualname__


def _setting_key(key: Any) -> str:
    """
    Name a mapping key as a string, joining a group of tool names in a fixed order.

    :param key: A key from an evaluator setting, such as one tool name or a tuple of them.
    :returns: The key as a string.
    """
    if isinstance(key, (tuple, list, set, frozenset)):
        return "+".join(sorted(str(part) for part in key))
    return str(key)


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
        prices: dict[str, ModelPrice],
        objectives: OptimizationObjectives,
        journal: ExperimentJournal,
        optimizer_llm: ChatGenerator | None = None,
        optimizer_system_prompt: str | None = None,
        optimizer_additional_instructions: str | None = None,
        optimizer_max_agent_steps: int = 24,
        optimizer_documentation_tools: bool = False,
        configuration_key: str | None = None,
        max_iterations: int = 8,
        history_digest_window: int = 1,
        on_baseline: Callable[[EvalMetrics, float | None], None] | None = None,
        on_candidate: Callable[[CandidateProgress], None] | None = None,
    ) -> None:
        """
        Configure an iterative, journaled harness optimization run.

        :param reference: Agent or Pipeline used to generate the initial pipeline YAML and baseline. An Agent is
            serialized wrapped in a one-component Pipeline; a Pipeline is serialized as itself.
        :param eval_cases: The labelled expectations every configuration is measured against.
        :param evaluator: Evaluator that measures the reference and each materialized candidate against the
            eval cases.
        :param prices: Token prices keyed by model identifier, used to calculate candidate costs and rank cost
            optimizations. Prices do not restrict which models the optimizer may choose.
        :param objectives: Quality gates and primary measurement used to rank eligible candidates.
        :param journal: Where every raw measurement the experiment takes is recorded.
        :param optimizer_llm: LLM the optimizer agent reasons with. Defaults to the one `create_harness_optimizer_agent`
            chooses.
        :param optimizer_system_prompt: Overrides the optimizer agent's pre-made system prompt.
        :param optimizer_additional_instructions: Guidance appended to the optimizer agent's system prompt about what
            a good configuration looks like for this harness.
        :param optimizer_max_agent_steps: Maximum steps the optimizer agent takes to produce one candidate.
        :param optimizer_documentation_tools: Give the optimizer agent read-only search over the public Haystack
            documentation. Requires `mcp-haystack`.
        :param configuration_key: Optional caller-supplied identifier for external measurement inputs, such as a
            corpus or harness version, that cannot be inferred from the serialized Agent and evaluator.
        :param max_iterations: Maximum number of candidate outcomes included in the experiment.
        :param history_digest_window: How many of the most recent outcomes are shown to the optimizer in full.
        :param on_baseline: Called with the reference measurement and its cost before the search starts.
        :param on_candidate: Called with each candidate's `CandidateProgress` right after it is measured, priced
            and gated.
        """
        if max_iterations < 0:
            msg = "max_iterations must be nonnegative."
            raise ValueError(msg)
        self.reference = reference
        self.evaluator = evaluator
        self.eval_cases = list(eval_cases)
        self.prices = prices
        self.objectives = objectives
        self.journal = journal
        # The optimizer validates every draft with this experiment's evaluator, and loads it as the same kind of
        # configuration as the reference
        _, load = _serialization(reference=reference)
        self.optimizer_agent = create_harness_optimizer_agent(
            evaluator=evaluator,
            loader=load,
            llm=optimizer_llm,
            system_prompt=optimizer_system_prompt,
            max_agent_steps=optimizer_max_agent_steps,
            additional_instructions=optimizer_additional_instructions,
            documentation_tools=optimizer_documentation_tools,
        )
        self.configuration_key = configuration_key
        self.max_iterations = max_iterations
        self.history_digest_window = history_digest_window
        self.on_baseline = on_baseline
        self.on_candidate = on_candidate

    def run(self) -> ExperimentResult:
        """Measure a baseline and a bounded number of validated YAML candidates."""
        dump, load = _serialization(reference=self.reference)
        reference_yaml = dump(self.reference)

        # Fingerprint what makes two measurements comparable: the evaluator and its settings, the eval cases and the
        # configuration key. The reference YAML is left out, since its serialization carries incidental values such
        # as an in-memory store's generated index; it is saved as `reference.yaml` instead.
        payload = {
            "evaluator": type(self.evaluator).__qualname__,
            "evaluator_settings": _describe_setting(value=vars(self.evaluator)),
            "eval_cases": _fingerprint_eval_cases(eval_cases=self.eval_cases),
            "configuration_key": self.configuration_key,
        }
        context = content_digest(payload=json.dumps(payload, sort_keys=True, default=str))

        # Claim a run directory for this run's artifacts
        run_id, artifacts = self.journal.claim_run()
        (artifacts / "reference.yaml").write_text(reference_yaml, encoding="utf-8")
        reference_id = _configuration_id(reference_yaml)

        # Measure the reference, which every candidate is ranked and gated against.
        ref_target = load(reference_yaml)
        try:
            ref_eval_metrics = self.evaluator.evaluate(target=ref_target, eval_cases=self.eval_cases)
        finally:
            ref_target.close()
        ref_cost = cost_of_model_usage(model_usage=ref_eval_metrics.model_usage, prices=self.prices)
        self.journal.append(
            CandidateEvaluation(
                measurement_context=context,
                run_id=run_id,
                candidate_id=reference_id,
                configuration=None,
                metrics=ref_eval_metrics,
                cost=ref_cost,
            )
        )
        if self.objectives.primary == "cost" and (ref_cost is None or not ref_eval_metrics.all_tokens_reported):
            msg = "The reference has unavailable cost; supply prices for its models and complete usage."
            raise ValueError(msg)
        self.objectives.get_quality(metrics=ref_eval_metrics)
        if self.on_baseline is not None:
            self.on_baseline(ref_eval_metrics, ref_cost)

        # Search: propose one candidate, measure it, and feed the outcome into the next proposal.
        outcomes: list[CandidateEvaluation] = []
        history: list[CandidateOutcome] = []
        best_id: str | None = None
        optimizer_usage: dict[str, ModelTokenUsage] = {}
        optimizer_tracer = HarnessTracer()
        while len(outcomes) < self.max_iterations:
            # Each turn starts from the best candidate so far, or from the last one submitted when none has cleared
            # the gates yet, and can restore any configuration already tried
            last_id = outcomes[-1].candidate_id if outcomes else None
            # Trace the optimizer's own model calls, so the search's spend can be reported
            with optimizer_tracer.activate(), tracing.tracer.trace(EVAL_CASE_SPAN) as turn_span:
                turn = propose_candidate(
                    optimizer_agent=self.optimizer_agent,
                    reference=self.reference,
                    reference_yaml=reference_yaml,
                    prices=self.prices,
                    objectives=self.objectives,
                    baseline=ref_eval_metrics,
                    history=history,
                    history_digest_window=self.history_digest_window,
                    remaining_evaluations=self.max_iterations - len(outcomes),
                    candidates={
                        outcome.candidate_id: outcome.configuration.yaml
                        for outcome in outcomes
                        if outcome.configuration is not None
                    },
                    base_id=best_id or last_id,
                )
            # An empty summary when a HarnessTracer was not the active tracer.
            collected = turn_span.collected if isinstance(turn_span, HarnessSpan) else None
            for model, tokens in (collected.summarize().model_usage if collected is not None else {}).items():
                optimizer_usage[model] = optimizer_usage.get(model, ModelTokenUsage()) + tokens
            # Journal drafts that failed validation, so the run shows what the optimizer had to repair
            for failure in turn.validation_failures:
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

            proposed = turn.candidate
            if proposed is None:
                logger.info(
                    "optimizer ended the search with {remaining} of {total} evaluations unused: {reason}",
                    remaining=self.max_iterations - len(outcomes),
                    total=self.max_iterations,
                    reason=turn.finish_reason or "no reason recorded",
                )
                break

            # Measure the proposal; a failure to build or run it is recorded and the search continues
            (artifacts / f"{proposed.candidate_id}.yaml").write_text(proposed.yaml, encoding="utf-8")
            try:
                candidate = load(proposed.yaml)
                try:
                    metrics = self.evaluator.evaluate(target=candidate, eval_cases=self.eval_cases)
                finally:
                    candidate.close()
                evaluation = CandidateEvaluation(
                    measurement_context=context,
                    run_id=run_id,
                    candidate_id=proposed.candidate_id,
                    configuration=proposed,
                    metrics=metrics,
                    cost=cost_of_model_usage(model_usage=metrics.model_usage, prices=self.prices),
                )
            except Exception as error:
                evaluation = CandidateEvaluation(
                    measurement_context=context,
                    run_id=run_id,
                    candidate_id=proposed.candidate_id,
                    configuration=proposed,
                    metrics=None,
                    failure="".join(traceback.format_exception_only(type(error), error)).strip(),
                )
            self.journal.append(evaluation)
            outcomes.append(evaluation)
            gate_failures = self._find_failed_gates(candidate=evaluation, baseline=ref_eval_metrics)
            history.append(
                CandidateOutcome(
                    candidate_id=proposed.candidate_id,
                    parent_id=proposed.parent_id,
                    rationale=proposed.rationale,
                    diff=proposed.diff,
                    metrics=evaluation.metrics,
                    cost=evaluation.cost,
                    failure=evaluation.failure,
                    gate_failures=gate_failures,
                )
            )
            # The next turn edits the best candidate so far
            eligible = [
                outcome
                for outcome in outcomes
                if not self._find_failed_gates(candidate=outcome, baseline=ref_eval_metrics)
            ]
            best_id = min(eligible, key=self._candidate_sort_key).candidate_id if eligible else None

            logger.info(
                "candidate {position}/{total}: {candidate_id}, gates={gates}",
                position=len(outcomes),
                total=self.max_iterations,
                candidate_id=proposed.candidate_id,
                gates=gate_failures,
            )
            if self.on_candidate is not None:
                self.on_candidate(
                    CandidateProgress(
                        position=len(outcomes),
                        total=self.max_iterations,
                        evaluation=evaluation,
                        gate_failures=gate_failures,
                        baseline=ref_eval_metrics,
                        baseline_cost=ref_cost,
                        is_best=best_id == evaluation.candidate_id,
                    )
                )

        # Recommend the best candidate that clears every gate and actually beats the reference.
        gates = {
            outcome.candidate_id: self._find_failed_gates(candidate=outcome, baseline=ref_eval_metrics)
            for outcome in outcomes
        }
        eligible = sorted(
            (outcome for outcome in outcomes if not gates[outcome.candidate_id]), key=self._candidate_sort_key
        )
        recommendation = None
        for evaluated in eligible:
            reasons = self._recommendation_reasons(
                candidate=evaluated, baseline=ref_eval_metrics, baseline_cost=ref_cost
            )
            if reasons is not None and evaluated.configuration is not None:
                recommendation = ExperimentRecommendation(
                    configuration=evaluated.configuration,
                    evaluation=evaluated,
                    reasons=reasons,
                )
                (artifacts / "recommended.yaml").write_text(evaluated.configuration.yaml, encoding="utf-8")
                break

        # The optimizer is not a candidate, so its spend is priced and reported but never ranked or gated.
        optimizer_cost = cost_of_model_usage(model_usage=optimizer_usage, prices=self.prices)
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

        return ExperimentResult(
            baseline=ref_eval_metrics,
            baseline_cost=ref_cost,
            candidates=tuple(outcomes),
            recommendation=recommendation,
            measurement_context=context,
            run_id=run_id,
            artifact_directory=artifacts,
            gate_failures=gates,
            optimizer_usage=optimizer_usage,
            optimizer_cost=optimizer_cost,
        )

    def _find_failed_gates(self, candidate: CandidateEvaluation, baseline: EvalMetrics) -> tuple[str, ...]:
        """Return the hard gates a candidate failed, or `("evaluation_failed",)` when it produced no measurement."""
        if candidate.metrics is None:
            return ("evaluation_failed",)
        return self.objectives.find_failed_gates(metrics=candidate.metrics, baseline=baseline)

    def _candidate_sort_key(self, candidate: CandidateEvaluation) -> tuple[float, float]:
        """Return a sortable rank that places failed candidates last."""
        if candidate.metrics is None:
            return (float("inf"), float("inf"))
        return self.objectives.sort_key(metrics=candidate.metrics, cost=candidate.cost)

    def _recommendation_reasons(
        self, candidate: CandidateEvaluation, baseline: EvalMetrics, baseline_cost: float | None
    ) -> tuple[str, ...] | None:
        """Explain an improvement or return `None` when the candidate does not beat the baseline."""
        if candidate.metrics is None:
            return None
        candidate_key = self.objectives.sort_key(metrics=candidate.metrics, cost=candidate.cost)
        if candidate_key >= self.objectives.sort_key(metrics=baseline, cost=baseline_cost):
            return None
        return (f"{self.objectives.primary}_improvement",)
