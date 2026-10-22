# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import traceback
from collections.abc import Callable
from dataclasses import asdict
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
    CandidateEvaluation,
    CandidateOutcome,
    CandidateProgress,
    ExperimentRecommendation,
    ExperimentResult,
    OptimizationObjectives,
    ProposalResult,
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


class ExperimentJournal:
    """Append-only JSON-lines record of raw experiment measurements, one file per experiment."""

    def __init__(self, directory: str | Path) -> None:
        """
        Open a directory of experiment journals.

        Each run gets its own run ID and journal file, even when its eval context ID matches an earlier run.

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
        # Keys keep the `to_dict` order, so each line starts with the eval context ID and the candidate
        with (
            self._lock,
            self.path_for(run_id=evaluation.run_id).open("a", encoding="utf-8") as stream,
        ):
            stream.write(json.dumps(evaluation.to_dict()) + "\n")


def _sorted_eval_case_dicts(eval_cases: list[Any]) -> list[dict[str, Any]]:
    """
    Serialize the eval cases and sort them by question, so the same set gives the same list in any order.

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
        *,
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
        baseline_callback: Callable[[EvalMetrics, float | None], None] | None = None,
        candidate_callback: Callable[[CandidateProgress], None] | None = None,
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
            documentation. Requires `mcp-haystack>=1.2.0`.
        :param configuration_key: Optional caller-supplied identifier for external measurement inputs, such as a
            corpus or harness version, that cannot be inferred from the serialized Agent and evaluator.
        :param max_iterations: Maximum number of candidate outcomes included in the experiment.
        :param history_digest_window: How many of the most recent outcomes are shown to the optimizer in full.
        :param baseline_callback: Called with the reference measurement and its cost before the search starts.
        :param candidate_callback: Called with each candidate's `CandidateProgress` right after it is measured, priced
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
        self.baseline_callback = baseline_callback
        self.candidate_callback = candidate_callback

    def run(self) -> ExperimentResult:
        """
        Measure the reference, then propose, measure and gate up to `max_iterations` candidates.

        :returns: The baseline, every candidate measured and its gate failures, the optimizer's own usage and cost,
            and a recommendation when a candidate cleared every gate and outranked the reference.
        """
        dump, load = _serialization(reference=self.reference)
        reference_yaml = dump(self.reference)
        eval_context_id, payload = self._eval_context()
        run_id, artifacts = self.journal.claim_run()
        (artifacts / "reference.yaml").write_text(reference_yaml, encoding="utf-8")

        # Measure the reference, which every candidate is ranked and gated against
        ref_eval_metrics, ref_cost = self._evaluate_config(yaml=reference_yaml, load=load)
        self.journal.append(
            CandidateEvaluation(
                eval_context_id=eval_context_id,
                run_id=run_id,
                candidate_id=_configuration_id(text=reference_yaml),
                configuration=None,
                metrics=ref_eval_metrics,
                cost=ref_cost,
            )
        )
        if self.objectives.primary == "cost" and (ref_cost is None or not ref_eval_metrics.all_tokens_reported):
            msg = "The reference has unavailable cost; supply prices for its models and complete usage."
            raise ValueError(msg)
        # Fail before the search when the evaluator does not report the metric the objectives treat as quality
        self.objectives.get_quality(metrics=ref_eval_metrics)
        if self.baseline_callback is not None:
            self.baseline_callback(ref_eval_metrics, ref_cost)

        # Search: propose one candidate, measure it, and feed the outcome into the next proposal
        outcomes: list[CandidateEvaluation] = []
        history: list[CandidateOutcome] = []
        gates: dict[str, tuple[str, ...]] = {}
        best_id: str | None = None
        optimizer_usage: dict[str, ModelTokenUsage] = {}
        optimizer_tracer = HarnessTracer()
        while len(outcomes) < self.max_iterations:
            # Each turn starts from the best candidate so far, or from the last one submitted when none has cleared
            # the gates yet
            turn, turn_usage = self._propose(
                tracer=optimizer_tracer,
                reference_yaml=reference_yaml,
                baseline=ref_eval_metrics,
                history=history,
                outcomes=outcomes,
                base_id=best_id or (outcomes[-1].candidate_id if outcomes else None),
            )
            for model, tokens in turn_usage.items():
                optimizer_usage[model] = optimizer_usage.get(model, ModelTokenUsage()) + tokens
            # Journal drafts that failed validation, so the run shows what the optimizer had to repair
            for failure in turn.validation_failures:
                self.journal.append(
                    CandidateEvaluation(
                        eval_context_id=eval_context_id,
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
                metrics, cost = self._evaluate_config(yaml=proposed.yaml, load=load)
                evaluation = CandidateEvaluation(
                    eval_context_id=eval_context_id,
                    run_id=run_id,
                    candidate_id=proposed.candidate_id,
                    configuration=proposed,
                    metrics=metrics,
                    cost=cost,
                )
            except Exception as error:
                evaluation = CandidateEvaluation(
                    eval_context_id=eval_context_id,
                    run_id=run_id,
                    candidate_id=proposed.candidate_id,
                    configuration=proposed,
                    metrics=None,
                    failure="".join(traceback.format_exception_only(type(error), error)).strip(),
                )
            self.journal.append(evaluation)
            outcomes.append(evaluation)
            gate_failures = self._find_failed_gates(candidate=evaluation, baseline=ref_eval_metrics)
            gates[proposed.candidate_id] = gate_failures
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
            eligible = [outcome for outcome in outcomes if not gates[outcome.candidate_id]]
            best_id = min(eligible, key=self._candidate_sort_key).candidate_id if eligible else None

            logger.info(
                "candidate {position}/{total}: {candidate_id}, gates={gates}",
                position=len(outcomes),
                total=self.max_iterations,
                candidate_id=proposed.candidate_id,
                gates=gate_failures,
            )
            if self.candidate_callback is not None:
                self.candidate_callback(
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

        recommendation = self._recommend(
            outcomes=outcomes, gates=gates, baseline=ref_eval_metrics, baseline_cost=ref_cost, artifacts=artifacts
        )
        # The optimizer is not a candidate, so its spend is priced and reported but never ranked or gated
        optimizer_cost = cost_of_model_usage(model_usage=optimizer_usage, prices=self.prices)
        logger.info(
            "optimizer spend across {turns} turns: {usage} ({cost})",
            turns=len(outcomes),
            usage={model: asdict(obj=tokens) for model, tokens in optimizer_usage.items()},
            cost="unpriced" if optimizer_cost is None else f"${optimizer_cost:.6f}",
        )
        self._write_context(
            artifacts=artifacts,
            eval_context_id=eval_context_id,
            run_id=run_id,
            payload=payload,
            optimizer_usage=optimizer_usage,
            optimizer_cost=optimizer_cost,
        )

        return ExperimentResult(
            baseline=ref_eval_metrics,
            baseline_cost=ref_cost,
            candidates=tuple(outcomes),
            recommendation=recommendation,
            eval_context_id=eval_context_id,
            run_id=run_id,
            artifact_directory=artifacts,
            gate_failures=gates,
            optimizer_usage=optimizer_usage,
            optimizer_cost=optimizer_cost,
        )

    def _eval_context(self) -> tuple[str, dict[str, Any]]:
        """
        Fingerprint what makes two runs' measurements comparable: the evaluator, eval cases and configuration key.

        :returns: The fingerprint, and the values it was taken from.
        """
        payload = {
            "evaluator": self.evaluator.to_dict(),
            "eval_cases": _sorted_eval_case_dicts(eval_cases=self.eval_cases),
            "configuration_key": self.configuration_key,
        }
        return content_digest(payload=json.dumps(payload, sort_keys=True, default=str)), payload

    def _evaluate_config(self, yaml: str, load: Callable[[str], Agent | Pipeline]) -> tuple[EvalMetrics, float | None]:
        """
        Build a configuration from its YAML, run the evaluator on it and price its token usage.

        :param yaml: The configuration to evaluate.
        :param load: Builds the configuration from YAML.
        :returns: The evaluator's measurement, and its cost, or `None` when a model it used has no known price.
        """
        target = load(yaml)
        try:
            metrics = self.evaluator.evaluate(target=target, eval_cases=self.eval_cases)
        finally:
            target.close()
        return metrics, cost_of_model_usage(model_usage=metrics.model_usage, prices=self.prices)

    def _propose(
        self,
        *,
        tracer: HarnessTracer,
        reference_yaml: str,
        baseline: EvalMetrics,
        history: list[CandidateOutcome],
        outcomes: list[CandidateEvaluation],
        base_id: str | None,
    ) -> tuple[ProposalResult, dict[str, ModelTokenUsage]]:
        """
        Run one optimizer turn, tracing the optimizer's own model calls so the search's spend can be reported.

        :param tracer: The tracer collecting the optimizer's spans.
        :param reference_yaml: The reference configuration.
        :param baseline: The reference measurement.
        :param history: Every candidate outcome so far, as the optimizer is shown it.
        :param outcomes: Every candidate measured so far, whose configurations the optimizer can restore.
        :param base_id: The candidate the turn starts from, or `None` for the reference.
        :returns: The turn's result, and the tokens the optimizer spent on it by model.
        """
        with tracer.activate(), tracing.tracer.trace(EVAL_CASE_SPAN) as turn_span:
            turn = propose_candidate(
                optimizer_agent=self.optimizer_agent,
                reference=self.reference,
                reference_yaml=reference_yaml,
                prices=self.prices,
                objectives=self.objectives,
                baseline=baseline,
                history=history,
                history_digest_window=self.history_digest_window,
                remaining_evaluations=self.max_iterations - len(outcomes),
                candidates={
                    outcome.candidate_id: outcome.configuration.yaml
                    for outcome in outcomes
                    if outcome.configuration is not None
                },
                base_id=base_id,
            )
        # No usage is reported when a HarnessTracer was not the active tracer
        collected = turn_span.collected if isinstance(turn_span, HarnessSpan) else None
        return turn, (collected.summarize().model_usage if collected is not None else {})

    def _recommend(
        self,
        *,
        outcomes: list[CandidateEvaluation],
        gates: dict[str, tuple[str, ...]],
        baseline: EvalMetrics,
        baseline_cost: float | None,
        artifacts: Path,
    ) -> ExperimentRecommendation | None:
        """
        Pick the best candidate that cleared every gate and outranks the reference, and save it as `recommended.yaml`.

        :param outcomes: Every candidate measured.
        :param gates: The gates each candidate failed, by candidate ID.
        :param baseline: The reference measurement.
        :param baseline_cost: The reference's cost, or `None` when it could not be priced.
        :param artifacts: The run's artifact directory.
        :returns: The recommendation, or `None` when no candidate qualifies.
        """
        eligible = sorted(
            (outcome for outcome in outcomes if not gates[outcome.candidate_id]), key=self._candidate_sort_key
        )
        for evaluated in eligible:
            reasons = self._recommendation_reasons(candidate=evaluated, baseline=baseline, baseline_cost=baseline_cost)
            if reasons is not None and evaluated.configuration is not None:
                (artifacts / "recommended.yaml").write_text(evaluated.configuration.yaml, encoding="utf-8")
                return ExperimentRecommendation(
                    configuration=evaluated.configuration, evaluation=evaluated, reasons=reasons
                )
        return None

    def _write_context(
        self,
        *,
        artifacts: Path,
        eval_context_id: str,
        run_id: str,
        payload: dict[str, Any],
        optimizer_usage: dict[str, ModelTokenUsage],
        optimizer_cost: float | None,
    ) -> None:
        """
        Write `context.json`, naming what the run's measurements compare against and what the search itself spent.

        :param artifacts: The run's artifact directory.
        :param eval_context_id: The eval context ID.
        :param run_id: The run's ID.
        :param payload: The values the fingerprint was taken from.
        :param optimizer_usage: The optimizer's token usage by model.
        :param optimizer_cost: What that usage costs, or `None` when a model has no known price.
        """
        (artifacts / "context.json").write_text(
            json.dumps(
                {
                    "eval_context_id": eval_context_id,
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
