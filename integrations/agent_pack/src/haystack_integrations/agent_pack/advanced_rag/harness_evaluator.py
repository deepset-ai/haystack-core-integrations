# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Experiment evaluator for Advanced RAG harnesses.

This module bridges the Advanced RAG agent to the generic optimization API, so unlike the rest of `advanced_rag` it
does depend on `optimization`. It is deliberately not re-exported from `haystack_integrations.agent_pack.advanced_rag`:
importing the agent must not drag the optimization package in with it. Import it by module path instead.
"""

import statistics
import time
from typing import Any

from haystack import Document, logging
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    AdvancedRAGCaseMetrics,
    AdvancedRAGEvaluationCase,
    score_advanced_rag_result,
)
from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.assets.model_identity import generator_model_id
from haystack_integrations.agent_pack.optimization.experiment.dataclasses import EvaluationMetrics
from haystack_integrations.agent_pack.tracing.dataclasses import TraceArtifact
from haystack_integrations.agent_pack.tracing.extraction import (
    extract_agent_reference_output,
    extract_agent_replay_inputs,
)

logger = logging.getLogger(__name__)


def messages_from_trace(artifact: TraceArtifact) -> list[ChatMessage]:
    """
    Reconstruct the Agent input messages recorded in a reference trace.

    :param artifact: The captured reference trace.
    :returns: The messages the reference Agent was run with.
    :raises ValueError: If the trace holds no replayable message list.
    """
    inputs = extract_agent_replay_inputs(artifact=artifact)
    serialized = inputs.get("messages")
    if not isinstance(serialized, list):
        msg = f"Trace {artifact.run_id} does not contain replayable Agent messages."
        raise ValueError(msg)
    return [item if isinstance(item, ChatMessage) else ChatMessage.from_dict(item) for item in serialized]


def question_from_messages(messages: list[ChatMessage]) -> str:
    """
    Return the last textual user question in a message list.

    :param messages: The Agent input messages.
    :returns: The text of the last user message.
    :raises ValueError: If no user message carries text.
    """
    for message in reversed(messages):
        if message.is_from("user") and message.text:
            return message.text
    msg = "Reference trace contains no textual user question."
    raise ValueError(msg)


def case_from_reference_trace(artifact: TraceArtifact) -> AdvancedRAGEvaluationCase:
    """
    Create a grounding-parity case from a reference trace.

    The resulting case asserts that a candidate retrieves the same documents the reference retrieved. That measures
    agreement with the incumbent harness, not correctness: a candidate that retrieves *better* evidence scores as a
    regression. Use it to detect drift when no labelled evaluation set exists, and treat any recommendation it
    produces as unvalidated. Supply explicit `AdvancedRAGEvaluationCase` objects for a decision you intend to act on.

    :param artifact: The captured reference trace.
    :returns: A case requiring the candidate to retrieve every document the reference retrieved.
    :raises ValueError: If the trace records no retrieved documents.
    """
    messages = messages_from_trace(artifact)
    output = extract_agent_reference_output(artifact=artifact)
    serialized_documents = output.get("documents") or []
    documents = [item if isinstance(item, Document) else Document.from_dict(item) for item in serialized_documents]
    if not documents:
        msg = f"Trace {artifact.run_id} contains no reference documents; supply an explicit evaluation case."
        raise ValueError(msg)
    return AdvancedRAGEvaluationCase(
        question=question_from_messages(messages),
        expected_document_ids=frozenset(document.id for document in documents),
        min_recall=1.0,
        min_precision=0.0,
    )


class AdvancedRAGHarnessEvaluator:
    """
    Replay trace-selected questions and score Advanced RAG candidates.

    """

    def __init__(self, *, cases: list[AdvancedRAGEvaluationCase] | None = None, repetitions: int = 1) -> None:
        """
        Create an evaluator.

        :param cases: Labelled expectations, keyed internally by question. A trace whose question has no labelled
            case falls back to a grounding-parity case derived from the trace itself, and the run is reported as
            unvalidated.
        :param repetitions: How many times each case is run. Agent runs are not deterministic, so a single sample
            makes a pass rate an unreliable basis for switching models. With more than one repetition, `quality` is
            the mean pass rate and `quality_lower_bound` is one standard deviation below it, which is what experiment
            gates compare against.
        :raises ValueError: If `repetitions` is below one.
        """
        if repetitions < 1:
            msg = "repetitions must be at least 1."
            raise ValueError(msg)
        self.cases = {case.question: case for case in (cases or [])}
        self.repetitions = repetitions

    def fingerprint(self) -> dict[str, Any]:
        """
        Describe the evaluation set so an experiment journal is invalidated when it changes.

        :returns: The repetition count and every configured case, ordered by question.
        """
        return {
            "repetitions": self.repetitions,
            "cases": sorted(
                (case.to_dict() for case in self.cases.values()),
                key=lambda entry: str(entry["question"]),
            ),
        }

    def _resolve(
        self, reference_traces: list[TraceArtifact]
    ) -> tuple[list[tuple[AdvancedRAGEvaluationCase, list[ChatMessage]]], list[str]]:
        """Pair each reference trace with the case that scores it, reporting which cases had to be derived."""
        resolved: list[tuple[AdvancedRAGEvaluationCase, list[ChatMessage]]] = []
        derived: list[str] = []
        for artifact in reference_traces:
            messages = messages_from_trace(artifact)
            question = question_from_messages(messages)
            case = self.cases.get(question)
            if case is None:
                case = case_from_reference_trace(artifact)
                derived.append(question)
            resolved.append((case, messages))
        return resolved, derived

    def evaluate(
        self, agent: Agent, reference_traces: list[TraceArtifact], assets: ApprovedAssetCatalog
    ) -> EvaluationMetrics:
        """
        Replay every selected trace and return experiment metrics priced from the approved asset catalog.

        :param agent: The materialized candidate to score.
        :param reference_traces: The reference traces supplying the questions to replay.
        :param assets: The approved asset catalog, used to price the candidate's token usage.
        :returns: Quality, cost, and latency for the candidate, with per-case detail.
        :raises ValueError: If no reference traces were supplied.
        """
        resolved, derived = self._resolve(reference_traces)
        if not resolved:
            msg = "No reference traces were supplied to the Advanced RAG evaluator."
            raise ValueError(msg)

        run_metrics: list[list[AdvancedRAGCaseMetrics]] = []
        for _ in range(self.repetitions):
            attempt: list[AdvancedRAGCaseMetrics] = []
            for case, messages in resolved:
                started = time.perf_counter()
                result = agent.run(messages=messages)
                latency_ms = (time.perf_counter() - started) * 1000
                attempt.append(score_advanced_rag_result(result, case, latency_ms=latency_ms))
            run_metrics.append(attempt)

        pass_rates = [sum(metric.passed for metric in attempt) / len(attempt) for attempt in run_metrics]
        quality = statistics.fmean(pass_rates)
        lower_bound = None
        if len(pass_rates) > 1:
            lower_bound = max(0.0, quality - statistics.stdev(pass_rates))

        flattened = [metric for attempt in run_metrics for metric in attempt]
        input_tokens = sum(metric.input_tokens for metric in flattened)
        output_tokens = sum(metric.output_tokens for metric in flattened)
        model_id = generator_model_id(agent.chat_generator)
        cost = self._cost(assets, model_id, input_tokens, output_tokens)

        return EvaluationMetrics(
            quality=quality,
            cost=cost,
            latency_ms=sum(metric.latency_ms for metric in flattened) / self.repetitions,
            quality_lower_bound=lower_bound,
            details={
                "model": model_id,
                "repetitions": self.repetitions,
                "pass_rates": pass_rates,
                "quality_stdev": statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0.0,
                "validated": not derived,
                "derived_cases": derived,
                "cases": [metric.to_dict() for metric in flattened],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    @staticmethod
    def _cost(assets: ApprovedAssetCatalog, model_id: str | None, input_tokens: int, output_tokens: int) -> float:
        """
        Price a run from the approved asset catalog, which is the experiment's single source of model prices.

        :param assets: The approved asset catalog.
        :param model_id: The candidate's model identifier.
        :param input_tokens: Total input tokens across every replayed case.
        :param output_tokens: Total output tokens across every replayed case.
        :returns: The priced cost, or zero when the catalog does not price this model.
        """
        asset = assets.models.get(model_id) if isinstance(model_id, str) else None
        if asset is None:
            logger.warning(
                "Candidate model {model} is not priced in the approved asset catalog; reporting zero cost.",
                model=model_id,
            )
            return 0.0
        return (input_tokens * asset.input_cost_per_million + output_tokens * asset.output_cost_per_million) / 1_000_000
