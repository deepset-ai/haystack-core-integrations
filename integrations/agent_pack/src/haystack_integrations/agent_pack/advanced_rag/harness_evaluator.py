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

from haystack import Document
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    AdvancedRAGCaseMetrics,
    AdvancedRAGEvaluationCase,
    score_advanced_rag_result,
)
from haystack_integrations.agent_pack.optimization.models import (
    EvaluationMetrics,
    ModelTokenUsage,
    generator_model_id,
)
from haystack_integrations.agent_pack.runs import AgentRunRecord


def messages_from_run(record: AgentRunRecord) -> list[ChatMessage]:
    """
    Reconstruct the Agent input messages recorded in a reference run.

    :param record: The recorded reference run.
    :returns: The messages the reference Agent was run with.
    :raises ValueError: If the run holds no replayable message list.
    """
    serialized = record.inputs.get("messages")
    if not isinstance(serialized, list):
        msg = f"Run {record.run_id} does not contain replayable Agent messages."
        raise ValueError(msg)
    return [item if isinstance(item, ChatMessage) else ChatMessage.from_dict(data=item) for item in serialized]


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
    msg = "Reference run contains no textual user question."
    raise ValueError(msg)


def case_from_reference_run(record: AgentRunRecord) -> AdvancedRAGEvaluationCase:
    """
    Create a grounding-parity case from a reference run.

    The resulting case asserts that a candidate retrieves the same documents the reference retrieved. That measures
    agreement with the incumbent harness, not correctness: a candidate that retrieves *better* evidence scores as a
    regression. Use it to detect drift when no labelled evaluation set exists, and treat any recommendation it
    produces as unvalidated. Supply explicit `AdvancedRAGEvaluationCase` objects for a decision you intend to act on.

    :param record: The recorded reference run.
    :returns: A case requiring the candidate to retrieve every document the reference retrieved.
    :raises ValueError: If the run records no retrieved documents.
    """
    messages = messages_from_run(record=record)
    serialized_documents = record.outputs.get("documents") or []
    documents = [item if isinstance(item, Document) else Document.from_dict(data=item) for item in serialized_documents]
    if not documents:
        msg = f"Run {record.run_id} contains no reference documents; supply an explicit evaluation case."
        raise ValueError(msg)
    return AdvancedRAGEvaluationCase(
        question=question_from_messages(messages=messages),
        expected_document_ids=frozenset(document.id for document in documents),
        min_recall=1.0,
        min_precision=0.0,
    )


class AdvancedRAGHarnessEvaluator:
    """
    Replay recorded questions and score Advanced RAG candidates.

    """

    def __init__(self, *, cases: list[AdvancedRAGEvaluationCase] | None = None, repetitions: int = 1) -> None:
        """
        Create an evaluator.

        :param cases: Labelled expectations, keyed internally by question. A run whose question has no labelled
            case falls back to a grounding-parity case derived from the run itself, and the evaluation is reported as
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
        self, reference_runs: list[AgentRunRecord]
    ) -> tuple[list[tuple[AdvancedRAGEvaluationCase, list[ChatMessage]]], list[str]]:
        """Pair each reference run with the case that scores it, reporting which cases had to be derived."""
        resolved: list[tuple[AdvancedRAGEvaluationCase, list[ChatMessage]]] = []
        derived: list[str] = []
        for record in reference_runs:
            messages = messages_from_run(record=record)
            question = question_from_messages(messages=messages)
            case = self.cases.get(question)
            if case is None:
                case = case_from_reference_run(record=record)
                derived.append(question)
            resolved.append((case, messages))
        return resolved, derived

    def evaluate(self, agent: Agent, reference_runs: list[AgentRunRecord]) -> EvaluationMetrics:
        """
        Replay every selected run and return raw experiment metrics.

        :param agent: The materialized candidate to score.
        :param reference_runs: The successful runs supplying the questions to replay.
        :returns: Quality, cost, and latency for the candidate, with per-case detail.
        :raises ValueError: If no reference runs were supplied.
        """
        resolved, derived = self._resolve(reference_runs=reference_runs)
        if not resolved:
            msg = "No reference runs were supplied to the Advanced RAG evaluator."
            raise ValueError(msg)

        agent.warm_up()
        run_metrics: list[list[AdvancedRAGCaseMetrics]] = []
        additional_usage: dict[str, ModelTokenUsage] = {}
        for _ in range(self.repetitions):
            attempt: list[AdvancedRAGCaseMetrics] = []
            for case, messages in resolved:
                started = time.perf_counter()
                result = agent.run(messages=messages)
                latency_ms = (time.perf_counter() - started) * 1000
                attempt.append(score_advanced_rag_result(result=result, case=case, latency_ms=latency_ms))
                for model, usage in (result.get("additional_model_usage") or {}).items():
                    current = additional_usage.get(model, ModelTokenUsage())
                    additional_usage[model] = ModelTokenUsage(
                        input_tokens=current.input_tokens + int(usage.get("input_tokens", 0)),
                        output_tokens=current.output_tokens + int(usage.get("output_tokens", 0)),
                    )
            run_metrics.append(attempt)

        pass_rates = [sum(metric.passed for metric in attempt) / len(attempt) for attempt in run_metrics]
        quality = statistics.fmean(pass_rates)
        lower_bound = None
        if len(pass_rates) > 1:
            lower_bound = max(0.0, quality - statistics.stdev(pass_rates))

        flattened = [metric for attempt in run_metrics for metric in attempt]
        input_tokens = sum(metric.input_tokens for metric in flattened)
        output_tokens = sum(metric.output_tokens for metric in flattened)
        model_id = generator_model_id(generator=agent.chat_generator)
        if model_id is None:
            msg = "The evaluator cannot attribute token usage because the Agent's model identifier is unknown."
            raise ValueError(msg)

        model_usage = dict(additional_usage)
        coordinator = model_usage.get(model_id, ModelTokenUsage())
        model_usage[model_id] = ModelTokenUsage(
            input_tokens=coordinator.input_tokens + input_tokens,
            output_tokens=coordinator.output_tokens + output_tokens,
        )

        return EvaluationMetrics(
            quality=quality,
            latency_ms=sum(metric.latency_ms for metric in flattened) / self.repetitions,
            model_usage=model_usage,
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
