# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Experiment evaluator for Advanced RAG harnesses.

The evaluator depends only on the shared Agent Pack harness contracts. Optimization packages can consume its raw
measurements without the Advanced RAG package depending on optimizer implementation details.
"""

import time
from collections.abc import Mapping
from typing import Any

from haystack import Document
from haystack.components.agents import Agent
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    AdvancedRAGCaseMetrics,
    AdvancedRAGEvaluationCase,
    score_advanced_rag_result,
)
from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics, ModelTokenUsage
from haystack_integrations.agent_pack.run_digest import RUN_DIGEST_KEY, RunDigestPolicy

_MODEL_KEYS = ("model", "azure_deployment", "model_name")
_NESTED_MODEL_CONTAINERS = ("api_params",)
_NESTED_MODEL_KEYS = ("model", "repo_id")


def _model_id_from_parameters(parameters: Mapping[str, Any]) -> str | None:
    """Return a model identifier from recognized generator parameter locations."""
    for key in _MODEL_KEYS:
        if isinstance(value := parameters.get(key), str):
            return value
    for container in _NESTED_MODEL_CONTAINERS:
        nested = parameters.get(container)
        if isinstance(nested, Mapping):
            for key in _NESTED_MODEL_KEYS:
                if isinstance(value := nested.get(key), str):
                    return value
    return None


def _generator_model_id(generator: Any) -> str | None:
    """Return a live generator's model identifier from attributes or its serialized parameters."""
    direct = {key: getattr(generator, key, None) for key in _MODEL_KEYS}
    direct.update({container: getattr(generator, container, None) for container in _NESTED_MODEL_CONTAINERS})
    if model_id := _model_id_from_parameters(parameters=direct):
        return model_id
    try:
        serialized = component_to_dict(obj=generator, name="chat_generator")
    except Exception:
        return None
    parameters = serialized.get("init_parameters") or serialized.get("data") or {}
    return _model_id_from_parameters(parameters=parameters) if isinstance(parameters, Mapping) else None


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

    Quality is the fraction of cases that pass, and is therefore normalized to `[0.0, 1.0]`. A case passes only when
    its retrieval, answer, citation, metadata-inspection, and tool-budget expectations pass. Each case is measured
    once, so `quality_lower_bound` is left unset and an experiment gate compares the fraction itself; breadth across
    cases, rather than repeated measurement of a few, is what makes that fraction discriminating.
    """

    def __init__(
        self,
        *,
        cases: list[AdvancedRAGEvaluationCase] | None = None,
        digest_policy: RunDigestPolicy | None = None,
        max_traced_cases: int | None = 6,
    ) -> None:
        """
        Create an evaluator.

        :param cases: Labelled expectations, keyed internally by question. A run whose question has no labelled
            case falls back to a grounding-parity case derived from the run itself, and the evaluation is reported as
            unvalidated.
        :param digest_policy: Caps applied to the tool trace recorded for each case.
        :param max_traced_cases: How many case traces to keep, or `None` to keep every one. A trace explains a
            result but a reader's history of them is cumulative, so failing cases keep theirs first: a passing case
            has nothing to diagnose.
        """
        self.cases = {case.question: case for case in (cases or [])}
        self.digest_policy = digest_policy
        self.max_traced_cases = max_traced_cases

    def _traced_cases(self, metrics: list[AdvancedRAGCaseMetrics]) -> list[dict[str, Any]]:
        """
        Report every case, keeping tool traces for the ones worth diagnosing.

        :param metrics: Every scored case of every repetition.
        :returns: JSON-compatible case records, with the trace dropped from cases beyond the cap.
        """
        cases = [metric.to_dict() for metric in metrics]
        if self.max_traced_cases is None:
            return cases
        ranked = sorted(range(len(metrics)), key=lambda index: (metrics[index].passed, index))
        for index in ranked[self.max_traced_cases :]:
            cases[index].pop(RUN_DIGEST_KEY, None)
        return cases

    def fingerprint(self) -> dict[str, Any]:
        """
        Describe the evaluation set so an experiment journal is invalidated when it changes.

        :returns: Every configured case, ordered by question.
        """
        return {
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
        :returns: Fraction of cases passed, raw model usage, and total latency for the candidate, with per-case detail.
        :raises ValueError: If no reference runs were supplied.
        """
        resolved, derived = self._resolve(reference_runs=reference_runs)
        if not resolved:
            msg = "No reference runs were supplied to the Advanced RAG evaluator."
            raise ValueError(msg)

        agent.warm_up()
        flattened: list[AdvancedRAGCaseMetrics] = []
        additional_usage: dict[str, ModelTokenUsage] = {}
        for case, messages in resolved:
            started = time.perf_counter()
            result = agent.run(messages=messages)
            latency_ms = (time.perf_counter() - started) * 1000
            flattened.append(
                score_advanced_rag_result(
                    result=result, case=case, latency_ms=latency_ms, digest_policy=self.digest_policy
                )
            )
            for model, usage in (result.get("additional_model_usage") or {}).items():
                current = additional_usage.get(model, ModelTokenUsage())
                additional_usage[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + int(usage.get("input_tokens", 0)),
                    output_tokens=current.output_tokens + int(usage.get("output_tokens", 0)),
                )

        quality = sum(metric.passed for metric in flattened) / len(flattened)
        input_tokens = sum(metric.input_tokens for metric in flattened)
        output_tokens = sum(metric.output_tokens for metric in flattened)
        model_id = _generator_model_id(generator=agent.chat_generator)
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
            latency_ms=sum(metric.latency_ms for metric in flattened),
            model_usage=model_usage,
            details={
                "model": model_id,
                "validated": not derived,
                "derived_cases": derived,
                "cases": self._traced_cases(metrics=flattened),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
