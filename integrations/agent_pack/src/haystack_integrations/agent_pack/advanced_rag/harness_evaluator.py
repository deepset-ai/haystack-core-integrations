# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Experiment evaluator for Advanced RAG harnesses.

The evaluator depends only on the shared Agent Pack harness contracts. Optimization packages can consume its raw
measurements without the Advanced RAG package depending on optimizer implementation details.
"""

import asyncio
import time
from typing import Any

from haystack import Document, logging
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    METADATA_TOOLS,
    RETRIEVAL_TOOLS,
    AdvancedRAGCaseMetrics,
    AdvancedRAGEvaluationCase,
    score_advanced_rag_result,
)
from haystack_integrations.agent_pack.advanced_rag.tools import (
    GetMetadataFieldRangeTool,
    GetMetadataFieldValuesTool,
    ListMetadataFieldsTool,
)
from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics, ModelTokenUsage
from haystack_integrations.agent_pack.run_digest import RUN_DIGEST_KEY, RunDigestPolicy
from haystack_integrations.agent_pack.usage_tracer import CaseUsage, UsageTracer

logger = logging.getLogger(__name__)


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
    once, so breadth across cases, rather than repeated measurement of a few, is what makes that fraction
    discriminating.
    """

    def __init__(
        self,
        *,
        cases: list[AdvancedRAGEvaluationCase] | None = None,
        digest_policy: RunDigestPolicy | None = None,
        max_traced_cases: int | None = 6,
        max_concurrent_cases: int = 1,
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
        :param max_concurrent_cases: How many cases to measure at once. Cases are independent and each one spends
            its time waiting on a model, so measuring several together is most of what makes an experiment
            affordable in wall-clock terms. Token usage, and therefore cost, is unaffected. Measured latency is
            not: concurrent runs contend for the same rate limits, so leave this at 1 when ranking by latency, or
            the objective measures this setting rather than the configuration. Cases are driven through
            `Agent.run_async` whatever this is set to, so one at a time is simply a concurrency of one.
        :raises ValueError: If `max_concurrent_cases` is below one.
        """
        if max_concurrent_cases < 1:
            msg = "max_concurrent_cases must be at least 1."
            raise ValueError(msg)
        self.cases = {case.question: case for case in (cases or [])}
        self.digest_policy = digest_policy
        self.max_concurrent_cases = max_concurrent_cases
        self.max_traced_cases = max_traced_cases

    def validate_agent(self, agent: Agent) -> None:
        """
        Require document state used for retrieval and citation scoring.

        :param agent: Candidate Agent deserialized from YAML.
        """
        if "documents" not in agent.resolved_state_schema:
            msg = "The RAG evaluator requires a documents state output."
            raise ValueError(msg)

    def _traced_cases(self, metrics: list[AdvancedRAGCaseMetrics]) -> list[dict[str, Any]]:
        """
        Report every case, keeping tool traces for the ones worth diagnosing.

        :param metrics: Every scored case.
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

    def _score(
        self,
        result: dict[str, Any],
        case: AdvancedRAGEvaluationCase,
        started: float,
        position: int,
        total: int,
        retrieval_tools: frozenset[str] = RETRIEVAL_TOOLS,
        metadata_tools: frozenset[str] = METADATA_TOOLS,
    ) -> AdvancedRAGCaseMetrics:
        """
        Score one completed Agent run and report it.

        :param result: What `Agent.run` returned.
        :param case: The expectations to score against.
        :param started: The `perf_counter` reading from before the run.
        :param position: Which case this is, for reporting.
        :param total: How many cases there are, for reporting.
        :param retrieval_tools: Names resolved from candidate document outputs.
        :param metadata_tools: Names resolved from metadata tool classes.
        :returns: The case score.
        """
        latency_ms = (time.perf_counter() - started) * 1000
        scored = score_advanced_rag_result(
            result=result,
            case=case,
            latency_ms=latency_ms,
            digest_policy=self.digest_policy,
            retrieval_tools=retrieval_tools,
            metadata_tools=metadata_tools,
        )
        logger.info(
            "case {position}/{total} {verdict} in {latency:.0f}ms: {question}",
            position=position,
            total=total,
            verdict="passed" if scored.passed else f"FAILED ({', '.join(scored.failures)})",
            latency=latency_ms,
            question=case.question[:80],
        )
        return scored

    async def _measure(
        self, agent: Agent, resolved: list[tuple[AdvancedRAGEvaluationCase, list[ChatMessage]]], tracer: UsageTracer
    ) -> list[tuple[AdvancedRAGCaseMetrics, CaseUsage]]:
        """
        Measure every case, running up to `max_concurrent_cases` of them at once.

        :param agent: The candidate to measure.
        :param resolved: Each case with the messages that pose it.
        :param tracer: Collector for per-case generator usage.
        :returns: One result per case, in case order.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_cases)
        tools = flatten_tools_or_toolsets(getattr(agent, "tools", []))
        retrieval_tools = RETRIEVAL_TOOLS | frozenset(
            tool.name for tool in tools if "documents" in (tool.outputs_to_state or {})
        )
        metadata_tools = METADATA_TOOLS | frozenset(
            tool.name
            for tool in tools
            if isinstance(tool, (ListMetadataFieldsTool, GetMetadataFieldValuesTool, GetMetadataFieldRangeTool))
        )

        async def measure(
            position: int, case: AdvancedRAGEvaluationCase, messages: list[ChatMessage]
        ) -> tuple[AdvancedRAGCaseMetrics, CaseUsage]:
            """Run one case, waiting for a slot first."""
            async with semaphore:
                started = time.perf_counter()
                with tracer.case() as usage:
                    result = await agent.run_async(messages=messages)
            scored = self._score(
                result=result,
                case=case,
                started=started,
                position=position,
                total=len(resolved),
                retrieval_tools=retrieval_tools,
                metadata_tools=metadata_tools,
            )
            return scored, usage

        return list(
            await asyncio.gather(
                *(measure(position, case, messages) for position, (case, messages) in enumerate(resolved, start=1))
            )
        )

    def evaluate(self, agent: Agent, reference_runs: list[AgentRunRecord]) -> EvaluationMetrics:
        """
        Replay every selected run and return raw experiment metrics.

        :param agent: The materialized candidate to score.
        :param reference_runs: The successful runs supplying the questions to replay.
        :returns: Fraction of cases passed, raw model usage, and mean latency for the candidate, with per-case detail.
        :raises ValueError: If no reference runs were supplied.
        """
        resolved, derived = self._resolve(reference_runs=reference_runs)
        if not resolved:
            msg = "No reference runs were supplied to the Advanced RAG evaluator."
            raise ValueError(msg)

        tracer = UsageTracer()
        agent.warm_up()
        with tracer.activate():
            measured = asyncio.run(self._measure(agent=agent, resolved=resolved, tracer=tracer))

        flattened = [scored for scored, _ in measured]
        model_usage: dict[str, ModelTokenUsage] = {}
        for _, usage in measured:
            for model, tokens in usage.models.items():
                current = model_usage.get(model, ModelTokenUsage())
                model_usage[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + tokens.input_tokens,
                    output_tokens=current.output_tokens + tokens.output_tokens,
                )
        quality = sum(metric.passed for metric in flattened) / len(flattened)
        input_tokens = sum(usage.input_tokens for usage in model_usage.values())
        output_tokens = sum(usage.output_tokens for usage in model_usage.values())
        model_id = getattr(agent.chat_generator, "model", None)
        return EvaluationMetrics(
            quality=quality,
            latency_ms=sum(metric.latency_ms for metric in flattened) / len(flattened),
            model_usage=model_usage,
            details={
                "model": model_id,
                "usage_complete": all(usage.complete and usage.calls > 0 for _, usage in measured),
                "mean_recall": sum(metric.recall for metric in flattened) / len(flattened),
                "mean_precision": sum(metric.precision for metric in flattened) / len(flattened),
                "answer_pass_rate": sum(metric.answer_requirements_met for metric in flattened) / len(flattened),
                "validated": not derived,
                "derived_cases": derived,
                "cases": self._traced_cases(metrics=flattened),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
