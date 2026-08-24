# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable evaluation primitives for Advanced RAG agents and optimization campaigns."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from haystack import Document
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.optimization.campaign import EvaluationMetrics, HarnessEvaluator
from haystack_integrations.agent_pack.optimization.policy import POLICY_DECISIONS_CONTEXT_KEY
from haystack_integrations.agent_pack.optimization.tracing import (
    TraceArtifact,
    extract_agent_reference_output,
    extract_agent_replay_inputs,
)

RETRIEVAL_TOOLS = {"search_documents", "fetch_documents_by_filter"}
METADATA_TOOLS = {"list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range"}
_CITATION_RE = re.compile(r"\[doc ([0-9a-fA-F]{8})\]")


@dataclass
class RunStats:
    """Tool-level process statistics extracted from an Agent conversation."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    errors: int = 0

    @property
    def inspected_first(self) -> bool:
        """Return whether metadata inspection preceded the first retrieval."""
        for name, _ in self.calls:
            if name == "list_metadata_fields":
                return True
            if name in RETRIEVAL_TOOLS:
                return False
        return False

    @property
    def metadata_calls(self) -> int:
        """Return the number of metadata-inspection calls."""
        return sum(1 for name, _ in self.calls if name in METADATA_TOOLS)

    @property
    def filtered_retrieval_calls(self) -> int:
        """Return the number of retrieval calls carrying a metadata filter."""
        return sum(1 for name, arguments in self.calls if name in RETRIEVAL_TOOLS and arguments.get("filters"))

    @property
    def retrieval_calls(self) -> int:
        """Return the number of retrieval calls."""
        return sum(1 for name, _ in self.calls if name in RETRIEVAL_TOOLS)


def extract_run_stats(messages: list[ChatMessage]) -> RunStats:
    """Extract tool calls and error results from one Agent run."""
    stats = RunStats()
    for message in messages:
        stats.calls.extend((call.tool_name, call.arguments or {}) for call in message.tool_calls)
        stats.errors += sum(1 for result in message.tool_call_results if result.error)
    return stats


@dataclass(frozen=True)
class AdvancedRAGEvaluationCase:
    """Grounding and process expectations for one enterprise RAG question."""

    question: str
    expected_document_ids: frozenset[str]
    answer_must_mention: tuple[str, ...] = ()
    min_recall: float = 1.0
    min_precision: float = 0.0
    expect_absent: bool = False
    require_metadata_inspection: bool = True
    max_metadata_calls: int = 5
    max_retrieval_calls: int = 5


@dataclass(frozen=True)
class AdvancedRAGCaseMetrics:
    """Detailed score for one Advanced RAG evaluation case."""

    passed: bool
    recall: float
    precision: float
    citations_resolved: bool
    answer_requirements_met: bool
    inspected_first: bool
    metadata_calls: int
    retrieval_calls: int
    tool_errors: int
    steps: int
    latency_ms: float
    token_usage: dict[str, int]


def score_advanced_rag_result(
    result: dict[str, Any], case: AdvancedRAGEvaluationCase, *, latency_ms: float
) -> AdvancedRAGCaseMetrics:
    """Score retrieval grounding, answer behavior, and process budgets for one Agent result."""
    messages = result.get("messages") or []
    stats = extract_run_stats(messages)
    answer = result["last_message"].text or ""
    retrieved_documents = result.get("documents") or []
    retrieved_ids = {document.id for document in retrieved_documents}
    matched_ids = retrieved_ids & case.expected_document_ids
    recall = len(matched_ids) / len(case.expected_document_ids) if case.expected_document_ids else 0.0
    precision = len(matched_ids) / len(retrieved_ids) if retrieved_ids else 0.0
    cited_refs = _CITATION_RE.findall(answer)
    citations_resolved = all(
        any(document.id.startswith(reference) for document in retrieved_documents) for reference in cited_refs
    )
    answer_requirements_met = all(term.lower() in answer.lower() for term in case.answer_must_mention)
    within_budget = (
        stats.metadata_calls <= case.max_metadata_calls and stats.retrieval_calls <= case.max_retrieval_calls
    )
    inspected_ok = stats.inspected_first if case.require_metadata_inspection else True
    if case.expect_absent:
        grounding_ok = not retrieved_documents and any(
            phrase in answer.lower() for phrase in ("no matching information", "not found", "no information")
        )
    else:
        grounding_ok = recall >= case.min_recall and precision >= case.min_precision
    passed = inspected_ok and within_budget and grounding_ok and answer_requirements_met and citations_resolved
    usage = {key: value for key, value in (result.get("token_usage") or {}).items() if isinstance(value, int)}
    return AdvancedRAGCaseMetrics(
        passed=passed,
        recall=recall,
        precision=precision,
        citations_resolved=citations_resolved,
        answer_requirements_met=answer_requirements_met,
        inspected_first=stats.inspected_first,
        metadata_calls=stats.metadata_calls,
        retrieval_calls=stats.retrieval_calls,
        tool_errors=stats.errors,
        steps=int(result.get("step_count") or 0),
        latency_ms=latency_ms,
        token_usage=usage,
    )


def _messages_from_trace(artifact: TraceArtifact) -> list[ChatMessage]:
    inputs = extract_agent_replay_inputs(artifact)
    serialized = inputs.get("messages")
    if not isinstance(serialized, list):
        msg = f"Trace {artifact.run_id} does not contain replayable Agent messages."
        raise ValueError(msg)
    return [item if isinstance(item, ChatMessage) else ChatMessage.from_dict(item) for item in serialized]


def _question_from_messages(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.is_from("user") and message.text:
            return message.text
    msg = "Reference trace contains no textual user question."
    raise ValueError(msg)


def case_from_reference_trace(artifact: TraceArtifact) -> AdvancedRAGEvaluationCase:
    """Create a grounding-parity case from a successful reference trace."""
    messages = _messages_from_trace(artifact)
    output = extract_agent_reference_output(artifact)
    serialized_documents = output.get("documents") or []
    documents = [item if isinstance(item, Document) else Document.from_dict(item) for item in serialized_documents]
    if not documents:
        msg = f"Trace {artifact.run_id} contains no reference documents; supply an explicit evaluation case."
        raise ValueError(msg)
    return AdvancedRAGEvaluationCase(
        question=_question_from_messages(messages),
        expected_document_ids=frozenset(document.id for document in documents),
        min_recall=1.0,
        min_precision=0.0,
    )


def _token_count(usage: dict[str, int], primary: str, fallback: str) -> int:
    return int(usage.get(primary, usage.get(fallback, 0)))


class AdvancedRAGHarnessEvaluator(HarnessEvaluator):
    """Replay trace-selected questions and score Advanced RAG candidates."""

    def __init__(
        self,
        *,
        cases: list[AdvancedRAGEvaluationCase] | None = None,
        model_prices: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.cases = {case.question: case for case in (cases or [])}
        self.model_prices = model_prices or {}

    def evaluate(self, agent: Agent, reference_traces: list[TraceArtifact]) -> EvaluationMetrics:
        """Evaluate the candidate on explicit cases or grounding parity derived from its reference traces."""
        selected_cases: list[AdvancedRAGEvaluationCase] = []
        messages_by_question: dict[str, list[ChatMessage]] = {}
        for artifact in reference_traces:
            messages = _messages_from_trace(artifact)
            question = _question_from_messages(messages)
            messages_by_question[question] = messages
            selected_cases.append(self.cases.get(question) or case_from_reference_trace(artifact))

        policy_decisions: list[dict[str, Any]] = []
        case_metrics: list[AdvancedRAGCaseMetrics] = []
        total_input_tokens = 0
        total_output_tokens = 0
        for case in selected_cases:
            started = time.perf_counter()
            result = agent.run(
                messages=messages_by_question[case.question],
                hook_context={POLICY_DECISIONS_CONTEXT_KEY: policy_decisions},
            )
            latency_ms = (time.perf_counter() - started) * 1000
            metrics = score_advanced_rag_result(result, case, latency_ms=latency_ms)
            case_metrics.append(metrics)
            total_input_tokens += _token_count(metrics.token_usage, "input_tokens", "prompt_tokens")
            total_output_tokens += _token_count(metrics.token_usage, "output_tokens", "completion_tokens")

        model_id = getattr(agent.chat_generator, "model", "")
        input_price, output_price = self.model_prices.get(model_id, (0.0, 0.0))
        total_cost = (total_input_tokens * input_price + total_output_tokens * output_price) / 1_000_000
        quality = sum(metrics.passed for metrics in case_metrics) / len(case_metrics)
        return EvaluationMetrics(
            quality=quality,
            cost=total_cost,
            latency_ms=sum(metrics.latency_ms for metrics in case_metrics),
            details={
                "cases": [asdict(metrics) for metrics in case_metrics],
                "policy_decisions": policy_decisions,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            },
        )
