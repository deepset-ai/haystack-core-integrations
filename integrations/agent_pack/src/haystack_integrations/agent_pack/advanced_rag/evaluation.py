# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Imported by `advanced_rag` itself, so this must stay usable without pulling in the optimization subpackage.
# The experiment-facing evaluator that bridges the two lives in `advanced_rag.harness_evaluator`, which is not
# re-exported for that reason.
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from haystack.components.agents.utils import _INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _first_numeric

from haystack_integrations.agent_pack.evaluation.dataclasses import RAGEvalCase, ToolRunStats
from haystack_integrations.agent_pack.evaluation.tool_budgets import budgets_exceeded, resolve_tool_budgets
from haystack_integrations.agent_pack.run_digest import RunDigestPolicy, digest_agent_run

RETRIEVAL_TOOLS = frozenset({"search_documents", "fetch_documents_by_filter"})
METADATA_TOOLS = frozenset({"list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range"})

# Citation format produced by the Advanced RAG toolset: the first eight characters of a document ID.
CITATION_PATTERN = re.compile(r"\[doc ([0-9a-fA-F]{8})\]")


@dataclass(kw_only=True)
class AdvancedRAGEvalCaseMetrics:
    """
    Detailed score for one Advanced RAG evaluation case.

    `failures` names every expectation the run missed, so a regression report says what broke rather than only that
    something did. `passed` is true exactly when `failures` is empty. `run_digest` records what the Agent actually
    did — every tool call with its arguments and result — so a failure can be diagnosed rather than only counted.
    `backup_answer_used` names one chain the counts hide: a run cut off by its step budget is answered by the
    backup-answer hook, which does not cite, so it fails a citation expectation for a reason that has nothing to do
    with retrieval.
    """

    question: str
    passed: bool
    failures: tuple[str, ...]
    recall: float
    precision: float
    citations_resolved: bool
    cited_document_ids: tuple[str, ...]
    inspected_first: bool
    metadata_calls: int
    retrieval_calls: int
    filtered_retrieval_calls: int
    tool_errors: int
    steps: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    token_usage: dict[str, Any]
    backup_answer_used: bool = False
    run_digest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the AdvancedRAGEvalCaseMetrics into a dictionary.

        :returns: A dictionary with one key per field, with the failure and citation tuples as lists.
        """
        data = asdict(self)
        data["failures"] = list(self.failures)
        data["cited_document_ids"] = list(self.cited_document_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdvancedRAGEvalCaseMetrics":
        """
        Create a new AdvancedRAGEvalCaseMetrics object from a dictionary.

        :param data: The dictionary to build the metrics from.
        :returns: The created object.
        """
        arguments = dict(data)
        arguments["failures"] = tuple(arguments.get("failures") or ())
        arguments["cited_document_ids"] = tuple(arguments.get("cited_document_ids") or ())
        return cls(**arguments)


def score_advanced_rag_result(
    result: dict[str, Any],
    eval_case: RAGEvalCase,
    *,
    latency_ms: float,
    digest_policy: RunDigestPolicy | None = None,
    retrieval_tools: frozenset[str] = RETRIEVAL_TOOLS,
    metadata_tools: frozenset[str] = METADATA_TOOLS,
    tool_budgets: dict[tuple[str, ...], int] | None = None,
    backup_answer_used: bool = False,
) -> AdvancedRAGEvalCaseMetrics:
    """
    Score retrieval grounding, answer behaviour, and process budgets for one Agent result.

    :param result: The dictionary returned by `Agent.run`.
    :param eval_case: The expectations to score the result against.
    :param latency_ms: Measured wall-clock duration of the run.
    :param digest_policy: Caps applied to the recorded tool trace.
    :param tool_budgets: Allowances already resolved against the tools the candidate has. Defaults to what the
        eval case names, which is all a caller scoring a single result knows.
    :param backup_answer_used: Whether the backup LLM wrote the answer, which the caller observes rather than
        the result reporting: the backup runs outside the Agent's step loop and leaves nothing in its output.
    :returns: The score, naming every expectation the run missed, and the trace explaining why.
    """
    messages = result.get("messages") or []
    stats = ToolRunStats.from_messages(messages=messages)
    retrieval_names, metadata_names = tuple(retrieval_tools), tuple(metadata_tools)
    # Reported rather than required: inspecting metadata first is good practice, not a correct answer.
    inspected_first = stats.called_before(tools=metadata_names, other=retrieval_names)
    metadata_calls = stats.calls_to(tools=metadata_names)
    retrieval_calls = stats.calls_to(tools=retrieval_names)
    filtered_retrieval_calls = stats.calls_with_argument(tools=retrieval_names, argument="filters")
    last_message = result.get("last_message")
    answer = (getattr(last_message, "text", None) or "") if last_message is not None else ""
    retrieved_documents = result.get("documents") or []
    retrieved_ids = {document.id for document in retrieved_documents}
    matched_ids = retrieved_ids & eval_case.expected_document_ids
    recall = len(matched_ids) / len(eval_case.expected_document_ids) if eval_case.expected_document_ids else 0.0
    precision = len(matched_ids) / len(retrieved_ids) if retrieved_ids else 0.0

    cited_refs = tuple(CITATION_PATTERN.findall(answer))
    citations_resolved = all(
        any(document.id.startswith(reference) for document in retrieved_documents) for reference in cited_refs
    )

    failures: list[str] = []

    if recall < eval_case.min_recall:
        failures.append(f"recall_below_{eval_case.min_recall:g}")
    if precision < eval_case.min_precision:
        failures.append(f"precision_below_{eval_case.min_precision:g}")
    if not citations_resolved:
        failures.append("unresolvable_citation")
    if eval_case.require_citations and retrieved_documents and not cited_refs:
        failures.append("answer_cites_nothing")

    if tool_budgets is None:
        tool_budgets = resolve_tool_budgets(budgets=eval_case.tool_budgets, tool_names=())
    for group, (calls, limit) in budgets_exceeded(stats=stats, budgets=tool_budgets).items():
        failures.append(f"tool_calls_over_budget:{'+'.join(group)}:{calls}/{limit}")
    if len(stats.errors) > eval_case.max_tool_errors:
        failures.append(f"tool_errors:{len(stats.errors)}")

    steps = int(result.get("step_count") or 0)
    if eval_case.max_steps is not None and steps > eval_case.max_steps:
        failures.append(f"steps_over_budget:{steps}")

    usage = result.get("token_usage") or {}
    return AdvancedRAGEvalCaseMetrics(
        question=eval_case.question,
        passed=not failures,
        failures=tuple(failures),
        recall=recall,
        precision=precision,
        citations_resolved=citations_resolved,
        cited_document_ids=cited_refs,
        inspected_first=inspected_first,
        metadata_calls=metadata_calls,
        retrieval_calls=retrieval_calls,
        filtered_retrieval_calls=filtered_retrieval_calls,
        tool_errors=len(stats.errors),
        steps=steps,
        latency_ms=latency_ms,
        backup_answer_used=backup_answer_used,
        run_digest=digest_agent_run(result=result, policy=digest_policy),
        input_tokens=_first_numeric(usage, _INPUT_TOKEN_KEYS),
        output_tokens=_first_numeric(usage, _OUTPUT_TOKEN_KEYS),
        token_usage=dict(usage),
    )
