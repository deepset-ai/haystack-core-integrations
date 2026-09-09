# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Imported by `advanced_rag` itself, so this must stay usable without pulling in the optimization subpackage.
# The experiment-facing evaluator that bridges the two lives in `advanced_rag.harness_evaluator`, which is not
# re-exported for that reason.
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from haystack import Document
from haystack.components.agents.utils import _INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _first_numeric
from haystack.dataclasses import ChatMessage
from haystack.utils.filters import document_matches_filter

from haystack_integrations.agent_pack.run_digest import RunDigestPolicy, digest_agent_run

RETRIEVAL_TOOLS = frozenset({"search_documents", "fetch_documents_by_filter"})
METADATA_TOOLS = frozenset({"list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range"})

# Citation format produced by the Advanced RAG toolset: the first eight characters of a document ID.
CITATION_PATTERN = re.compile(r"\[doc ([0-9a-fA-F]{8})\]")

# A tool's own explanation of a refusal is the most useful part of a failing run, and the useful part comes first.
_MAX_TRACE_ERROR_CHARS = 200


@dataclass
class RunStats:
    """Tool-level process statistics extracted from an Agent conversation."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    retrieval_tools: frozenset[str] = RETRIEVAL_TOOLS
    metadata_tools: frozenset[str] = METADATA_TOOLS

    @property
    def errors(self) -> int:
        """
        Return how many tool results reported an error.

        :returns: The number of failing tool calls.
        """
        return len(self.error_messages)

    @property
    def inspected_first(self) -> bool:
        """
        Return whether metadata inspection preceded the first retrieval.

        :returns: True if a metadata tool was called before any retrieval tool.
        """
        for name, _ in self.calls:
            if name in self.metadata_tools:
                return True
            if name in self.retrieval_tools:
                return False
        return False

    @property
    def metadata_calls(self) -> int:
        """
        Return the number of metadata-inspection calls.

        :returns: How many calls targeted a metadata tool.
        """
        return sum(1 for name, _ in self.calls if name in self.metadata_tools)

    @property
    def filtered_retrieval_calls(self) -> int:
        """
        Return the number of retrieval calls carrying a metadata filter.

        :returns: How many retrieval calls passed a non-empty `filters` argument.
        """
        return sum(1 for name, arguments in self.calls if name in self.retrieval_tools and arguments.get("filters"))

    @property
    def retrieval_calls(self) -> int:
        """
        Return the number of retrieval calls.

        :returns: How many calls targeted a retrieval tool.
        """
        return sum(1 for name, _ in self.calls if name in self.retrieval_tools)


def extract_run_stats(
    messages: list[ChatMessage],
    retrieval_tools: frozenset[str] = RETRIEVAL_TOOLS,
    metadata_tools: frozenset[str] = METADATA_TOOLS,
) -> RunStats:
    """
    Extract tool calls and error results from one Agent run.

    :param messages: The messages an Agent run produced.
    :param retrieval_tools: Resolved retrieval tool names.
    :param metadata_tools: Resolved metadata tool names.
    :returns: The tool calls made, in order, and the message of every tool result that reported an error.
    """
    stats = RunStats(retrieval_tools=retrieval_tools, metadata_tools=metadata_tools)
    for message in messages:
        stats.calls.extend((call.tool_name, call.arguments or {}) for call in message.tool_calls)
        stats.error_messages.extend(
            str(result.result)[:_MAX_TRACE_ERROR_CHARS] for result in message.tool_call_results if result.error
        )
    return stats


@dataclass(frozen=True, kw_only=True)
class AdvancedRAGEvaluationCase:
    """
    Grounding and process expectations for one enterprise RAG question.

    :param question: The user question to ask.
    :param expected_document_ids: Documents the answer must be grounded in.
    :param expected_metadata_filter: A Haystack metadata filter describing relevant documents when the complete
        expected ID set is too large to enumerate.
    :param min_matching_documents: Minimum number of retrieved documents that must match `expected_metadata_filter`.
    :param answer_must_mention: Case-insensitive substrings the answer must contain.
    :param answer_must_not_mention: Case-insensitive substrings the answer must not contain, for checking that a
        known wrong or out-of-scope claim is absent.
    :param min_recall: Minimum share of `expected_document_ids` that must be retrieved.
    :param min_precision: Minimum share of retrieved documents that must be expected. Left at 0 by default because
        an agent legitimately retrieves context beyond the labelled evidence; raise it to penalise over-retrieval.
    :param require_metadata_inspection: Whether metadata must be inspected before the first retrieval.
    :param require_citations: Whether an answer grounded in retrieved documents must cite at least one of them. An
        answer with no citations at all otherwise passes a citation check trivially.
    :param max_metadata_calls: Budget for metadata-inspection calls.
    :param max_retrieval_calls: Budget for retrieval calls.
    :param max_tool_errors: Tolerated failing tool calls.
    :param max_steps: Optional cap on Agent steps.
    """

    question: str
    expected_document_ids: frozenset[str] = frozenset()
    expected_metadata_filter: dict[str, Any] | None = None
    min_matching_documents: int = 1
    answer_must_mention: tuple[str, ...] = ()
    answer_must_not_mention: tuple[str, ...] = ()
    min_recall: float = 1.0
    min_precision: float = 0.0
    require_metadata_inspection: bool = True
    require_citations: bool = True
    max_metadata_calls: int = 5
    max_retrieval_calls: int = 5
    max_tool_errors: int = 0
    max_steps: int | None = None

    def __post_init__(self) -> None:
        """Require exactly one applicable form of retrieval ground truth."""
        if not (self.expected_document_ids or self.expected_metadata_filter):
            msg = f"Case {self.question!r} needs expected document IDs or a metadata filter."
            raise ValueError(msg)
        if self.expected_document_ids and self.expected_metadata_filter:
            msg = f"Case {self.question!r} cannot combine expected document IDs with a metadata filter."
            raise ValueError(msg)
        if self.min_matching_documents < 1:
            msg = "min_matching_documents must be at least 1."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the AdvancedRAGEvaluationCase into a dictionary.

        The document ID set and the term tuples become sorted lists and lists respectively, so the result is JSON
        compatible and stable enough to identify an evaluation set.

        :returns: A dictionary with one key per field.
        """
        data = asdict(self)
        data["expected_document_ids"] = sorted(self.expected_document_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdvancedRAGEvaluationCase":
        """
        Create a new AdvancedRAGEvaluationCase object from a dictionary.

        :param data: The dictionary to build the case from.
        :returns: The created object.
        """
        arguments = dict(data)
        arguments["expected_document_ids"] = frozenset(arguments.get("expected_document_ids") or ())
        for key in ("answer_must_mention", "answer_must_not_mention"):
            if key in arguments:
                arguments[key] = tuple(arguments[key])
        return cls(**arguments)


@dataclass(frozen=True, kw_only=True)
class AdvancedRAGCaseMetrics:
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
    answer_requirements_met: bool
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
        Convert the AdvancedRAGCaseMetrics into a dictionary.

        :returns: A dictionary with one key per field, with the failure and citation tuples as lists.
        """
        data = asdict(self)
        data["failures"] = list(self.failures)
        data["cited_document_ids"] = list(self.cited_document_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdvancedRAGCaseMetrics":
        """
        Create a new AdvancedRAGCaseMetrics object from a dictionary.

        :param data: The dictionary to build the metrics from.
        :returns: The created object.
        """
        arguments = dict(data)
        arguments["failures"] = tuple(arguments.get("failures") or ())
        arguments["cited_document_ids"] = tuple(arguments.get("cited_document_ids") or ())
        return cls(**arguments)


def score_advanced_rag_result(
    result: dict[str, Any],
    case: AdvancedRAGEvaluationCase,
    *,
    latency_ms: float,
    digest_policy: RunDigestPolicy | None = None,
    retrieval_tools: frozenset[str] = RETRIEVAL_TOOLS,
    metadata_tools: frozenset[str] = METADATA_TOOLS,
    backup_answer_used: bool = False,
) -> AdvancedRAGCaseMetrics:
    """
    Score retrieval grounding, answer behaviour, and process budgets for one Agent result.

    :param result: The dictionary returned by `Agent.run`.
    :param case: The expectations to score the result against.
    :param latency_ms: Measured wall-clock duration of the run.
    :param digest_policy: Caps applied to the recorded tool trace.
    :param backup_answer_used: Whether the backup LLM wrote the answer, which the caller observes rather than
        the result reporting: the backup runs outside the Agent's step loop and leaves nothing in its output.
    :returns: The score, naming every expectation the run missed, and the trace explaining why.
    """
    messages = result.get("messages") or []
    stats = extract_run_stats(messages=messages, retrieval_tools=retrieval_tools, metadata_tools=metadata_tools)
    last_message = result.get("last_message")
    answer = (getattr(last_message, "text", None) or "") if last_message is not None else ""
    lowered = answer.lower()
    retrieved_documents = result.get("documents") or []
    retrieved_ids = {document.id for document in retrieved_documents}
    matched_ids = retrieved_ids & case.expected_document_ids
    matching_documents: list[Document] = []
    if case.expected_metadata_filter is not None:
        matching_documents = [
            document
            for document in retrieved_documents
            if document_matches_filter(filters=case.expected_metadata_filter, document=document)
        ]
    matched_count = len(matching_documents) if case.expected_metadata_filter is not None else len(matched_ids)
    recall = (
        min(matched_count / case.min_matching_documents, 1.0)
        if case.expected_metadata_filter is not None
        else len(matched_ids) / len(case.expected_document_ids)
        if case.expected_document_ids
        else 0.0
    )
    precision = matched_count / len(retrieved_ids) if retrieved_ids else 0.0

    cited_refs = tuple(CITATION_PATTERN.findall(answer))
    citations_resolved = all(
        any(document.id.startswith(reference) for document in retrieved_documents) for reference in cited_refs
    )

    failures: list[str] = []

    if case.expected_metadata_filter is not None and matched_count < case.min_matching_documents:
        failures.append(f"matching_documents_below_{case.min_matching_documents}")
    elif case.expected_metadata_filter is None and recall < case.min_recall:
        failures.append(f"recall_below_{case.min_recall:g}")
    if precision < case.min_precision:
        failures.append(f"precision_below_{case.min_precision:g}")
    if not citations_resolved:
        failures.append("unresolvable_citation")
    if case.require_citations and retrieved_documents and not cited_refs:
        failures.append("answer_cites_nothing")
    missing_terms = [term for term in case.answer_must_mention if term.lower() not in lowered]
    if missing_terms:
        failures.append(f"answer_missing:{','.join(missing_terms)}")
    forbidden_terms = [term for term in case.answer_must_not_mention if term.lower() in lowered]
    if forbidden_terms:
        failures.append(f"answer_mentions_forbidden:{','.join(forbidden_terms)}")

    if case.require_metadata_inspection and not stats.inspected_first:
        failures.append("metadata_not_inspected_first")
    if stats.metadata_calls > case.max_metadata_calls:
        failures.append(f"metadata_calls_over_budget:{stats.metadata_calls}")
    if stats.retrieval_calls > case.max_retrieval_calls:
        failures.append(f"retrieval_calls_over_budget:{stats.retrieval_calls}")
    if stats.errors > case.max_tool_errors:
        failures.append(f"tool_errors:{stats.errors}")

    steps = int(result.get("step_count") or 0)
    if case.max_steps is not None and steps > case.max_steps:
        failures.append(f"steps_over_budget:{steps}")

    usage = result.get("token_usage") or {}
    return AdvancedRAGCaseMetrics(
        question=case.question,
        passed=not failures,
        failures=tuple(failures),
        recall=recall,
        precision=precision,
        citations_resolved=citations_resolved,
        cited_document_ids=cited_refs,
        answer_requirements_met=not missing_terms and not forbidden_terms,
        inspected_first=stats.inspected_first,
        metadata_calls=stats.metadata_calls,
        retrieval_calls=stats.retrieval_calls,
        filtered_retrieval_calls=stats.filtered_retrieval_calls,
        tool_errors=stats.errors,
        steps=steps,
        latency_ms=latency_ms,
        backup_answer_used=backup_answer_used,
        run_digest=digest_agent_run(result=result, policy=digest_policy),
        input_tokens=_first_numeric(usage, _INPUT_TOKEN_KEYS),
        output_tokens=_first_numeric(usage, _OUTPUT_TOKEN_KEYS),
        token_usage=dict(usage),
    )
