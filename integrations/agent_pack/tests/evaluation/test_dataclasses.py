import json

import pytest
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.evaluation.agent_pack import (
    EvaluationMetrics,
    ModelTokenUsage,
    RAGEvalCase,
    RetrievalEvalCase,
    ToolRunStats,
)


def test_the_documents_an_answer_needs_are_the_ones_its_evidence_is_in():
    case = RetrievalEvalCase(question="q", evidence={"a": "a quote", "b": "another quote"})

    assert case.expected_document_ids == frozenset({"a", "b"})


def test_only_the_documents_above_the_cutoff_count():
    """A pipeline is measured on what it put at the top, so a needed document ranked below k earns nothing."""
    eval_case = RetrievalEvalCase(question="q", evidence={"a": "first quote", "b": "second quote"})
    ranked = ["a", "filler", "filler2", "b"]

    assert eval_case.found_at(document_ids=ranked, k=2) == frozenset({"a"})
    assert eval_case.recall_at(document_ids=ranked, k=2) == 0.5
    assert eval_case.recall_at(document_ids=ranked) == 1.0
    # Precision is over what was scored, not over everything returned.
    assert eval_case.precision_at(document_ids=ranked, k=2) == 0.5
    assert eval_case.precision_at(document_ids=ranked) == 0.5


def test_a_document_returned_twice_does_not_occupy_the_cutoff_twice():
    eval_case = RetrievalEvalCase(question="q", evidence={"b": "a quote"})
    ranked = ["a", "a", "b"]

    assert eval_case.recall_at(document_ids=ranked, k=2) == 1.0
    assert eval_case.precision_at(document_ids=ranked, k=2) == 0.5


def test_scoring_a_run_that_returned_nothing_is_zero_rather_than_an_error():
    eval_case = RetrievalEvalCase(question="q", evidence={"a": "a quote"})

    assert eval_case.recall_at(document_ids=[]) == 0.0
    assert eval_case.precision_at(document_ids=[]) == 0.0


def test_an_eval_case_without_evidence_has_nothing_to_score_against():
    with pytest.raises(ValueError, match="needs evidence to score against"):
        RetrievalEvalCase(question="q", evidence={})


def test_a_rag_eval_case_round_trips_through_a_journal():
    """Budget groups are tuples, which JSON has no key for, so they survive as pairs instead."""
    original = RAGEvalCase(
        question="q",
        evidence={"b": "second quote", "a": "first quote"},
        tool_budgets={("search_documents", "fetch_documents_by_filter"): 12, "list_metadata_fields": 8, "*": 2},
        max_steps=20,
        min_precision=0.5,
    )

    data = original.to_dict()

    assert json.loads(json.dumps(data)) == data
    assert RAGEvalCase.from_dict(data=data) == original


def test_serialized_budgets_are_ordered_so_a_fingerprint_is_stable():
    one = RAGEvalCase(question="q", evidence={"a": ""}, tool_budgets={"b": 1, "a": 2})
    other = RAGEvalCase(question="q", evidence={"a": ""}, tool_budgets={"a": 2, "b": 1})

    assert one.to_dict() == other.to_dict()


def call_and_result(name, arguments=None, result="payload", error=False):
    """One assistant turn calling a tool, followed by the tool's answer."""
    call = ToolCall(name, arguments or {}, id=name)
    return [ChatMessage.from_assistant(tool_calls=[call]), ChatMessage.from_tool(result, origin=call, error=error)]


def test_calls_and_errors_are_extracted_in_the_order_they_happened():
    messages = [
        *call_and_result("list_metadata_fields"),
        *call_and_result("search_documents", {"query": "CRISPR"}),
        *call_and_result("get_metadata_field_values", result="field 'nope' does not exist", error=True),
    ]

    stats = ToolRunStats.from_messages(messages=messages)

    assert [name for name, _ in stats.calls] == [
        "list_metadata_fields",
        "search_documents",
        "get_metadata_field_values",
    ]
    assert stats.calls[1] == ("search_documents", {"query": "CRISPR"})
    # The failing tool names itself, so a report can say which one refused and why.
    assert stats.errors == [("get_metadata_field_values", "field 'nope' does not exist")]


def test_a_group_of_tools_is_counted_as_one_allowance():
    """Harnesses budget retrieval as a whole rather than per tool, so a group counts together."""
    stats = ToolRunStats(calls=[("search_documents", {}), ("fetch_documents_by_filter", {}), ("finish", {})])

    assert stats.calls_to(tools="search_documents") == 1
    assert stats.calls_to(tools=("search_documents", "fetch_documents_by_filter")) == 2
    assert stats.calls_to(tools=["nothing_called"]) == 0


def test_only_calls_that_passed_something_for_the_argument_are_counted():
    stats = ToolRunStats(
        calls=[
            ("search_documents", {"query": "x", "filters": {"field": "meta.year"}}),
            ("search_documents", {"query": "x", "filters": None}),
            ("search_documents", {"query": "x"}),
        ]
    )

    assert stats.calls_to(tools="search_documents") == 3
    assert stats.calls_with_argument(tools="search_documents", argument="filters") == 1


def test_ordering_holds_only_when_the_first_tool_actually_ran_first():
    metadata, retrieval = "list_metadata_fields", ("search_documents", "fetch_documents_by_filter")
    inspected = ToolRunStats(calls=[(metadata, {}), ("search_documents", {})])
    retrieved = ToolRunStats(calls=[("search_documents", {}), (metadata, {})])

    assert inspected.called_before(tools=metadata, other=retrieval) is True
    assert retrieved.called_before(tools=metadata, other=retrieval) is False
    # Neither ran, so nothing came first and the expectation is not met.
    assert ToolRunStats().called_before(tools=metadata, other=retrieval) is False


def test_evaluation_metrics_roundtrip() -> None:
    """Shared harness measurements retain raw usage and evaluator-specific details when serialized."""
    metrics = EvaluationMetrics(
        quality=0.75,
        latency_ms=12.5,
        model_usage={"model": ModelTokenUsage(input_tokens=100, output_tokens=20)},
        details={"mean_recall": 0.5},
    )

    assert EvaluationMetrics.from_dict(data=metrics.to_dict()) == metrics


@pytest.mark.parametrize("quality", [-0.01, 1.01])
def test_evaluation_metrics_reject_quality_outside_normalized_range(quality: float) -> None:
    """Every harness evaluator must use the shared normalized quality scale."""
    with pytest.raises(ValueError, match="quality must be between"):
        EvaluationMetrics(quality=quality, latency_ms=1.0)
