import json

import pytest
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.evaluation import EvalCase, RAGEvalCase, ToolRunStats


def test_the_documents_an_answer_needs_are_the_ones_its_evidence_is_in():
    case = EvalCase(question="q", evidence={"a": "a quote", "b": "another quote"})

    assert case.expected_document_ids == frozenset({"a", "b"})


def test_an_eval_case_without_evidence_has_nothing_to_score_against():
    with pytest.raises(ValueError, match="needs evidence to score against"):
        EvalCase(question="q", evidence={})


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
