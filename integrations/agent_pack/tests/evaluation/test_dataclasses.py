import json

import pytest

from haystack_integrations.agent_pack.evaluation import EvalCase, RAGEvalCase


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
