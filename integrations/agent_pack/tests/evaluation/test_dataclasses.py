import json

import pytest
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.evaluation import (
    EvalMetrics,
    ModelTokenUsage,
    RAGEvalCase,
    RetrievalEvalCase,
    ToolRunStats,
)


def call_and_result(name, arguments=None, result="payload", error=False):
    """One assistant turn calling a tool, followed by the tool's answer."""
    call = ToolCall(name, arguments or {}, id=name)
    return [ChatMessage.from_assistant(tool_calls=[call]), ChatMessage.from_tool(result, origin=call, error=error)]


class TestRetrievalEvalCase:
    def test_init_without_evidence(self):
        with pytest.raises(ValueError, match="needs evidence to score against"):
            RetrievalEvalCase(question="q", evidence={})

    def test_expected_document_ids(self):
        eval_case = RetrievalEvalCase(question="q", evidence={"a": "a quote", "b": "another quote"})
        assert eval_case.expected_document_ids == frozenset({"a", "b"})

    @pytest.mark.parametrize(
        ("ranked", "k", "found", "recall", "precision"),
        [
            pytest.param(["a", "x", "y", "b"], 2, {"a"}, 0.5, 0.5, id="counts_only_the_first_k"),
            pytest.param(["a", "x", "y", "b"], None, {"a", "b"}, 1.0, 0.5, id="no_cutoff_scores_everything"),
            pytest.param(["a", "a", "b"], 2, {"a", "b"}, 1.0, 1.0, id="duplicates_do_not_fill_the_cutoff"),
            pytest.param([], None, set(), 0.0, 0.0, id="an_empty_run_scores_zero"),
        ],
    )
    def test_scoring(self, ranked: list[str], k: int | None, found: set[str], recall: float, precision: float):
        """A pipeline is measured on what it put at the top, so a needed document ranked below k earns nothing."""
        eval_case = RetrievalEvalCase(question="q", evidence={"a": "first quote", "b": "second quote"})
        assert eval_case.found_at(document_ids=ranked, k=k) == frozenset(found)
        assert eval_case.recall_at(document_ids=ranked, k=k) == recall
        assert eval_case.precision_at(document_ids=ranked, k=k) == precision

    def test_serialization_roundtrip(self):
        eval_case = RetrievalEvalCase(question="q", evidence={"b": "second", "a": "first"}, min_precision=0.5)
        assert RetrievalEvalCase.from_dict(data=eval_case.to_dict()) == eval_case
        # Evidence is ordered, so the same set is identified the same way whichever order it was built in.
        assert list(eval_case.to_dict()["evidence"]) == ["a", "b"]


class TestRAGEvalCase:
    def test_init_without_evidence(self):
        with pytest.raises(ValueError, match="needs evidence to score against"):
            RAGEvalCase(question="q")

    def test_serialization_roundtrip(self):
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

    def test_serialized_budgets_are_ordered(self):
        """A fingerprint is only stable when the same budgets serialize the same way."""
        one = RAGEvalCase(question="q", evidence={"a": ""}, tool_budgets={"b": 1, "a": 2})
        other = RAGEvalCase(question="q", evidence={"a": ""}, tool_budgets={"a": 2, "b": 1})
        assert one.to_dict() == other.to_dict()


class TestToolRunStats:
    def test_from_messages(self):
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

    def test_calls_to_a_group(self):
        """Harnesses budget retrieval as a whole rather than per tool, so a group counts together."""
        stats = ToolRunStats(calls=[("search_documents", {}), ("fetch_documents_by_filter", {}), ("finish", {})])
        assert stats.calls_to(tools="search_documents") == 1
        assert stats.calls_to(tools=("search_documents", "fetch_documents_by_filter")) == 2
        assert stats.calls_to(tools=["nothing_called"]) == 0

    def test_calls_with_argument(self):
        stats = ToolRunStats(
            calls=[
                ("search_documents", {"query": "x", "filters": {"field": "meta.year"}}),
                ("search_documents", {"query": "x", "filters": None}),
                ("search_documents", {"query": "x"}),
            ]
        )
        assert stats.calls_to(tools="search_documents") == 3
        # Only the call that passed something for the argument counts.
        assert stats.calls_with_argument(tools="search_documents", argument="filters") == 1

    def test_called_before(self):
        metadata, retrieval = "list_metadata_fields", ("search_documents", "fetch_documents_by_filter")
        inspected = ToolRunStats(calls=[(metadata, {}), ("search_documents", {})])
        retrieved = ToolRunStats(calls=[("search_documents", {}), (metadata, {})])
        assert inspected.called_before(tools=metadata, other=retrieval) is True
        assert retrieved.called_before(tools=metadata, other=retrieval) is False
        # Neither ran, so nothing came first and the expectation is not met.
        assert ToolRunStats().called_before(tools=metadata, other=retrieval) is False


class TestEvalMetrics:
    @pytest.mark.parametrize("quality", [-0.01, 1.01])
    def test_init_invalid_quality(self, quality: float):
        """Every harness evaluator must use the shared normalized quality scale."""
        with pytest.raises(ValueError, match="quality must be between"):
            EvalMetrics(quality=quality, latency_ms=1.0)

    def test_serialization_roundtrip(self):
        metrics = EvalMetrics(
            quality=0.75,
            latency_ms=12.5,
            model_usage={"model": ModelTokenUsage(input_tokens=100, output_tokens=20)},
            eval_cases=[{"question": "q", "passed": True}],
            details={"mean_recall": 0.5},
        )
        assert EvalMetrics.from_dict(data=metrics.to_dict()) == metrics
