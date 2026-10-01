import pytest
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.evaluation import (
    EvalMetrics,
    ModelPrice,
    ModelTokenUsage,
    RetrievalEvalCase,
    ToolRunStats,
    cost_of_model_usage,
)

PRICES = {"known": ModelPrice(input_cost_per_million=2.0, output_cost_per_million=4.0)}


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
        eval_case = RetrievalEvalCase(question="q", evidence={"b": "second", "a": "first"})
        assert RetrievalEvalCase.from_dict(data=eval_case.to_dict()) == eval_case
        # Evidence is ordered, so the same set is identified the same way whichever order it was built in.
        assert list(eval_case.to_dict()["evidence"]) == ["a", "b"]


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
    def test_serialization_roundtrip(self):
        metrics = EvalMetrics(
            durations=[12.5, 3.0],
            model_usage={"model": ModelTokenUsage(input_tokens=100, output_tokens=20)},
            all_tokens_reported=False,
            eval_cases=[{"question": "q", "passed": True}],
            details={"mean_recall": 0.5},
        )
        assert EvalMetrics.from_dict(data=metrics.to_dict()) == metrics


class TestCostOfModelUsage:
    def test_known_models(self):
        usage = {"known": ModelTokenUsage(input_tokens=100, output_tokens=20)}
        assert cost_of_model_usage(model_usage=usage, prices=PRICES) == (100 * 2.0 + 20 * 4.0) / 1_000_000

    def test_unknown_model(self):
        usage = {"known": ModelTokenUsage(input_tokens=100), "unknown": ModelTokenUsage(input_tokens=100)}
        assert cost_of_model_usage(model_usage=usage, prices=PRICES) is None
