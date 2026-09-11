import pytest
from haystack import Document
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.advanced_rag.evaluation import score_advanced_rag_result
from haystack_integrations.evaluation import RAGEvalCase

# What the labelled document was needed for; the quote itself is not scored, only which document holds it.
EVIDENCE = "CRISPR is used for gene editing"


def result_for(answer, documents, *, calls=("list_metadata_fields", "search_documents"), errors=(), steps=3):
    messages = []
    for index, name in enumerate(calls):
        call = ToolCall(name, {"query": "CRISPR"} if name == "search_documents" else {}, id=f"call-{index}")
        messages.append(ChatMessage.from_assistant(tool_calls=[call]))
        messages.append(ChatMessage.from_tool("payload", origin=call, error=name in errors))
    messages.append(ChatMessage.from_assistant(answer))
    return {
        "messages": messages,
        "last_message": ChatMessage.from_assistant(answer),
        "documents": list(documents),
        "step_count": steps,
        "token_usage": {"input_tokens": 100, "output_tokens": 20},
    }


@pytest.fixture
def document():
    return Document(content="CRISPR is used for gene editing")


class TestScoreAdvancedRagResult:
    def test_scores_a_passing_run(self, document):
        eval_case = RAGEvalCase(
            question="What is CRISPR used for?",
            evidence={document.id: EVIDENCE},
        )
        metrics = score_advanced_rag_result(
            result=result_for(f"CRISPR evidence [doc {document.id[:8]}]", [document]),
            eval_case=eval_case,
            latency_ms=12,
        )
        assert metrics.passed is True
        assert metrics.failures == ()
        assert metrics.recall == 1.0
        assert metrics.precision == 1.0
        assert metrics.citations_resolved is True
        assert metrics.cited_document_ids == (document.id[:8],)
        assert metrics.inspected_first is True
        assert metrics.retrieval_calls == 1
        assert metrics.metadata_calls == 1
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 20

    def test_answer_without_citations(self, document):
        """An `all()` over zero citations is trivially true, so an uncited answer must be caught explicitly."""
        eval_case = RAGEvalCase(question="What is CRISPR used for?", evidence={document.id: EVIDENCE})
        metrics = score_advanced_rag_result(
            result=result_for("CRISPR edits genes.", [document]), eval_case=eval_case, latency_ms=1
        )
        assert metrics.passed is False
        assert metrics.failures == ("answer_cites_nothing",)
        assert metrics.citations_resolved is True

    def test_citations_to_unretrieved_documents(self, document):
        eval_case = RAGEvalCase(question="q", evidence={document.id: EVIDENCE}, require_citations=False)
        metrics = score_advanced_rag_result(
            result=result_for("Invented [doc deadbeef]", [document]), eval_case=eval_case, latency_ms=1
        )
        assert metrics.failures == ("unresolvable_citation",)
        assert metrics.citations_resolved is False

    def test_error_and_step_budgets(self, document):
        """Every budget the run broke is named, so one report says everything that has to change."""
        eval_case = RAGEvalCase(
            question="q",
            evidence={document.id: EVIDENCE},
            require_citations=False,
            max_tool_errors=0,
            max_steps=1,
        )
        metrics = score_advanced_rag_result(
            result=result_for("CRISPR is used for gene editing", [document], errors=("search_documents",)),
            eval_case=eval_case,
            latency_ms=1,
        )
        assert metrics.passed is False
        assert set(metrics.failures) == {"tool_errors:1", "steps_over_budget:3"}

    def test_reports_tool_order(self, document):
        eval_case = RAGEvalCase(question="q", evidence={document.id: EVIDENCE}, require_citations=False)
        metrics = score_advanced_rag_result(
            result=result_for(
                f"answer [doc {document.id[:8]}]", [document], calls=("search_documents", "list_metadata_fields")
            ),
            eval_case=eval_case,
            latency_ms=1,
        )
        # Reported so a reader can see the order, but not a failure: what matters is the answer, not the route.
        assert metrics.inspected_first is False
        assert metrics.failures == ()

    def test_counts_filtered_retrievals(self, document):
        """The filtered count is what shows whether metadata inspection changed how the Agent retrieved."""
        eval_case = RAGEvalCase(question="q", evidence={document.id: EVIDENCE})
        result = result_for("answer", [document])
        result["messages"].append(
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall("search_documents", {"query": "x", "filters": {"field": "meta.year"}}, id="filtered")
                ]
            )
        )
        metrics = score_advanced_rag_result(result=result, eval_case=eval_case, latency_ms=1)
        assert metrics.retrieval_calls == 2
        assert metrics.filtered_retrieval_calls == 1
