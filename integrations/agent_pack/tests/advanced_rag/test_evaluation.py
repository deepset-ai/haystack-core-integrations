from datetime import UTC, datetime
from types import SimpleNamespace

from haystack import Document
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    AdvancedRAGEvaluationCase,
    AdvancedRAGHarnessEvaluator,
    case_from_reference_trace,
    score_advanced_rag_result,
)
from haystack_integrations.agent_pack.optimization import TraceArtifact


def successful_result(document):
    metadata_call = ToolCall("list_metadata_fields", {}, id="metadata")
    retrieval_call = ToolCall("search_documents", {"query": "CRISPR"}, id="retrieval")
    return {
        "messages": [
            ChatMessage.from_assistant(tool_calls=[metadata_call]),
            ChatMessage.from_tool("fields", origin=metadata_call),
            ChatMessage.from_assistant(tool_calls=[retrieval_call]),
            ChatMessage.from_tool("documents", origin=retrieval_call),
            ChatMessage.from_assistant(f"CRISPR evidence [doc {document.id[:8]}]"),
        ],
        "last_message": ChatMessage.from_assistant(f"CRISPR evidence [doc {document.id[:8]}]"),
        "documents": [document],
        "step_count": 3,
        "token_usage": {"input_tokens": 100, "output_tokens": 20},
    }


def reference_trace(document):
    now = datetime.now(tz=UTC).isoformat()
    return TraceArtifact(
        run_id="rag-reference",
        started_at=now,
        finished_at=now,
        duration_ms=10,
        status="success",
        traces=(
            {
                "span_id": "root",
                "parent_span_id": None,
                "operation_name": "haystack.agent.run",
                "component": None,
                "start_time": now,
                "end_time": now,
                "duration_ms": 10,
                "tags": {
                    "haystack.agent.input": {"messages": [ChatMessage.from_user("What is CRISPR used for?").to_dict()]},
                    "haystack.agent.output": {
                        "last_message": ChatMessage.from_assistant("reference").to_dict(),
                        "documents": [document.to_dict()],
                    },
                },
            },
        ),
    )


def test_scores_grounding_citations_and_process_budgets():
    document = Document(content="CRISPR is used for gene editing")
    case = AdvancedRAGEvaluationCase(
        question="What is CRISPR used for?",
        expected_document_ids=frozenset({document.id}),
        answer_must_mention=("CRISPR",),
    )

    metrics = score_advanced_rag_result(successful_result(document), case, latency_ms=12)

    assert metrics.passed is True
    assert metrics.recall == 1.0
    assert metrics.citations_resolved is True
    assert metrics.inspected_first is True
    assert metrics.retrieval_calls == 1


def test_derives_grounding_parity_case_from_reference_trace():
    document = Document(content="CRISPR is used for gene editing")
    case = case_from_reference_trace(reference_trace(document))
    assert case.question == "What is CRISPR used for?"
    assert case.expected_document_ids == frozenset({document.id})


def test_harness_evaluator_replays_trace_and_calculates_model_cost():
    document = Document(content="CRISPR is used for gene editing")

    class FakeAgent:
        chat_generator = SimpleNamespace(model="cheap")

        def run(self, **kwargs):
            assert kwargs["messages"][0].text == "What is CRISPR used for?"
            return successful_result(document)

    evaluator = AdvancedRAGHarnessEvaluator(model_prices={"cheap": (2.0, 4.0)})
    metrics = evaluator.evaluate(FakeAgent(), [reference_trace(document)])

    assert metrics.quality == 1.0
    assert metrics.cost == (100 * 2.0 + 20 * 4.0) / 1_000_000
    assert metrics.details["input_tokens"] == 100
    assert metrics.details["cases"][0]["passed"] is True
