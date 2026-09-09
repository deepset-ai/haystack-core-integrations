import pytest
from haystack import Document
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    AdvancedRAGEvaluationCase,
    score_advanced_rag_result,
)


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


def test_scores_grounding_citations_and_process_budgets(document):
    case = AdvancedRAGEvaluationCase(
        question="What is CRISPR used for?",
        expected_document_ids=frozenset({document.id}),
        answer_must_mention=("CRISPR",),
    )

    metrics = score_advanced_rag_result(
        result=result_for(f"CRISPR evidence [doc {document.id[:8]}]", [document]), case=case, latency_ms=12
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


def test_an_answer_with_no_citations_does_not_pass_the_citation_check(document):
    """An `all()` over zero citations is trivially true, so an uncited answer must be caught explicitly."""
    case = AdvancedRAGEvaluationCase(
        question="What is CRISPR used for?", expected_document_ids=frozenset({document.id})
    )

    metrics = score_advanced_rag_result(result=result_for("CRISPR edits genes.", [document]), case=case, latency_ms=1)

    assert metrics.passed is False
    assert metrics.failures == ("answer_cites_nothing",)
    assert metrics.citations_resolved is True


def test_citations_pointing_at_unretrieved_documents_are_reported(document):
    case = AdvancedRAGEvaluationCase(
        question="q", expected_document_ids=frozenset({document.id}), require_citations=False
    )
    metrics = score_advanced_rag_result(
        result=result_for("Invented [doc deadbeef]", [document]), case=case, latency_ms=1
    )
    assert metrics.failures == ("unresolvable_citation",)
    assert metrics.citations_resolved is False


def test_forbidden_terms_and_error_budgets_are_enforced(document):
    case = AdvancedRAGEvaluationCase(
        question="q",
        expected_document_ids=frozenset({document.id}),
        answer_must_mention=("CRISPR",),
        answer_must_not_mention=("guaranteed cure",),
        require_citations=False,
        max_tool_errors=0,
        max_steps=1,
    )

    metrics = score_advanced_rag_result(
        result=result_for("CRISPR is a guaranteed cure", [document], errors=("search_documents",)),
        case=case,
        latency_ms=1,
    )

    assert metrics.passed is False
    assert set(metrics.failures) == {
        "answer_mentions_forbidden:guaranteed cure",
        "tool_errors:1",
        "steps_over_budget:3",
    }


def test_metadata_inspection_order_is_checked(document):
    case = AdvancedRAGEvaluationCase(
        question="q", expected_document_ids=frozenset({document.id}), require_citations=False
    )
    metrics = score_advanced_rag_result(
        result=result_for(
            f"answer [doc {document.id[:8]}]", [document], calls=("search_documents", "list_metadata_fields")
        ),
        case=case,
        latency_ms=1,
    )
    assert metrics.inspected_first is False
    assert metrics.failures == ("metadata_not_inspected_first",)


def test_a_case_needs_expected_documents_unless_absence_is_expected():
    with pytest.raises(ValueError, match="needs expected document IDs or a metadata filter"):
        AdvancedRAGEvaluationCase(question="q")


def test_metadata_filter_ground_truth_scores_large_corpus_constraints():
    """A large-corpus case scores relevant retrieved documents without enumerating every expected ID."""
    matching = [
        Document(content="good", meta={"category": "beauty", "rating": 1}),
        Document(content="also good", meta={"category": "beauty", "rating": 2}),
    ]
    unrelated = Document(content="other", meta={"category": "music", "rating": 1})
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.category", "operator": "==", "value": "beauty"},
            {"field": "meta.rating", "operator": "<=", "value": 2},
        ],
    }
    case = AdvancedRAGEvaluationCase(
        question="q",
        expected_metadata_filter=filters,
        min_matching_documents=2,
        min_precision=0.5,
    )

    passed = score_advanced_rag_result(
        result=result_for(f"summary [doc {matching[0].id[:8]}]", [*matching, unrelated]),
        case=case,
        latency_ms=1,
    )
    failed = score_advanced_rag_result(
        result=result_for(f"summary [doc {matching[0].id[:8]}]", [matching[0], unrelated]),
        case=case,
        latency_ms=1,
    )

    assert passed.passed is True
    assert passed.recall == 1.0
    assert passed.precision == pytest.approx(2 / 3)
    assert failed.failures == ("matching_documents_below_2",)


def test_metadata_filter_case_roundtrips():
    """Serializable filter ground truth remains stable in experiment fingerprints."""
    original = AdvancedRAGEvaluationCase(
        question="q",
        expected_metadata_filter={"field": "meta.year", "operator": ">=", "value": 2020},
        min_matching_documents=3,
        min_precision=0.5,
    )
    assert AdvancedRAGEvaluationCase.from_dict(data=original.to_dict()) == original


def test_a_retrieval_carrying_a_filter_is_reported_as_one(document):
    """The filtered count is what shows whether metadata inspection changed how the Agent retrieved."""
    case = AdvancedRAGEvaluationCase(question="q", expected_document_ids=frozenset({document.id}))
    result = result_for("answer", [document])
    result["messages"].append(
        ChatMessage.from_assistant(
            tool_calls=[ToolCall("search_documents", {"query": "x", "filters": {"field": "meta.year"}}, id="filtered")]
        )
    )

    metrics = score_advanced_rag_result(result=result, case=case, latency_ms=1)

    assert metrics.retrieval_calls == 2
    assert metrics.filtered_retrieval_calls == 1
