from types import SimpleNamespace

import pytest
from haystack import Document
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.advanced_rag.evaluation import AdvancedRAGEvaluationCase
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import (
    AdvancedRAGHarnessEvaluator,
    case_from_reference_run,
)
from haystack_integrations.agent_pack.optimization import ApprovedAssetCatalog, ModelAsset
from haystack_integrations.agent_pack.runs import AgentRunRecord

QUESTION = "What is CRISPR used for?"


def successful_result(document):
    metadata_call = ToolCall("list_metadata_fields", {}, id="metadata")
    retrieval_call = ToolCall("search_documents", {"query": "CRISPR"}, id="retrieval")
    answer = f"CRISPR evidence [doc {document.id[:8]}]"
    return {
        "messages": [
            ChatMessage.from_assistant(tool_calls=[metadata_call]),
            ChatMessage.from_tool("fields", origin=metadata_call),
            ChatMessage.from_assistant(tool_calls=[retrieval_call]),
            ChatMessage.from_tool("documents", origin=retrieval_call),
            ChatMessage.from_assistant(answer),
        ],
        "last_message": ChatMessage.from_assistant(answer),
        "documents": [document],
        "step_count": 3,
        "token_usage": {"input_tokens": 100, "output_tokens": 20},
    }


def reference_run(document):
    return AgentRunRecord(
        run_id="rag-reference",
        inputs={"messages": [ChatMessage.from_user(QUESTION)]},
        outputs={"last_message": ChatMessage.from_assistant("reference"), "documents": [document]},
    )


class FakeAgent:
    def __init__(self, document, model="cheap"):
        self.document = document
        self.chat_generator = SimpleNamespace(model=model)
        self.runs = 0
        self.warmups = 0

    def run(self, **kwargs):
        assert kwargs["messages"][0].text == QUESTION
        self.runs += 1
        return successful_result(self.document)

    def warm_up(self):
        self.warmups += 1


@pytest.fixture
def document():
    return Document(content="CRISPR is used for gene editing")


def catalog():
    return ApprovedAssetCatalog(
        models=[
            ModelAsset(
                model_id="cheap",
                input_cost_per_million=2.0,
                output_cost_per_million=4.0,
            ),
            ModelAsset(model_id="reference", input_cost_per_million=10.0),
            ModelAsset(model_id="backup", input_cost_per_million=3.0, output_cost_per_million=5.0),
        ],
    )


def test_derives_grounding_parity_case_from_reference_run(document):
    case = case_from_reference_run(record=reference_run(document))
    assert case.question == QUESTION
    assert case.expected_document_ids == frozenset({document.id})


def test_evaluator_prices_the_run_from_the_approved_asset_catalog(document):
    """Cost must come from the catalog the experiment gates against, not a second price table."""
    case = AdvancedRAGEvaluationCase(
        question=QUESTION, expected_document_ids=frozenset({document.id}), answer_must_mention=("CRISPR",)
    )
    evaluator = AdvancedRAGHarnessEvaluator(cases=[case])

    metrics = evaluator.evaluate(agent=FakeAgent(document), reference_runs=[reference_run(document=document)]).price(
        assets=catalog()
    )

    assert metrics.quality == 1.0
    assert metrics.cost == (100 * 2.0 + 20 * 4.0) / 1_000_000
    assert metrics.details["input_tokens"] == 100
    assert metrics.details["validated"] is True
    assert metrics.details["cases"][0]["passed"] is True


def test_evaluator_includes_secondary_model_usage(document):
    class BackupAgent(FakeAgent):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            result["additional_model_usage"] = {"backup": {"input_tokens": 7, "output_tokens": 2}}
            return result

    evaluator = AdvancedRAGHarnessEvaluator(
        cases=[AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({document.id}))]
    )
    metrics = evaluator.evaluate(agent=BackupAgent(document), reference_runs=[reference_run(document=document)]).price(
        assets=catalog()
    )

    assert metrics.model_usage["backup"].input_tokens == 7
    assert metrics.cost == pytest.approx((100 * 2.0 + 20 * 4.0 + 7 * 3.0 + 2 * 5.0) / 1_000_000)


def test_unpriced_models_fail_when_results_are_priced(document):
    evaluator = AdvancedRAGHarnessEvaluator(cases=[AdvancedRAGEvaluationCase(question=QUESTION, expect_absent=True)])
    metrics = evaluator.evaluate(
        agent=FakeAgent(document, model="unknown"), reference_runs=[reference_run(document=document)]
    )
    with pytest.raises(ValueError, match="not in the approved asset catalog"):
        metrics.price(assets=catalog())


def test_derived_cases_are_reported_as_unvalidated(document):
    """Grounding parity with the incumbent is not a correctness measurement, and must be flagged as such."""
    evaluator = AdvancedRAGHarnessEvaluator()
    metrics = evaluator.evaluate(agent=FakeAgent(document), reference_runs=[reference_run(document=document)])
    assert metrics.details["validated"] is False
    assert metrics.details["derived_cases"] == [QUESTION]


def test_repetitions_produce_a_quality_lower_bound(document):
    case = AdvancedRAGEvaluationCase(
        question=QUESTION, expected_document_ids=frozenset({document.id}), answer_must_mention=("CRISPR",)
    )
    agent = FakeAgent(document)
    metrics = AdvancedRAGHarnessEvaluator(cases=[case], repetitions=3).evaluate(
        agent=agent, reference_runs=[reference_run(document=document)]
    )

    assert agent.runs == 3
    assert agent.warmups == 1
    assert metrics.details["repetitions"] == 3
    assert metrics.quality == 1.0
    assert metrics.quality_lower_bound == 1.0
    assert metrics.details["quality_stdev"] == 0.0
    # Latency is averaged per repetition so it stays comparable to a single-repetition baseline.
    assert metrics.latency_ms == pytest.approx(
        sum(case_metrics["latency_ms"] for case_metrics in metrics.details["cases"]) / 3
    )


def test_a_flaky_candidate_reports_a_lower_bound_below_its_mean(document):
    class FlakyAgent(FakeAgent):
        def run(self, **kwargs):  # noqa: ARG002 - the reply does not depend on the request
            self.runs += 1
            result = successful_result(self.document)
            if self.runs == 1:
                result["documents"] = []
            return result

    case = AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({document.id}))
    metrics = AdvancedRAGHarnessEvaluator(cases=[case], repetitions=2).evaluate(
        agent=FlakyAgent(document), reference_runs=[reference_run(document=document)]
    )

    assert metrics.quality == 0.5
    assert metrics.quality_lower_bound is not None
    assert metrics.quality_lower_bound < metrics.quality


def test_evaluator_fingerprint_changes_with_the_evaluation_set(document):
    first = AdvancedRAGHarnessEvaluator(
        cases=[AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({document.id}))]
    )
    second = AdvancedRAGHarnessEvaluator(
        cases=[AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({"other"}))]
    )
    assert first.fingerprint() != second.fingerprint()
    assert first.fingerprint() == AdvancedRAGHarnessEvaluator(cases=list(first.cases.values())).fingerprint()


def test_repetitions_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        AdvancedRAGHarnessEvaluator(repetitions=0)
