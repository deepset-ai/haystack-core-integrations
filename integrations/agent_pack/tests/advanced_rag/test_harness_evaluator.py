from types import SimpleNamespace

import pytest
from haystack import Document
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.advanced_rag.evaluation import AdvancedRAGEvaluationCase
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import (
    AdvancedRAGHarnessEvaluator,
    case_from_reference_run,
)
from haystack_integrations.agent_pack.dataclasses import AgentRunRecord
from haystack_integrations.agent_pack.optimization import ModelPrice, ModelPriceCatalog

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
    """Create prices for all models attributed by the evaluator."""
    return ModelPriceCatalog(
        prices=[
            ModelPrice(
                model_id="cheap",
                input_cost_per_million=2.0,
                output_cost_per_million=4.0,
            ),
            ModelPrice(model_id="reference", input_cost_per_million=10.0),
            ModelPrice(model_id="backup", input_cost_per_million=3.0, output_cost_per_million=5.0),
        ],
    )


def test_derives_grounding_parity_case_from_reference_run(document):
    case = case_from_reference_run(record=reference_run(document))
    assert case.question == QUESTION
    assert case.expected_document_ids == frozenset({document.id})


def test_evaluator_prices_the_run_from_the_price_catalog(document):
    """Cost must come from the catalog the experiment ranks against, not a second price table."""
    case = AdvancedRAGEvaluationCase(
        question=QUESTION, expected_document_ids=frozenset({document.id}), answer_must_mention=("CRISPR",)
    )
    evaluator = AdvancedRAGHarnessEvaluator(cases=[case])

    metrics = catalog().price(
        metrics=evaluator.evaluate(agent=FakeAgent(document), reference_runs=[reference_run(document=document)])
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
    metrics = catalog().price(
        metrics=evaluator.evaluate(agent=BackupAgent(document), reference_runs=[reference_run(document=document)])
    )

    assert metrics.model_usage["backup"].input_tokens == 7
    assert metrics.cost == pytest.approx((100 * 2.0 + 20 * 4.0 + 7 * 3.0 + 2 * 5.0) / 1_000_000)


def test_unpriced_models_are_reported_without_restricting_evaluation(document):
    """Unknown model usage remains a valid measurement with unavailable cost."""
    evaluator = AdvancedRAGHarnessEvaluator(cases=[AdvancedRAGEvaluationCase(question=QUESTION, expect_absent=True)])
    metrics = evaluator.evaluate(
        agent=FakeAgent(document, model="unknown"), reference_runs=[reference_run(document=document)]
    )
    priced = catalog().price(metrics=metrics)
    assert priced.cost is None
    assert priced.details["unpriced_models"] == ["unknown"]


def test_derived_cases_are_reported_as_unvalidated(document):
    """Grounding parity with the incumbent is not a correctness measurement, and must be flagged as such."""
    evaluator = AdvancedRAGHarnessEvaluator()
    metrics = evaluator.evaluate(agent=FakeAgent(document), reference_runs=[reference_run(document=document)])
    assert metrics.details["validated"] is False
    assert metrics.details["derived_cases"] == [QUESTION]


def test_every_case_is_measured_once_and_latency_is_their_total(document):
    """One measurement per case: quality is the fraction that passed, with no variance estimate to report."""
    case = AdvancedRAGEvaluationCase(
        question=QUESTION, expected_document_ids=frozenset({document.id}), answer_must_mention=("CRISPR",)
    )
    agent = FakeAgent(document)

    metrics = AdvancedRAGHarnessEvaluator(cases=[case]).evaluate(
        agent=agent, reference_runs=[reference_run(document=document)]
    )

    assert agent.runs == 1
    assert agent.warmups == 1
    assert metrics.quality == 1.0
    assert metrics.quality_lower_bound is None
    assert metrics.latency_ms == pytest.approx(
        sum(case_metrics["latency_ms"] for case_metrics in metrics.details["cases"])
    )


def test_evaluator_fingerprint_changes_with_the_evaluation_set(document):
    first = AdvancedRAGHarnessEvaluator(
        cases=[AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({document.id}))]
    )
    second = AdvancedRAGHarnessEvaluator(
        cases=[AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({"other"}))]
    )
    assert first.fingerprint() != second.fingerprint()
    assert first.fingerprint() == AdvancedRAGHarnessEvaluator(cases=list(first.cases.values())).fingerprint()


def test_case_details_carry_the_tool_trace(document):
    """A trace explains a result, and the digest must not become part of what identifies a measurement."""
    case = AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({document.id}))
    evaluator = AdvancedRAGHarnessEvaluator(cases=[case])
    fingerprint = evaluator.fingerprint()

    metrics = evaluator.evaluate(agent=FakeAgent(document), reference_runs=[reference_run(document=document)])

    trace = metrics.details["cases"][0]["run_digest"]
    assert [step["tool"] for step in trace["tool_steps"]] == ["list_metadata_fields", "search_documents"]
    assert trace["tool_steps"][1]["arguments"] == '{"query": "CRISPR"}'
    assert trace["tool_steps"][0]["result"] == "fields"
    assert metrics.details["cases"][0]["backup_answer_used"] is False
    assert evaluator.fingerprint() == fingerprint
    assert set(fingerprint) == {"cases"}


def test_a_run_cut_off_by_its_step_budget_is_reported_as_backup_answered(document):
    """`answer_cites_nothing` on a truncated run is the backup hook's doing, not a retrieval fault."""

    class CutOffAgent(FakeAgent):
        """Return a run that ended on the step budget with the backup hook's usage attached."""

        def run(self, **kwargs):  # noqa: ARG002
            """Report the shape a cut-off run has."""
            result = successful_result(self.document)
            result["exit_reason"] = "max_agent_steps"
            result["additional_model_usage"] = {"backup": {"input_tokens": 10, "output_tokens": 5}}
            return result

    case = AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({document.id}))
    evaluator = AdvancedRAGHarnessEvaluator(cases=[case])

    metrics = evaluator.evaluate(agent=CutOffAgent(document), reference_runs=[reference_run(document=document)])

    assert metrics.details["cases"][0]["backup_answer_used"] is True


def test_traces_are_dropped_from_passing_cases_before_failing_ones(document):
    """Under a cap, the cases that need explaining keep their evidence, and every case is still reported."""
    failing = AdvancedRAGEvaluationCase(
        question=QUESTION, expected_document_ids=frozenset({document.id}), answer_must_mention=("absent term",)
    )
    passing = AdvancedRAGEvaluationCase(question=QUESTION, expected_document_ids=frozenset({document.id}))

    failing_metrics = AdvancedRAGHarnessEvaluator(cases=[failing]).evaluate(
        agent=FakeAgent(document), reference_runs=[reference_run(document=document)]
    )
    passing_metrics = AdvancedRAGHarnessEvaluator(cases=[passing], max_traced_cases=0).evaluate(
        agent=FakeAgent(document), reference_runs=[reference_run(document=document)]
    )

    assert failing_metrics.details["cases"][0]["passed"] is False
    assert "run_digest" in failing_metrics.details["cases"][0]
    # The trace is withheld past the cap, but the case is still reported.
    assert passing_metrics.details["cases"][0]["passed"] is True
    assert "run_digest" not in passing_metrics.details["cases"][0]
