from types import SimpleNamespace

import pytest
from haystack import Document, Pipeline, tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import (
    AdvancedRAGHarnessEvaluator,
    eval_case_from_reference_run,
)
from haystack_integrations.agent_pack.advanced_rag.tools import _make_retrieval_pipeline_tool
from haystack_integrations.agent_pack.dataclasses import RunRecord
from haystack_integrations.agent_pack.evaluation import RAGEvalCase
from haystack_integrations.agent_pack.optimization import ModelPrice, ModelPriceCatalog

EVIDENCE = "CRISPR is used for gene editing"

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
    return RunRecord(
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

    async def run_async(self, **kwargs):
        """Answer the way `run` does, through the entry point concurrent measurement uses."""
        result = self.run(**kwargs)
        usages = {self.chat_generator.model: result["token_usage"], **result.get("additional_model_usage", {})}
        for model, usage in usages.items():
            with tracing.tracer.trace("haystack.chat_generator.run") as span:
                span.set_content_tag(
                    "haystack.component.output",
                    {"replies": [ChatMessage.from_assistant("answer", meta={"model": model, "usage": usage})]},
                )
        return result

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


def test_derives_a_grounding_parity_eval_case_from_reference_run(document):
    eval_case = eval_case_from_reference_run(record=reference_run(document))
    assert eval_case.question == QUESTION
    assert eval_case.expected_document_ids == frozenset({document.id})


def test_evaluator_prices_the_run_from_the_price_catalog(document):
    """Cost must come from the catalog the experiment ranks against, not a second price table."""
    eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
    evaluator = AdvancedRAGHarnessEvaluator(eval_cases=[eval_case])

    metrics = catalog().price(
        metrics=evaluator.evaluate(target=FakeAgent(document), reference_runs=[reference_run(document=document)])
    )

    assert metrics.quality == 1.0
    assert metrics.cost == (100 * 2.0 + 20 * 4.0) / 1_000_000
    assert metrics.details["input_tokens"] == 100
    assert metrics.details["validated"] is True
    assert metrics.details["eval_cases"][0]["passed"] is True


def test_evaluator_includes_secondary_model_usage(document):
    class BackupAgent(FakeAgent):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            result["additional_model_usage"] = {"backup": {"input_tokens": 7, "output_tokens": 2}}
            return result

    evaluator = AdvancedRAGHarnessEvaluator(
        eval_cases=[RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})]
    )
    metrics = catalog().price(
        metrics=evaluator.evaluate(target=BackupAgent(document), reference_runs=[reference_run(document=document)])
    )

    assert metrics.model_usage["backup"].input_tokens == 7
    assert metrics.cost == pytest.approx((100 * 2.0 + 20 * 4.0 + 7 * 3.0 + 2 * 5.0) / 1_000_000)


def test_unpriced_models_are_reported_without_restricting_evaluation(document):
    """Unknown model usage remains a valid measurement with unavailable cost."""
    evaluator = AdvancedRAGHarnessEvaluator(
        eval_cases=[RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})]
    )
    metrics = evaluator.evaluate(
        target=FakeAgent(document, model="unknown"), reference_runs=[reference_run(document=document)]
    )
    priced = catalog().price(metrics=metrics)
    assert priced.cost is None
    assert priced.details["unpriced_models"] == ["unknown"]


def test_derived_eval_cases_are_reported_as_unvalidated(document):
    """Grounding parity with the incumbent is not a correctness measurement, and must be flagged as such."""
    evaluator = AdvancedRAGHarnessEvaluator()
    metrics = evaluator.evaluate(target=FakeAgent(document), reference_runs=[reference_run(document=document)])
    assert metrics.details["validated"] is False
    assert metrics.details["derived_cases"] == [QUESTION]


def test_every_eval_case_is_measured_once_and_latency_is_their_total(document):
    """One measurement per eval case: quality is the fraction that passed, with no variance estimate to report."""
    eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
    agent = FakeAgent(document)

    metrics = AdvancedRAGHarnessEvaluator(eval_cases=[eval_case]).evaluate(
        target=agent, reference_runs=[reference_run(document=document)]
    )

    assert agent.runs == 1
    assert agent.warmups == 1
    assert metrics.quality == 1.0
    assert metrics.latency_ms == pytest.approx(
        sum(eval_case_metrics["latency_ms"] for eval_case_metrics in metrics.details["eval_cases"])
    )


def test_evaluator_fingerprint_changes_with_the_evaluation_set(document):
    first = AdvancedRAGHarnessEvaluator(eval_cases=[RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})])
    second = AdvancedRAGHarnessEvaluator(eval_cases=[RAGEvalCase(question=QUESTION, evidence={"other": EVIDENCE})])
    assert first.fingerprint() != second.fingerprint()
    assert first.fingerprint() == AdvancedRAGHarnessEvaluator(eval_cases=list(first.eval_cases.values())).fingerprint()


def test_eval_case_details_carry_the_tool_trace(document):
    """A trace explains a result, and the digest must not become part of what identifies a measurement."""
    eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
    evaluator = AdvancedRAGHarnessEvaluator(eval_cases=[eval_case])
    fingerprint = evaluator.fingerprint()

    metrics = evaluator.evaluate(target=FakeAgent(document), reference_runs=[reference_run(document=document)])

    trace = metrics.details["eval_cases"][0]["run_digest"]
    assert [step["tool"] for step in trace["tool_steps"]] == ["list_metadata_fields", "search_documents"]
    assert trace["tool_steps"][1]["arguments"] == '{"query": "CRISPR"}'
    assert trace["tool_steps"][0]["result"] == "fields"
    assert metrics.details["eval_cases"][0]["backup_answer_used"] is False
    assert evaluator.fingerprint() == fingerprint
    assert set(fingerprint) == {"eval_cases"}


def test_a_run_cut_off_by_its_step_budget_is_reported_as_backup_answered(document):
    """
    The backup LLM is called by an after_run hook, so it is absent from what the Agent returns. A real Agent is
    driven to step exhaustion here rather than a fabricated result, since the hook running is the thing measured.
    """
    store = InMemoryDocumentStore()
    store.write_documents([document])
    searching = ChatMessage.from_assistant(tool_calls=[ToolCall("search_documents", {"query": "CRISPR"}, id="s")])
    agent = create_advanced_rag_agent(
        document_store=store,
        retriever=InMemoryBM25Retriever(document_store=store),
        # Never answers in text, so the run is cut off and the hook fires.
        llm=MockChatGenerator(response_fn=lambda *_args, **_kwargs: searching, model="main"),
        backup_answer_llm=MockChatGenerator(
            f"backup answer [doc {document.id[:8]}]",
            model="backup",
            meta={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        ),
        max_agent_steps=2,
    )
    eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})

    metrics = AdvancedRAGHarnessEvaluator(eval_cases=[eval_case]).evaluate(
        target=agent, reference_runs=[reference_run(document=document)]
    )

    assert metrics.details["eval_cases"][0]["backup_answer_used"] is True
    # The backup model is priced alongside the Agent's own, which is what the hook span also makes visible.
    assert "backup" in metrics.model_usage


def test_traces_are_dropped_from_passing_eval_cases_before_failing_ones(document):
    """Under a cap, the eval cases that need explaining keep their evidence, and every one is still reported."""
    failing = RAGEvalCase(question=QUESTION, evidence={"never retrieved": EVIDENCE})
    passing = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})

    failing_metrics = AdvancedRAGHarnessEvaluator(eval_cases=[failing]).evaluate(
        target=FakeAgent(document), reference_runs=[reference_run(document=document)]
    )
    passing_metrics = AdvancedRAGHarnessEvaluator(eval_cases=[passing], max_traced_eval_cases=0).evaluate(
        target=FakeAgent(document), reference_runs=[reference_run(document=document)]
    )

    assert failing_metrics.details["eval_cases"][0]["passed"] is False
    assert "run_digest" in failing_metrics.details["eval_cases"][0]
    # The trace is withheld past the cap, but the eval case is still reported.
    assert passing_metrics.details["eval_cases"][0]["passed"] is True
    assert "run_digest" not in passing_metrics.details["eval_cases"][0]


def test_eval_cases_measured_concurrently_are_reported_in_order(document):
    """Concurrency must change how long an evaluation takes, not what it measures."""
    questions = [f"{QUESTION} ({index})" for index in range(4)]

    class MultiQuestionAgent(FakeAgent):
        """Answer any of the questions, recording the order runs were started in."""

        def run(self, **kwargs):  # noqa: ARG002 - the reply does not depend on which question was asked
            """Return a successful result for whichever question was asked."""
            self.runs += 1
            return successful_result(self.document)

    eval_cases = [RAGEvalCase(question=q, evidence={document.id: EVIDENCE}) for q in questions]
    runs = [
        RunRecord(run_id=f"run-{index}", inputs={"messages": [ChatMessage.from_user(q)]}, outputs={})
        for index, q in enumerate(questions)
    ]

    sequential = AdvancedRAGHarnessEvaluator(eval_cases=eval_cases, max_concurrent_eval_cases=1).evaluate(
        target=MultiQuestionAgent(document), reference_runs=runs
    )
    concurrent = AdvancedRAGHarnessEvaluator(eval_cases=eval_cases, max_concurrent_eval_cases=4).evaluate(
        target=MultiQuestionAgent(document), reference_runs=runs
    )

    assert concurrent.quality == sequential.quality
    assert [eval_case["question"] for eval_case in concurrent.details["eval_cases"]] == [
        eval_case["question"] for eval_case in sequential.details["eval_cases"]
    ]
    assert concurrent.model_usage == sequential.model_usage


def test_concurrency_must_be_positive():
    """A concurrency of zero would measure nothing at all."""
    with pytest.raises(ValueError, match="at least 1"):
        AdvancedRAGHarnessEvaluator(max_concurrent_eval_cases=0)


def test_renamed_pipeline_tool_keeps_budget_and_nested_ranker_usage(document):
    store = InMemoryDocumentStore()
    store.write_documents([document])
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    pipeline.add_component(
        "ranker",
        LLMRanker(
            chat_generator=MockChatGenerator(
                '{"documents": [{"index": 1}]}',
                model="ranker",
                meta={"usage": {"input_tokens": 7, "output_tokens": 2}},
            )
        ),
    )
    pipeline.connect("retriever.documents", "ranker.documents")
    tool = _make_retrieval_pipeline_tool(
        pipeline=pipeline,
        name="ranked_search",
        input_mapping={"query": ["retriever.query", "ranker.query"], "filters": ["retriever.filters"]},
        output_mapping={"ranker.documents": "documents"},
    )
    agent = Agent(
        chat_generator=MockChatGenerator(
            [
                ChatMessage.from_assistant(tool_calls=[ToolCall("ranked_search", {"query": "CRISPR"}, id="call")]),
                ChatMessage.from_assistant(f"CRISPR [doc {document.id[:8]}]"),
            ],
            model="cheap",
            meta={"usage": {"input_tokens": 10, "output_tokens": 3}},
        ),
        tools=[tool],
        state_schema={"documents": {"type": list[Document]}},
    )
    # The eval case budgets the tool the reference had. The candidate renamed it, and the budget still binds.
    eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE}, tool_budgets={"search_documents": 0})
    metrics = AdvancedRAGHarnessEvaluator(eval_cases=[eval_case]).evaluate(agent, [reference_run(document)])
    assert metrics.details["usage_complete"]
    assert metrics.model_usage["ranker"].input_tokens == 7
    assert metrics.model_usage["cheap"].input_tokens == 20
    assert metrics.details["eval_cases"][0]["retrieval_calls"] == 1
    failures = metrics.details["eval_cases"][0]["failures"]
    assert failures == ["tool_calls_over_budget:fetch_documents_by_filter+ranked_search+search_documents:1/0"]
