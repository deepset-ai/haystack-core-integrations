import asyncio
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
    _score_advanced_rag_result,
)
from haystack_integrations.agent_pack.advanced_rag.tools import _make_retrieval_pipeline_tool
from haystack_integrations.agent_pack.optimization import ModelPrice, ModelPriceCatalog
from haystack_integrations.evaluation import RAGEvalCase

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
        "exit_reason": "text",
        "token_usage": {"input_tokens": 100, "output_tokens": 20},
    }


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

    async def warm_up_async(self):
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


class TestScoreAdvancedRagResult:
    def test_scores_a_passing_run(self, document):
        eval_case = RAGEvalCase(
            question="What is CRISPR used for?",
            evidence={document.id: EVIDENCE},
        )
        metrics = _score_advanced_rag_result(
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
        metrics = _score_advanced_rag_result(
            result=result_for("CRISPR edits genes.", [document]), eval_case=eval_case, latency_ms=1
        )
        assert metrics.passed is False
        assert metrics.failures == ("answer_cites_nothing",)
        assert metrics.citations_resolved is True

    def test_citations_to_unretrieved_documents(self, document):
        eval_case = RAGEvalCase(question="q", evidence={document.id: EVIDENCE}, require_citations=False)
        metrics = _score_advanced_rag_result(
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
        metrics = _score_advanced_rag_result(
            result=result_for("CRISPR is used for gene editing", [document], errors=("search_documents",)),
            eval_case=eval_case,
            latency_ms=1,
        )
        assert metrics.passed is False
        assert set(metrics.failures) == {"tool_errors:1", "steps_over_budget:3"}

    def test_reports_tool_order(self, document):
        eval_case = RAGEvalCase(question="q", evidence={document.id: EVIDENCE}, require_citations=False)
        metrics = _score_advanced_rag_result(
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
        metrics = _score_advanced_rag_result(result=result, eval_case=eval_case, latency_ms=1)
        assert metrics.retrieval_calls == 2
        assert metrics.filtered_retrieval_calls == 1


class TestEvaluate:
    def test_prices_from_the_catalog(self, document):
        """Cost must come from the catalog the experiment ranks against, not a second price table."""
        eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
        evaluator = AdvancedRAGHarnessEvaluator()
        metrics = catalog().price(metrics=evaluator.evaluate(target=FakeAgent(document), eval_cases=[eval_case]))
        assert metrics.quality == 1.0
        assert metrics.cost == (100 * 2.0 + 20 * 4.0) / 1_000_000
        assert metrics.details["input_tokens"] == 100
        assert metrics.eval_cases[0]["passed"] is True

    def test_secondary_model_usage(self, document):
        class BackupAgent(FakeAgent):
            def run(self, **kwargs):
                result = super().run(**kwargs)
                result["additional_model_usage"] = {"backup": {"input_tokens": 7, "output_tokens": 2}}
                return result

        eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
        evaluator = AdvancedRAGHarnessEvaluator()
        metrics = catalog().price(metrics=evaluator.evaluate(target=BackupAgent(document), eval_cases=[eval_case]))
        assert metrics.model_usage["backup"].input_tokens == 7
        assert metrics.cost == pytest.approx((100 * 2.0 + 20 * 4.0 + 7 * 3.0 + 2 * 5.0) / 1_000_000)

    def test_unpriced_models(self, document):
        """Unknown model usage remains a valid measurement with unavailable cost."""
        eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
        evaluator = AdvancedRAGHarnessEvaluator()
        metrics = evaluator.evaluate(target=FakeAgent(document, model="unknown"), eval_cases=[eval_case])
        priced = catalog().price(metrics=metrics)
        assert priced.cost is None
        assert priced.details["unpriced_models"] == ["unknown"]

    def test_latency_totals_the_eval_cases(self, document):
        """One measurement per eval case: quality is the fraction that passed, with no variance estimate to report."""
        eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
        agent = FakeAgent(document)
        metrics = AdvancedRAGHarnessEvaluator().evaluate(target=agent, eval_cases=[eval_case])
        assert agent.runs == 1
        assert agent.warmups == 1
        assert metrics.quality == 1.0
        assert metrics.latency_ms == pytest.approx(
            sum(eval_case_metrics["latency_ms"] for eval_case_metrics in metrics.eval_cases)
        )

    def test_eval_cases_carry_the_run_digest(self, document):
        """A trace is what explains a result, so a passing eval case still reports what the run actually did."""
        eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
        metrics = AdvancedRAGHarnessEvaluator().evaluate(target=FakeAgent(document), eval_cases=[eval_case])
        trace = metrics.eval_cases[0]["agent_run_digest"]
        assert [step["tool"] for step in trace["tool_steps"]] == ["list_metadata_fields", "search_documents"]
        assert trace["tool_steps"][1]["arguments"] == '{"query": "CRISPR"}'
        assert trace["tool_steps"][0]["result"] == "fields"
        assert metrics.eval_cases[0]["exit_reason"] == "text"

    def test_step_budget_exit_reason(self, document):
        """
        A citation failure means something different when the step budget ran out: the backup LLM wrote the answer
        and does not cite. A real Agent is driven to exhaustion here, so the hook's own model call is measured too.
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
        metrics = AdvancedRAGHarnessEvaluator().evaluate(target=agent, eval_cases=[eval_case])
        assert metrics.eval_cases[0]["exit_reason"] == "max_agent_steps"
        # The backup model is priced alongside the Agent's own, which is what the hook span also makes visible.
        assert "backup" in metrics.model_usage

    def test_digests_dropped_from_passing_first(self, document):
        """Under a cap, the eval cases that need explaining keep their evidence, and every one is still reported."""
        failing = RAGEvalCase(question=QUESTION, evidence={"never retrieved": EVIDENCE})
        passing = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
        failing_metrics = AdvancedRAGHarnessEvaluator().evaluate(target=FakeAgent(document), eval_cases=[failing])
        passing_metrics = AdvancedRAGHarnessEvaluator(max_traced_eval_cases=0).evaluate(
            target=FakeAgent(document), eval_cases=[passing]
        )
        assert failing_metrics.eval_cases[0]["passed"] is False
        assert "agent_run_digest" in failing_metrics.eval_cases[0]
        # The trace is withheld past the cap, but the eval case is still reported.
        assert passing_metrics.eval_cases[0]["passed"] is True
        assert "agent_run_digest" not in passing_metrics.eval_cases[0]

    def test_concurrency(self, document):
        """Concurrency must change how long an evaluation takes, not what it measures."""
        questions = [f"{QUESTION} ({index})" for index in range(4)]

        class MultiQuestionAgent(FakeAgent):
            """Answer any of the questions, recording the order runs were started in."""

            def run(self, **kwargs):  # noqa: ARG002 - the reply does not depend on which question was asked
                """Return a successful result for whichever question was asked."""
                self.runs += 1
                return successful_result(self.document)

        eval_cases = [RAGEvalCase(question=q, evidence={document.id: EVIDENCE}) for q in questions]
        sequential = AdvancedRAGHarnessEvaluator(max_concurrent_eval_cases=1).evaluate(
            target=MultiQuestionAgent(document), eval_cases=eval_cases
        )
        concurrent = AdvancedRAGHarnessEvaluator(max_concurrent_eval_cases=4).evaluate(
            target=MultiQuestionAgent(document), eval_cases=eval_cases
        )
        assert concurrent.quality == sequential.quality
        assert [eval_case["question"] for eval_case in concurrent.eval_cases] == [
            eval_case["question"] for eval_case in sequential.eval_cases
        ]
        assert concurrent.model_usage == sequential.model_usage

    def test_renamed_pipeline_tool(self, document):
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
        eval_case = RAGEvalCase(
            question=QUESTION, evidence={document.id: EVIDENCE}, tool_budgets={"search_documents": 0}
        )
        metrics = AdvancedRAGHarnessEvaluator().evaluate(agent, eval_cases=[eval_case])
        assert metrics.details["all_tokens_reported"]
        assert metrics.model_usage["ranker"].input_tokens == 7
        assert metrics.model_usage["cheap"].input_tokens == 20
        assert metrics.eval_cases[0]["retrieval_calls"] == 1
        failures = metrics.eval_cases[0]["failures"]
        assert failures == ["tool_calls_over_budget:fetch_documents_by_filter+ranked_search+search_documents:1/0"]

    def test_init_invalid_concurrency(self):
        """A concurrency of zero would measure nothing at all."""
        with pytest.raises(ValueError, match="at least 1"):
            AdvancedRAGHarnessEvaluator(max_concurrent_eval_cases=0)


class TestEvaluateAsync:
    @pytest.mark.asyncio
    async def test_matches_sync(self, document):
        """The sync entry point cannot run under a loop, so an async caller has to reach the same work directly."""
        eval_case = RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})
        awaited = await AdvancedRAGHarnessEvaluator().evaluate_async(target=FakeAgent(document), eval_cases=[eval_case])
        # The sync entry point needs a thread of its own here, since it starts a loop and one is already running.
        blocking = await asyncio.to_thread(
            AdvancedRAGHarnessEvaluator().evaluate,
            target=FakeAgent(document),
            eval_cases=[eval_case],
        )
        assert awaited.quality == blocking.quality == 1.0
        assert awaited.model_usage == blocking.model_usage
        assert [entry["failures"] for entry in awaited.eval_cases] == [[]]
