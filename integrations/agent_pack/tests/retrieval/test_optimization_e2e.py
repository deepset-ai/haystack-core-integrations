import yaml
from haystack import Document, Pipeline
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.query import QueryExpander
from haystack.components.retrievers import InMemoryBM25Retriever, MultiQueryTextRetriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore
from retrieval import RetrievalEvaluationCase, RetrievalHarnessEvaluator

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    load_pipeline,
)
from haystack_integrations.agent_pack.optimization.local_run_store import LocalRunStore

QUESTION = "Which breakthroughs involved gene editing?"
EXPANSION = '{"queries": ["gene editing breakthrough"]}'


def optimizer_agent_for(change):
    """Drive real file edits and validation on the pipeline YAML using a scripted model."""
    stage = 0

    def respond(_messages, tools):
        nonlocal stage
        current = next(tool for tool in tools if tool.name == "read_config").function()
        if stage == 0:
            data = yaml.safe_load(current["yaml"])
            change(data["components"])
            call = ToolCall(
                "edit_config",
                {"old": current["yaml"], "new": yaml.safe_dump(data), "expected_revision": current["revision"]},
                id="edit",
            )
        elif stage == 1:
            call = ToolCall("validate_config", {}, id="validate")
        elif stage == 2:
            call = ToolCall(
                "submit_candidate",
                {"expected_revision": current["revision"], "rationale": "retrieve more per query"},
                id="submit",
            )
        else:
            call = ToolCall("finish", {"reason": "nothing left worth measuring"}, id="finish")
        stage += 1
        return ChatMessage.from_assistant(tool_calls=[call])

    return create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=respond))


def build_pipeline(store, top_k):
    pipeline = Pipeline()
    pipeline.add_component(
        "expander",
        QueryExpander(
            chat_generator=MockChatGenerator(EXPANSION, model="expander"),
            n_expansions=1,
            include_original_query=False,
        ),
    )
    pipeline.add_component(
        "retriever", MultiQueryTextRetriever(retriever=InMemoryBM25Retriever(document_store=store, top_k=top_k))
    )
    pipeline.connect("expander.queries", "retriever.queries")
    return pipeline


def test_a_plain_pipeline_is_optimized_without_being_wrapped_in_an_agent(tmp_path):
    """The experiment serializes a Pipeline as itself, so its components and connections are what gets edited."""
    documents = [
        Document(content="CRISPR gene editing corrected a hereditary blindness mutation."),
        Document(content="A second gene editing breakthrough silenced a disease gene."),
    ]
    store = InMemoryDocumentStore()
    store.write_documents(documents)

    reference = build_pipeline(store, top_k=1)
    run_store = LocalRunStore()
    run_store.add(record=AgentRunRecord(run_id="case-0", inputs={"query": QUESTION}, outputs={}))

    experiment = HarnessOptimizationExperiment(
        reference=reference,
        run_store=run_store,
        evaluator=RetrievalHarnessEvaluator(
            cases=[RetrievalEvaluationCase(question=QUESTION, expected_document_ids=frozenset(d.id for d in documents))]
        ),
        pricing=ModelPriceCatalog([ModelPrice(model_id="expander", input_cost_per_million=1)]),
        objectives=OptimizationObjectives(primary="quality"),
        journal=ExperimentJournal(directory=tmp_path / "journals"),
        optimizer_agent=optimizer_agent_for(
            lambda components: components["retriever"]["init_parameters"]["retriever"]["init_parameters"].update(
                top_k=2
            )
        ),
        max_iterations=1,
    )

    result = experiment.run()

    # One document of the two: quality is the recall, not whether the case cleared a threshold, so finding half
    # the evidence scores half rather than nothing.
    assert result.baseline.quality == 0.5
    assert result.baseline.details["mean_recall"] == 0.5
    assert result.baseline.details["cases"][0]["passed"] is False
    assert result.recommendation is not None
    assert result.recommendation.evaluation.metrics.quality == 1.0
    assert result.recommendation.reasons == ("quality_improvement",)

    approved = load_pipeline(result.recommendation.configuration.yaml)
    # A bare Pipeline, not an Agent wrapped in one.
    assert set(approved.graph.nodes) == {"expander", "retriever"}
    assert (result.artifact_directory / "reference.yaml").read_text().startswith("components:")
