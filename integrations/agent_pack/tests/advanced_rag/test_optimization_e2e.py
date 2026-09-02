from haystack import Document
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.evaluation import AdvancedRAGEvaluationCase
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import AdvancedRAGHarnessEvaluator
from haystack_integrations.agent_pack.optimization import (
    AgentMutation,
    ExperimentJournal,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    MutationOperation,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.runs import AgentRunRecorder, LocalRunStore

QUESTION = "What is CRISPR used for?"


class ProposeOnce:
    """Return one arbitrary configuration mutation and then stop."""

    def __init__(self, mutation):
        """Store the sole mutation."""
        self.mutation = mutation
        self.proposed = False

    def propose(self, **kwargs):  # noqa: ARG002
        """Return the stored mutation at most once."""
        if self.proposed:
            return None
        self.proposed = True
        return self.mutation


def scripted_agent(store, document, model):
    responses = [
        ChatMessage.from_assistant(tool_calls=[ToolCall("list_metadata_fields", {}, id="metadata")]),
        ChatMessage.from_assistant(
            tool_calls=[ToolCall("search_documents", {"query": "CRISPR", "filters": None}, id="retrieval")]
        ),
        ChatMessage.from_assistant(f"CRISPR can treat hereditary blindness [doc {document.id[:8]}]"),
    ]
    return create_advanced_rag_agent(
        document_store=store,
        retriever=InMemoryBM25Retriever(document_store=store),
        llm=MockChatGenerator(responses, model=model, meta={"usage": {"input_tokens": 100, "output_tokens": 20}}),
        backup_answer_llm=MockChatGenerator("backup", model=model),
        system_prompt="Inspect metadata, retrieve evidence, and cite it.",
    )


def test_advanced_rag_experiment_recommends_cheaper_model_at_quality_parity(tmp_path):
    document = Document(
        content="CRISPR gene editing can correct hereditary blindness mutations.",
        meta={"category": "science", "year": 2021},
    )
    store = InMemoryDocumentStore()
    store.write_documents([document])

    reference = scripted_agent(store, document, "reference")
    run_store = LocalRunStore()
    AgentRunRecorder(store=run_store).run(agent=reference, messages=[ChatMessage.from_user(text=QUESTION)])

    pricing = ModelPriceCatalog(
        prices=[
            ModelPrice(
                model_id="reference",
                input_cost_per_million=10,
                output_cost_per_million=20,
            ),
            ModelPrice(
                model_id="cheap",
                input_cost_per_million=1,
                output_cost_per_million=2,
            ),
        ],
    )
    evaluator = AdvancedRAGHarnessEvaluator(
        cases=[
            AdvancedRAGEvaluationCase(
                question=QUESTION,
                expected_document_ids=frozenset({document.id}),
                answer_must_mention=("CRISPR", "blindness"),
            )
        ]
    )
    experiment = HarnessOptimizationExperiment(
        reference=reference,
        run_source=run_store,
        evaluator=evaluator,
        pricing=pricing,
        objectives=OptimizationObjectives(min_quality=1.0),
        journal=ExperimentJournal(path=tmp_path / "advanced-rag-experiment.jsonl"),
        proposer=ProposeOnce(
            mutation=AgentMutation(
                operations=(
                    MutationOperation(
                        op="set",
                        path="/init_parameters/chat_generator/init_parameters/model",
                        value="cheap",
                    ),
                )
            )
        ),
    )

    result = experiment.run()

    assert result.baseline.quality == 1.0
    assert result.recommendation is not None
    assert result.recommendation.evaluation.mutation["operations"][0]["value"] == "cheap"
    # Scored once per case, so the recommendation says so.
    assert result.recommendation.reasons == ("single_sample", "cost_improvement")
    assert result.recommendation.evaluation.metrics.quality == 1.0
    assert result.recommendation.evaluation.metrics.cost < result.baseline.cost
    assert result.recommendation.evaluation.metrics.details["validated"] is True

    approved = result.recommendation.materialize(reference=reference)
    assert approved.chat_generator.model == "cheap"
    assert reference.chat_generator.model == "reference"


def test_experiment_withholds_a_recommendation_when_quality_regresses(tmp_path):
    """The cheaper model is only recommended while it still answers the labelled case."""
    document = Document(content="CRISPR gene editing can correct hereditary blindness mutations.")
    store = InMemoryDocumentStore()
    store.write_documents([document])
    reference = scripted_agent(store, document, "reference")

    run_store = LocalRunStore()
    AgentRunRecorder(store=run_store).run(agent=reference, messages=[ChatMessage.from_user(text=QUESTION)])

    pricing = ModelPriceCatalog(
        prices=[
            ModelPrice(model_id="reference", input_cost_per_million=10),
        ]
    )
    experiment = HarnessOptimizationExperiment(
        reference=reference,
        run_source=run_store,
        evaluator=AdvancedRAGHarnessEvaluator(
            cases=[
                AdvancedRAGEvaluationCase(
                    question=QUESTION,
                    expected_document_ids=frozenset({document.id}),
                    answer_must_mention=("CRISPR", "blindness"),
                )
            ]
        ),
        pricing=pricing,
        objectives=OptimizationObjectives(min_quality=1.0),
        journal=ExperimentJournal(path=tmp_path / "experiment.jsonl"),
        proposer=ProposeOnce(
            mutation=AgentMutation(
                operations=(MutationOperation(op="set", path="/init_parameters/max_agent_steps", value=1),)
            )
        ),
    )

    result = experiment.run()

    assert result.baseline.quality == 1.0
    assert result.recommendation is None
    candidate = result.candidates[0]
    assert candidate.metrics.quality == 0.0
    assert result.gate_failures[candidate.candidate_id] == ("quality_below_floor:1.0000",)
    assert "recall_below_1" in candidate.metrics.details["cases"][0]["failures"]
