from haystack import Document
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.evaluation import AdvancedRAGEvaluationCase
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import AdvancedRAGHarnessEvaluator
from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    ExperimentJournal,
    HarnessOptimizationExperiment,
    ModelAsset,
    ModelSubstitutionRecipe,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.runs import AgentRunRecorder, LocalRunStore

QUESTION = "What is CRISPR used for?"


class TryCheapModel:
    def __init__(self):
        self.proposed = False

    def propose(self, **kwargs):  # noqa: ARG002
        if self.proposed:
            return None
        self.proposed = True
        return ModelSubstitutionRecipe(model_id="cheap")


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
    AgentRunRecorder(run_store).run(reference, messages=[ChatMessage.from_user(QUESTION)])

    assets = ApprovedAssetCatalog(
        models=[
            ModelAsset(
                model_id="reference",
                input_cost_per_million=10,
                output_cost_per_million=20,
            ),
            ModelAsset(
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
        assets=assets,
        objectives=OptimizationObjectives(min_quality=1.0),
        journal=ExperimentJournal(path=tmp_path / "advanced-rag-experiment.jsonl"),
        proposer=TryCheapModel(),
    )

    result = experiment.run()

    assert result.baseline.quality == 1.0
    assert result.recommendation is not None
    assert result.recommendation.evaluation.recipe == {"kind": "model_substitution", "model_id": "cheap"}
    # Scored once per case, so the recommendation says so.
    assert result.recommendation.reasons == ("single_sample", "cost_improvement")
    assert result.recommendation.evaluation.metrics.quality == 1.0
    assert result.recommendation.evaluation.metrics.cost < result.baseline.cost
    assert result.recommendation.evaluation.metrics.details["validated"] is True

    approved = result.recommendation.materialize(reference, assets)
    assert approved.chat_generator.model == "cheap"
    assert reference.chat_generator.model == "reference"


def test_experiment_withholds_a_recommendation_when_quality_regresses(tmp_path):
    """The cheaper model is only recommended while it still answers the labelled case."""
    document = Document(content="CRISPR gene editing can correct hereditary blindness mutations.")
    store = InMemoryDocumentStore()
    store.write_documents([document])
    reference = scripted_agent(store, document, "reference")

    run_store = LocalRunStore()
    AgentRunRecorder(run_store).run(reference, messages=[ChatMessage.from_user(QUESTION)])

    assets = ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference", input_cost_per_million=10),
            ModelAsset(
                model_id="cheap",
                input_cost_per_million=1,
                # The cheap deployment answers without retrieving or citing anything.
                generator={
                    "type": "haystack.components.generators.chat.mock.MockChatGenerator",
                    "init_parameters": {
                        "model": "cheap",
                        "responses": [ChatMessage.from_assistant("CRISPR treats blindness.").to_dict()],
                        "meta": {"usage": {"input_tokens": 10, "output_tokens": 5}},
                    },
                },
            ),
        ],
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
        assets=assets,
        objectives=OptimizationObjectives(min_quality=1.0),
        journal=ExperimentJournal(path=tmp_path / "experiment.jsonl"),
        proposer=TryCheapModel(),
    )

    result = experiment.run()

    assert result.baseline.quality == 1.0
    assert result.recommendation is None
    candidate = result.candidates[0]
    assert candidate.metrics.quality == 0.0
    assert result.gate_failures[candidate.candidate_id] == ("quality_below_floor:1.0000",)
    assert "metadata_not_inspected_first" in candidate.metrics.details["cases"][0]["failures"]
