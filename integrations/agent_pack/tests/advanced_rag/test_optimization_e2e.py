import yaml
from haystack import Document
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import AdvancedRAGHarnessEvaluator
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    load_agent,
)
from haystack_integrations.evaluation import RAGEvalCase

EVIDENCE = "CRISPR gene editing can correct hereditary blindness mutations."

QUESTION = "What is CRISPR used for?"


def optimizer_agent_for(change):
    """Drive actual file edits and validation using a scripted model."""
    stage = 0

    def respond(_messages, tools):
        nonlocal stage
        current = next(tool for tool in tools if tool.name == "read_config").function()
        if stage == 0:
            data = yaml.safe_load(current["yaml"])
            change(data["components"]["agent"]["init_parameters"])
            call = ToolCall(
                "edit_config",
                {
                    "old": current["yaml"],
                    "new": yaml.safe_dump(data),
                    "expected_revision": current["revision"],
                },
                id="edit",
            )
        elif stage == 1:
            call = ToolCall("validate_config", {}, id="validate")
        elif stage == 2:
            call = ToolCall(
                "submit_candidate",
                {
                    "expected_revision": current["revision"],
                    "rationale": "test hypothesis",
                },
                id="submit",
            )
        else:
            call = ToolCall("finish", {"reason": "nothing left worth measuring"}, id="finish")
        stage += 1
        return ChatMessage.from_assistant(tool_calls=[call])

    return create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=respond))


def scripted_agent(store, document, model):
    """Build a deterministic Advanced RAG Agent for end-to-end optimization tests."""
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


class TestAdvancedRagExperiment:
    def test_recommends_cheaper_model_at_parity(self, tmp_path):
        document = Document(
            content="CRISPR gene editing can correct hereditary blindness mutations.",
            meta={"category": "science", "year": 2021},
        )
        store = InMemoryDocumentStore()
        store.write_documents([document])
        reference = scripted_agent(store, document, "reference")
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
        experiment = HarnessOptimizationExperiment(
            reference=reference,
            eval_cases=[RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})],
            evaluator=AdvancedRAGHarnessEvaluator(),
            pricing=pricing,
            objectives=OptimizationObjectives(min_quality=1.0),
            journal=ExperimentJournal(directory=tmp_path / "journals"),
            optimizer_agent=optimizer_agent_for(
                lambda params: params["chat_generator"]["init_parameters"].update(model="cheap")
            ),
        )
        result = experiment.run()
        assert result.baseline.quality == 1.0
        assert result.recommendation is not None
        assert "model: cheap" in result.recommendation.configuration.yaml
        assert result.recommendation.reasons == ("cost_improvement",)
        assert result.recommendation.evaluation.metrics.quality == 1.0
        assert result.recommendation.evaluation.metrics.cost < result.baseline.cost
        approved = load_agent(result.recommendation.configuration.yaml)
        assert approved.chat_generator.model == "cheap"
        assert reference.chat_generator.model == "reference"

    def test_withholds_on_quality_regression(self, tmp_path):
        """The cheaper model is only recommended while it still answers the labelled eval case."""
        document = Document(content="CRISPR gene editing can correct hereditary blindness mutations.")
        store = InMemoryDocumentStore()
        store.write_documents([document])
        reference = scripted_agent(store, document, "reference")
        pricing = ModelPriceCatalog(
            prices=[
                ModelPrice(model_id="reference", input_cost_per_million=10),
            ]
        )
        experiment = HarnessOptimizationExperiment(
            reference=reference,
            eval_cases=[RAGEvalCase(question=QUESTION, evidence={document.id: EVIDENCE})],
            evaluator=AdvancedRAGHarnessEvaluator(),
            pricing=pricing,
            objectives=OptimizationObjectives(min_quality=1.0),
            journal=ExperimentJournal(directory=tmp_path / "journals"),
            optimizer_agent=optimizer_agent_for(lambda params: params.update(max_agent_steps=1)),
        )
        result = experiment.run()
        assert result.baseline.quality == 1.0
        assert result.recommendation is None
        candidate = result.candidates[0]
        assert candidate.metrics.quality == 0.0
        assert result.gate_failures[candidate.candidate_id] == ("quality_below_floor:1.0000",)
        assert "recall_below_1" in candidate.metrics.eval_cases[0]["failures"]
