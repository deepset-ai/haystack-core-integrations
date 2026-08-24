from haystack import Document
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    AdvancedRAGEvaluationCase,
    AdvancedRAGHarnessEvaluator,
)
from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    CampaignJournal,
    HarnessOptimizationCampaign,
    LocalTraceStore,
    ModelAsset,
    OptimizationObjectives,
    ToolAsset,
    TraceCapturingAgentRunner,
)


def test_advanced_rag_campaign_recommends_cheaper_model_at_quality_parity(tmp_path):
    document = Document(
        content="CRISPR gene editing can correct hereditary blindness mutations.",
        meta={"category": "science", "year": 2021},
    )
    store = InMemoryDocumentStore()
    store.write_documents([document])
    responses = [
        ChatMessage.from_assistant(tool_calls=[ToolCall("list_metadata_fields", {}, id="metadata")]),
        ChatMessage.from_assistant(
            tool_calls=[ToolCall("search_documents", {"query": "CRISPR", "filters": None}, id="retrieval")]
        ),
        ChatMessage.from_assistant(f"CRISPR can treat hereditary blindness [doc {document.id[:8]}]"),
    ]
    reference = create_advanced_rag_agent(
        document_store=store,
        retriever=InMemoryBM25Retriever(document_store=store),
        llm=MockChatGenerator(
            responses,
            model="reference",
            meta={"usage": {"input_tokens": 100, "output_tokens": 20}},
        ),
        backup_answer_llm=MockChatGenerator("backup", model="reference"),
        system_prompt="Inspect metadata, retrieve evidence, and cite it.",
    )
    trace_store = LocalTraceStore()
    captured = TraceCapturingAgentRunner().run(reference, messages=[ChatMessage.from_user("What is CRISPR used for?")])
    trace_store.add(captured.trace)

    tool_assets = [ToolAsset(tool.name) for tool in flatten_tools_or_toolsets(reference.tools)]
    assets = ApprovedAssetCatalog(
        models=[
            ModelAsset("reference", "closed", "remote", input_cost_per_million=10, output_cost_per_million=20),
            ModelAsset("cheap", "local", "eu", input_cost_per_million=1, output_cost_per_million=2),
        ],
        tools=tool_assets,
    )
    evaluator = AdvancedRAGHarnessEvaluator(
        cases=[
            AdvancedRAGEvaluationCase(
                question="What is CRISPR used for?",
                expected_document_ids=frozenset({document.id}),
                answer_must_mention=("CRISPR", "blindness"),
            )
        ],
        model_prices={"reference": (10, 20), "cheap": (1, 2)},
    )
    campaign = HarnessOptimizationCampaign(
        reference=reference,
        trace_source=trace_store,
        evaluator=evaluator,
        assets=assets,
        objectives=OptimizationObjectives(min_quality=1.0),
        journal=CampaignJournal(tmp_path / "advanced-rag-campaign.jsonl"),
        isolate_evaluations=False,
    )

    result = campaign.run()

    assert result.baseline.quality == 1.0
    assert result.recommendation is not None
    assert result.recommendation.evaluation.recipe == {"kind": "model_substitution", "model_id": "cheap"}
    assert result.recommendation.evaluation.metrics.quality == 1.0
    assert result.recommendation.evaluation.metrics.cost < result.baseline.cost
