from datetime import UTC, datetime

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    CampaignJournal,
    EvaluationMetrics,
    HarnessOptimizationCampaign,
    IsolatedHarnessEvaluator,
    LocalTraceStore,
    ModelAsset,
    OptimizationObjectives,
    TraceArtifact,
)


def reference_trace():
    now = datetime.now(tz=UTC).isoformat()
    return TraceArtifact(
        run_id="reference-run",
        started_at=now,
        finished_at=now,
        duration_ms=1.0,
        status="success",
        traces=(
            {
                "span_id": "root",
                "parent_span_id": None,
                "operation_name": "haystack.agent.run",
                "component": None,
                "start_time": now,
                "end_time": now,
                "duration_ms": 1.0,
                "tags": {
                    "haystack.agent.input": {"messages": [ChatMessage.from_user("question").to_dict()]},
                    "haystack.agent.output": {
                        "last_message": ChatMessage.from_assistant("answer").to_dict(),
                        "documents": [],
                    },
                },
            },
        ),
    )


class ModelEvaluator:
    def __init__(self, metrics_by_model):
        self.metrics_by_model = metrics_by_model
        self.calls = []

    def evaluate(self, agent, reference_traces):
        assert reference_traces[0].run_id == "reference-run"
        self.calls.append(agent.chat_generator.model)
        return self.metrics_by_model[agent.chat_generator.model]


def build_campaign(tmp_path, evaluator, *, objectives=None):
    store = LocalTraceStore()
    store.add(reference_trace())
    reference = Agent(chat_generator=MockChatGenerator(model="reference"))
    assets = ApprovedAssetCatalog(
        models=[
            ModelAsset("reference", "closed", "remote", input_cost_per_million=10),
            ModelAsset("cheap", "local", "eu", sovereign=True, input_cost_per_million=2),
            ModelAsset("bad", "local", "eu", sovereign=True, input_cost_per_million=1),
        ],
        tools=[],
    )
    return (
        HarnessOptimizationCampaign(
            reference=reference,
            trace_source=store,
            evaluator=evaluator,
            assets=assets,
            objectives=objectives or OptimizationObjectives(min_quality=0.8),
            journal=CampaignJournal(tmp_path / "campaign.jsonl"),
            isolate_evaluations=False,
        ),
        assets,
        reference,
    )


def test_campaign_applies_gates_then_recommends_cheaper_candidate(tmp_path):
    evaluator = ModelEvaluator(
        {
            "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
            "cheap": EvaluationMetrics(
                quality=1.0,
                cost=2.0,
                latency_ms=90,
                details={"policy_decisions": [{"policy_version": "v1", "rule_id": "allow"}]},
            ),
            "bad": EvaluationMetrics(quality=0.5, cost=1.0, latency_ms=50),
        }
    )
    campaign, assets, reference = build_campaign(tmp_path, evaluator)

    result = campaign.run()

    assert result.recommendation is not None
    assert result.recommendation.evaluation.metrics.cost == 2.0
    assert result.recommendation.evaluation.policy_decisions[0]["policy_version"] == "v1"
    approved = result.recommendation.materialize(reference, assets)
    assert approved.chat_generator.model == "cheap"
    assert reference.chat_generator.model == "reference"
    assert {candidate.valid for candidate in result.candidates} == {True, False}


def test_campaign_resumes_journaled_candidates(tmp_path):
    evaluator = ModelEvaluator(
        {
            "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
            "cheap": EvaluationMetrics(quality=1.0, cost=2.0, latency_ms=90),
            "bad": EvaluationMetrics(quality=0.5, cost=1.0, latency_ms=50),
        }
    )
    campaign, _, _ = build_campaign(tmp_path, evaluator)
    campaign.run()
    campaign.run()

    assert evaluator.calls.count("reference") == 2
    assert evaluator.calls.count("cheap") == 1
    assert evaluator.calls.count("bad") == 1


def test_sovereignty_gate_can_outweigh_baseline_cost(tmp_path):
    evaluator = ModelEvaluator(
        {
            "reference": EvaluationMetrics(quality=1.0, cost=1.0, latency_ms=50),
            "cheap": EvaluationMetrics(quality=1.0, cost=2.0, latency_ms=60),
            "bad": EvaluationMetrics(quality=1.0, cost=3.0, latency_ms=70),
        }
    )
    campaign, _, _ = build_campaign(
        tmp_path, evaluator, objectives=OptimizationObjectives(min_quality=1.0, require_sovereign=True)
    )

    result = campaign.run()

    assert result.recommendation is not None
    assert result.recommendation.evaluation.recipe["model_id"] == "cheap"


def test_campaign_requires_replayable_successful_traces(tmp_path):
    store = LocalTraceStore()
    evaluator = ModelEvaluator({})
    campaign = HarnessOptimizationCampaign(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        trace_source=store,
        evaluator=evaluator,
        assets=ApprovedAssetCatalog(models=[ModelAsset("reference", "p", "d")], tools=[]),
        objectives=OptimizationObjectives(),
        journal=CampaignJournal(tmp_path / "campaign.jsonl"),
        isolate_evaluations=False,
    )
    with pytest.raises(ValueError, match="no successful reference traces"):
        campaign.run()


def test_isolated_evaluator_roundtrips_serializable_agent():
    evaluator = ModelEvaluator({"reference": EvaluationMetrics(quality=1.0, cost=1.0, latency_ms=10)})
    isolated = IsolatedHarnessEvaluator(evaluator, timeout_seconds=10)
    metrics = isolated.evaluate(Agent(chat_generator=MockChatGenerator(model="reference")), [reference_trace()])
    assert metrics == EvaluationMetrics(quality=1.0, cost=1.0, latency_ms=10)
