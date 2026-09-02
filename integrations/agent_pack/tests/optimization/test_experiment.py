from datetime import datetime, timezone

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool

from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    EvaluationMetrics,
    ExperimentJournal,
    HarnessOptimizationExperiment,
    ModelAsset,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.tracing import (
    LocalTraceStore,
    TraceArtifact,
)


@tool
def remote_tool(query: str) -> str:
    """A tool the reference harness exposes."""
    return query


def reference_trace(run_id="reference-run"):
    now = datetime.now(tz=timezone.utc).isoformat()
    return TraceArtifact(
        run_id=run_id,
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
    """Scores a candidate purely from its configured model, so experiment logic can be tested deterministically."""

    def __init__(self, metrics_by_model, *, failing_models=()):
        self.metrics_by_model = metrics_by_model
        self.failing_models = set(failing_models)
        self.calls = []

    def evaluate(self, agent, reference_traces, assets):
        assert reference_traces[0].run_id.startswith("reference-run")
        assert isinstance(assets, ApprovedAssetCatalog)
        model = agent.chat_generator.model
        self.calls.append(model)
        if model in self.failing_models:
            message = f"provider unavailable for {model}"
            raise RuntimeError(message)
        return self.metrics_by_model[model]


def build_experiment(tmp_path, evaluator, *, objectives=None, tools=None, tool_assets=None, journal=None):
    store = LocalTraceStore()
    store.add(reference_trace())
    reference = Agent(chat_generator=MockChatGenerator(model="reference"), tools=tools)
    assets = ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference", provider="closed", deployment="remote", input_cost_per_million=10),
            ModelAsset(model_id="cheap", provider="local", deployment="eu", input_cost_per_million=2),
            ModelAsset(model_id="bad", provider="local", deployment="eu", input_cost_per_million=1),
        ],
        tools=tool_assets or [],
    )
    experiment = HarnessOptimizationExperiment(
        reference=reference,
        trace_source=store,
        evaluator=evaluator,
        assets=assets,
        objectives=objectives or OptimizationObjectives(min_quality=0.8),
        journal=journal or ExperimentJournal(path=tmp_path / "experiment.jsonl"),
    )
    return experiment, assets, reference


def default_metrics():
    return {
        "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
        "cheap": EvaluationMetrics(quality=1.0, cost=2.0, latency_ms=90),
        "bad": EvaluationMetrics(quality=0.5, cost=1.0, latency_ms=50),
    }


def test_experiment_applies_gates_then_recommends_cheaper_candidate(tmp_path):
    evaluator = ModelEvaluator(default_metrics())
    experiment, assets, reference = build_experiment(tmp_path, evaluator)

    result = experiment.run()

    assert result.recommendation is not None
    assert result.recommendation.evaluation.metrics.cost == 2.0
    assert result.recommendation.reasons == ("single_sample", "cost_improvement")
    approved = result.recommendation.materialize(reference, assets)
    assert approved.chat_generator.model == "cheap"
    assert reference.chat_generator.model == "reference"

    failures = {
        candidate.recipe["model_id"]: result.gate_failures[candidate.candidate_id] for candidate in result.candidates
    }
    assert failures["cheap"] == ()
    assert failures["bad"] == ("quality_below_floor:1.0000",)


def test_gates_are_recomputed_rather_than_replayed_from_the_journal(tmp_path):
    """Tightening the quality floor must re-rank journaled measurements instead of reusing a stale verdict."""
    journal = ExperimentJournal(path=tmp_path / "experiment.jsonl")
    metrics = {
        "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
        "cheap": EvaluationMetrics(quality=0.9, cost=2.0, latency_ms=90),
        "bad": EvaluationMetrics(quality=0.5, cost=1.0, latency_ms=50),
    }
    lenient, _, _ = build_experiment(
        tmp_path,
        ModelEvaluator(metrics),
        objectives=OptimizationObjectives(min_quality=0.8, max_quality_loss=0.2),
        journal=journal,
    )
    assert lenient.run().recommendation is not None

    strict_evaluator = ModelEvaluator(metrics)
    strict, _, _ = build_experiment(
        tmp_path,
        strict_evaluator,
        objectives=OptimizationObjectives(min_quality=1.0),
        journal=journal,
    )
    result = strict.run()
    assert result.recommendation is None
    # Objectives are part of the candidate identity, so the candidates are measured again under the new gates.
    assert sorted(strict_evaluator.calls) == ["bad", "cheap", "reference"]


def test_experiment_resumes_journaled_candidates(tmp_path):
    """A resumed experiment with nothing left to measure costs nothing, baseline included."""
    evaluator = ModelEvaluator(default_metrics())
    experiment, _, _ = build_experiment(tmp_path, evaluator)
    first = experiment.run()
    second = experiment.run()

    # Approved models are proposed in sorted order, and nothing is measured twice.
    assert evaluator.calls == ["reference", "bad", "cheap"]
    assert first.configuration_hash == second.configuration_hash
    assert first.baseline == second.baseline
    assert first.recommendation is not None
    assert second.recommendation is not None


def test_a_changed_evaluation_set_invalidates_journaled_candidates(tmp_path):
    class FingerprintedEvaluator(ModelEvaluator):
        def __init__(self, metrics, version):
            super().__init__(metrics)
            self.version = version

        def fingerprint(self):
            return {"version": self.version}

    journal = ExperimentJournal(path=tmp_path / "experiment.jsonl")
    first, _, _ = build_experiment(tmp_path, FingerprintedEvaluator(default_metrics(), "v1"), journal=journal)
    first.run()

    second_evaluator = FingerprintedEvaluator(default_metrics(), "v2")
    second, _, _ = build_experiment(tmp_path, second_evaluator, journal=journal)
    second.run()
    assert sorted(second_evaluator.calls) == ["bad", "cheap", "reference"]


def test_failed_candidates_are_retried_on_resume(tmp_path):
    """A transient provider failure must not be journaled as a permanent property of the candidate."""
    journal = ExperimentJournal(path=tmp_path / "experiment.jsonl")
    failing = ModelEvaluator(default_metrics(), failing_models=["cheap"])
    experiment, _, _ = build_experiment(tmp_path, failing, journal=journal)
    result = experiment.run()

    failure = next(c for c in result.candidates if c.recipe["model_id"] == "cheap")
    assert failure.failure is not None
    assert "provider unavailable for cheap" in failure.failure
    assert result.gate_failures[failure.candidate_id] == ("evaluation_failed",)
    assert result.recommendation is None

    recovered = ModelEvaluator(default_metrics())
    retry, _, _ = build_experiment(tmp_path, recovered, journal=ExperimentJournal(path=tmp_path / "experiment.jsonl"))
    retried = retry.run()
    assert "cheap" in recovered.calls
    assert retried.recommendation is not None


def test_quality_lower_bound_is_what_gates_compare(tmp_path):
    """A noisy candidate whose mean clears the floor but whose lower bound does not is not recommended."""
    evaluator = ModelEvaluator(
        {
            "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100, quality_lower_bound=1.0),
            "cheap": EvaluationMetrics(quality=1.0, cost=2.0, latency_ms=90, quality_lower_bound=0.5),
            "bad": EvaluationMetrics(quality=1.0, cost=1.0, latency_ms=50, quality_lower_bound=1.0),
        }
    )
    experiment, _, _ = build_experiment(tmp_path, evaluator, objectives=OptimizationObjectives(min_quality=0.9))

    result = experiment.run()

    assert result.recommendation is not None
    assert result.recommendation.evaluation.recipe["model_id"] == "bad"
    cheap = next(c for c in result.candidates if c.recipe["model_id"] == "cheap")
    # The floor is the reference's own lower bound, so pessimistic estimates are compared with each other.
    assert result.gate_failures[cheap.candidate_id] == ("quality_below_floor:1.0000",)


def test_unvalidated_quality_is_reported_on_the_recommendation(tmp_path):
    evaluator = ModelEvaluator(
        {
            "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
            "cheap": EvaluationMetrics(quality=1.0, cost=2.0, latency_ms=90, details={"validated": False}),
            "bad": EvaluationMetrics(quality=0.5, cost=1.0, latency_ms=50),
        }
    )
    experiment, _, _ = build_experiment(tmp_path, evaluator)
    recommendation = experiment.run().recommendation
    assert recommendation is not None
    assert recommendation.reasons == ("quality_unvalidated", "single_sample", "cost_improvement")


def test_experiment_requires_replayable_successful_traces(tmp_path):
    store = LocalTraceStore()
    evaluator = ModelEvaluator({})
    experiment = HarnessOptimizationExperiment(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        trace_source=store,
        evaluator=evaluator,
        assets=ApprovedAssetCatalog(models=[ModelAsset(model_id="reference", provider="p", deployment="d")], tools=[]),
        objectives=OptimizationObjectives(),
        journal=ExperimentJournal(path=tmp_path / "experiment.jsonl"),
    )
    with pytest.raises(ValueError, match="no successful reference traces"):
        experiment.run()

    unreplayable = reference_trace()
    store.add(TraceArtifact(**{**unreplayable.__dict__, "traces": ({"operation_name": "other", "tags": {}},)}))
    with pytest.raises(ValueError, match="cannot be replayed"):
        experiment.run()


def test_configuration_key_invalidates_results_the_experiment_cannot_see(tmp_path):
    journal = ExperimentJournal(path=tmp_path / "experiment.jsonl")
    store = LocalTraceStore()
    store.add(reference_trace())
    assets = ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference", provider="p", deployment="d"),
            ModelAsset(model_id="cheap", provider="p", deployment="d", input_cost_per_million=1),
        ],
        tools=[],
    )

    def experiment_for(key, evaluator):
        return HarnessOptimizationExperiment(
            reference=Agent(chat_generator=MockChatGenerator(model="reference")),
            trace_source=store,
            evaluator=evaluator,
            assets=assets,
            objectives=OptimizationObjectives(),
            journal=journal,
            configuration_key=key,
        )

    metrics = {
        "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=10),
        "cheap": EvaluationMetrics(quality=1.0, cost=1.0, latency_ms=10),
    }
    experiment_for("corpus-v1", ModelEvaluator(metrics)).run()
    second = ModelEvaluator(metrics)
    experiment_for("corpus-v2", second).run()
    assert "cheap" in second.calls
