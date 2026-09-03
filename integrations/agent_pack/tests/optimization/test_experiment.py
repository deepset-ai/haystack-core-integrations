import json
from collections import deque

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics, ModelTokenUsage
from haystack_integrations.agent_pack.local_run_store import LocalRunStore
from haystack_integrations.agent_pack.optimization import (
    AgentMutation,
    ExperimentJournal,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    MutationOperation,
    OptimizationObjectives,
    apply_mutation,
    create_harness_optimizer_agent,
    rebuild_agent,
)

MODEL_PATH = "/init_parameters/chat_generator/init_parameters/model"
STEPS_PATH = "/init_parameters/max_agent_steps"


def reference_run(question="question"):
    """Create one replayable successful reference run."""
    return AgentRunRecord(
        run_id="reference-run",
        inputs={"messages": [ChatMessage.from_user(question)]},
        outputs={"last_message": ChatMessage.from_assistant("answer")},
    )


def set_value(path, value):
    """Build one scalar Agent configuration mutation."""
    return AgentMutation(operations=(MutationOperation(op="set", path=path, value=value),))


def optimizer_agent_for(mutations):
    """Build a deterministic optimizer Agent and expose every history it observes."""
    remaining = deque(mutations)
    histories = []

    def respond(messages):
        """Record optimizer context and return the next structured decision."""
        request = json.loads(messages[-1].text)
        histories.append(request["history"])
        mutation = remaining.popleft() if remaining else None
        return json.dumps({"mutation": mutation.model_dump() if mutation is not None else None})

    agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=respond))
    return agent, histories


class ModelEvaluator:
    """Return configured metrics based on the candidate coordinator model."""

    def __init__(self, metrics_by_model, failing=()):
        """Configure model outcomes and optional runtime failures."""
        self.metrics_by_model = metrics_by_model
        self.failing = set(failing)
        self.calls = []

    def evaluate(self, agent, reference_runs):
        """Measure a candidate by looking up its model."""
        assert reference_runs[0].run_id == "reference-run"
        model = agent.chat_generator.model
        self.calls.append(model)
        if model in self.failing:
            msg = f"provider unavailable for {model}"
            raise RuntimeError(msg)
        return self.metrics_by_model[model]

    def fingerprint(self):
        """Return stable evaluator configuration for journaling."""
        return {"kind": "model-evaluator"}


def fixed_metrics():
    """Return baseline, quality-regressing, and improving model measurements."""
    return {
        "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
        "cheap": EvaluationMetrics(quality=1.0, cost=2.0, latency_ms=90),
        "bad": EvaluationMetrics(quality=0.5, cost=1.0, latency_ms=50),
        "unknown": EvaluationMetrics(
            quality=1.0,
            latency_ms=80,
            model_usage={"unknown": ModelTokenUsage(input_tokens=100)},
        ),
    }


def pricing(cheap_price=2.0):
    """Create known prices without constraining candidate model identifiers."""
    return ModelPriceCatalog(
        prices=[
            ModelPrice(model_id="reference", input_cost_per_million=10.0),
            ModelPrice(model_id="cheap", input_cost_per_million=cheap_price),
            ModelPrice(model_id="bad", input_cost_per_million=1.0),
        ]
    )


def experiment(tmp_path, evaluator, optimizer_agent, pricing_context=None, objectives=None, store=None, journal=None):
    """Build an experiment around a serializable reference Agent."""
    store = store or LocalRunStore()
    if not store.list():
        store.add(record=reference_run())
    return HarnessOptimizationExperiment(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        run_store=store,
        evaluator=evaluator,
        pricing=pricing_context or pricing(),
        objectives=objectives or OptimizationObjectives(min_quality=0.8),
        journal=journal or ExperimentJournal(path=tmp_path / "experiment.jsonl"),
        optimizer_agent=optimizer_agent,
    )


def test_optimizer_learns_from_each_choice_before_making_the_next(tmp_path):
    """Every candidate's measured outcome is supplied to the following optimizer turn."""
    optimizer_agent, histories = optimizer_agent_for(
        mutations=[set_value(path=MODEL_PATH, value="bad"), set_value(path=MODEL_PATH, value="cheap"), None]
    )
    result = experiment(
        tmp_path=tmp_path,
        evaluator=ModelEvaluator(metrics_by_model=fixed_metrics()),
        optimizer_agent=optimizer_agent,
    ).run()

    assert histories[0] == []
    assert histories[1][0]["mutation"]["operations"][0]["value"] == "bad"
    assert histories[1][0]["gate_failures"] == ["quality_below_floor:1.0000"]
    assert histories[2][1]["metrics"]["cost"] == 2.0
    assert result.recommendation is not None
    assert result.recommendation.mutation == set_value(path=MODEL_PATH, value="cheap")


def test_completed_measurements_seed_history_on_resume(tmp_path):
    """A resumed search learns from journaled configurations without remeasuring them."""
    journal = ExperimentJournal(path=tmp_path / "experiment.jsonl")
    initial_agent, _ = optimizer_agent_for(mutations=[set_value(path=MODEL_PATH, value="cheap"), None])
    experiment(
        tmp_path=tmp_path,
        evaluator=ModelEvaluator(metrics_by_model=fixed_metrics()),
        optimizer_agent=initial_agent,
        journal=journal,
    ).run()

    resumed_evaluator = ModelEvaluator(metrics_by_model=fixed_metrics())
    resumed_agent, resumed_histories = optimizer_agent_for(mutations=[None])
    result = experiment(
        tmp_path=tmp_path,
        evaluator=resumed_evaluator,
        optimizer_agent=resumed_agent,
        journal=journal,
    ).run()
    assert resumed_evaluator.calls == []
    assert resumed_histories[0][0]["mutation"]["operations"][0]["value"] == "cheap"
    assert result.recommendation is not None


def test_resulting_full_configuration_defines_candidate_identity(tmp_path):
    """Different multi-operation decisions that produce one configuration are measured only once."""
    same_twice = AgentMutation(
        operations=(
            MutationOperation(op="set", path=MODEL_PATH, value="cheap"),
            MutationOperation(op="set", path=MODEL_PATH, value="cheap"),
        )
    )
    optimizer_agent, histories = optimizer_agent_for(
        mutations=[set_value(path=MODEL_PATH, value="cheap"), same_twice, None]
    )
    evaluator = ModelEvaluator(metrics_by_model=fixed_metrics())
    result = experiment(tmp_path=tmp_path, evaluator=evaluator, optimizer_agent=optimizer_agent).run()
    assert evaluator.calls == ["reference", "cheap"]
    assert len(result.candidates) == 1
    assert histories[2][-1]["reason"] == "duplicate_or_no_op"


def test_invalid_configuration_is_an_outcome_the_optimizer_can_learn_from(tmp_path):
    """Haystack deserialization failures feed the next decision instead of aborting the experiment."""
    invalid = set_value(path="/init_parameters/unsupported_parameter", value=True)
    optimizer_agent, histories = optimizer_agent_for(
        mutations=[invalid, set_value(path=MODEL_PATH, value="cheap"), None]
    )
    result = experiment(
        tmp_path=tmp_path,
        evaluator=ModelEvaluator(metrics_by_model=fixed_metrics()),
        optimizer_agent=optimizer_agent,
    ).run()
    assert result.candidates[0].metrics is None
    assert "could not be rebuilt" in (result.candidates[0].failure or "")
    assert histories[1][0]["status"] == "failed"
    assert result.recommendation is not None


def test_evaluation_failure_is_journaled_and_supplied_to_next_turn(tmp_path):
    """Provider/runtime failures also become optimizer evidence."""
    optimizer_agent, histories = optimizer_agent_for(
        mutations=[set_value(path=MODEL_PATH, value="bad"), set_value(path=MODEL_PATH, value="cheap"), None]
    )
    result = experiment(
        tmp_path=tmp_path,
        evaluator=ModelEvaluator(metrics_by_model=fixed_metrics(), failing=("bad",)),
        optimizer_agent=optimizer_agent,
    ).run()
    assert result.candidates[0].failure == "RuntimeError: provider unavailable for bad"
    assert histories[1][0]["gate_failures"] == ["evaluation_failed"]


def test_unknown_model_can_run_but_cannot_win_a_cost_objective(tmp_path):
    """Lack of pricing is a ranking limitation rather than an edit authorization failure."""
    optimizer_agent, _ = optimizer_agent_for(mutations=[set_value(path=MODEL_PATH, value="unknown"), None])
    result = experiment(
        tmp_path=tmp_path,
        evaluator=ModelEvaluator(metrics_by_model=fixed_metrics()),
        optimizer_agent=optimizer_agent,
    ).run()
    assert result.candidates[0].metrics is not None
    assert result.candidates[0].metrics.details["unpriced_models"] == ["unknown"]
    assert result.gate_failures[result.candidates[0].candidate_id] == ("cost_unavailable",)


def test_current_prices_rerank_journaled_raw_usage_without_remeasurement(tmp_path):
    """Changing informational prices does not invalidate expensive raw measurements."""
    raw = {
        "reference": EvaluationMetrics(
            quality=1.0,
            latency_ms=100,
            model_usage={"reference": ModelTokenUsage(input_tokens=1_000_000)},
        ),
        "cheap": EvaluationMetrics(
            quality=1.0,
            latency_ms=90,
            model_usage={"cheap": ModelTokenUsage(input_tokens=1_000_000)},
        ),
    }
    journal = ExperimentJournal(path=tmp_path / "experiment.jsonl")
    initial_agent, _ = optimizer_agent_for(mutations=[set_value(path=MODEL_PATH, value="cheap"), None])
    experiment(
        tmp_path=tmp_path,
        evaluator=ModelEvaluator(metrics_by_model=raw),
        optimizer_agent=initial_agent,
        pricing_context=pricing(cheap_price=2.0),
        journal=journal,
    ).run()
    resumed = ModelEvaluator(metrics_by_model=raw)
    resumed_agent, _ = optimizer_agent_for(mutations=[None])
    result = experiment(
        tmp_path=tmp_path,
        evaluator=resumed,
        optimizer_agent=resumed_agent,
        pricing_context=pricing(cheap_price=20.0),
        journal=journal,
    ).run()
    assert resumed.calls == []
    assert result.recommendation is None


def test_recommendation_can_be_rebuilt_from_its_mutation_and_the_reference(tmp_path):
    """A recommendation needs only its complete mutation and the unchanged reference, without a patch catalog."""
    optimizer_agent, _ = optimizer_agent_for(mutations=[set_value(path=MODEL_PATH, value="cheap"), None])
    configured = experiment(
        tmp_path=tmp_path,
        evaluator=ModelEvaluator(metrics_by_model=fixed_metrics()),
        optimizer_agent=optimizer_agent,
    )
    result = configured.run()
    assert result.recommendation is not None
    reference = configured.reference
    candidate = rebuild_agent(
        serialized_agent=apply_mutation(serialized_agent=reference.to_dict(), mutation=result.recommendation.mutation)
    )
    assert candidate.chat_generator.model == "cheap"
    assert reference.chat_generator.model == "reference"


def test_empty_run_store_is_rejected(tmp_path):
    """Optimization requires at least one successful input/output example."""
    optimizer_agent, _ = optimizer_agent_for(mutations=[None])
    configured = HarnessOptimizationExperiment(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        run_store=LocalRunStore(),
        evaluator=ModelEvaluator(metrics_by_model=fixed_metrics()),
        pricing=pricing(),
        objectives=OptimizationObjectives(),
        journal=ExperimentJournal(path=tmp_path / "experiment.jsonl"),
        optimizer_agent=optimizer_agent,
    )
    with pytest.raises(ValueError, match="no successful reference runs"):
        configured.run()
