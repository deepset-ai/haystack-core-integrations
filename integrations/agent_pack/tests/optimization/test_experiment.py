import json
from collections import deque

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics, ModelTokenUsage
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    load_agent,
)
from haystack_integrations.agent_pack.optimization.local_run_store import LocalRunStore


def optimizer_agent_for(models):
    """Script real edit, validate and submit tool calls for a sequence of model changes."""
    remaining = deque(models)
    histories = []
    stage = 0

    def respond(messages, tools):
        nonlocal stage
        read = next(tool for tool in tools if tool.name == "read_config")
        current = read.function()
        if stage == 0:
            histories.append([m.text for m in messages if m.is_from("user")][-2])
            model = remaining.popleft() if remaining else None
            if model is None:
                return ChatMessage.from_assistant(
                    tool_calls=[ToolCall("finish", {"reason": "nothing left worth measuring"}, id="finish")]
                )
            original = load_agent(current["yaml"]).chat_generator.model
            call = ToolCall(
                "edit_config",
                {
                    "old": f"model: {original}",
                    "new": f"model: {model}",
                    "expected_revision": current["revision"],
                },
                id="edit",
            )
        elif stage == 1:
            call = ToolCall("validate_config", {}, id="validate")
        else:
            call = ToolCall(
                "submit_candidate",
                {
                    "expected_revision": current["revision"],
                    "rationale": "test model change",
                },
                id="submit",
            )
        stage = (stage + 1) % 3
        return ChatMessage.from_assistant(tool_calls=[call])

    generator = MockChatGenerator(
        response_fn=respond, model="optimizer", meta={"usage": {"prompt_tokens": 20, "completion_tokens": 4}}
    )
    return create_harness_optimizer_agent(chat_generator=generator), histories


class ModelEvaluator:
    def __init__(self, metrics=None, failing=()):
        self.metrics = metrics or {
            "reference": EvaluationMetrics(quality=1, cost=10, latency_ms=100),
            "cheap": EvaluationMetrics(quality=1, cost=2, latency_ms=90),
            "bad": EvaluationMetrics(quality=0.5, cost=1, latency_ms=50),
        }
        self.failing = failing
        self.calls = []

    def evaluate(self, target, reference_runs):
        assert reference_runs
        model = target.chat_generator.model
        self.calls.append(model)
        if model in self.failing:
            msg = f"provider unavailable for {model}"
            raise RuntimeError(msg)
        return self.metrics[model]

    def fingerprint(self):
        return {"kind": "model-evaluator"}


def configured(tmp_path, models, evaluator=None, objectives=None):
    optimizer, histories = optimizer_agent_for(models)
    store = LocalRunStore()
    store.add(
        AgentRunRecord(
            run_id="reference",
            inputs={"messages": [ChatMessage.from_user("question")]},
            outputs={"last_message": ChatMessage.from_assistant("answer")},
        )
    )
    experiment = HarnessOptimizationExperiment(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        run_store=store,
        evaluator=evaluator or ModelEvaluator(),
        pricing=ModelPriceCatalog(
            [
                ModelPrice(model_id="reference", input_cost_per_million=10),
                ModelPrice(model_id="cheap", input_cost_per_million=2),
                ModelPrice(model_id="optimizer", input_cost_per_million=5),
            ]
        ),
        objectives=objectives or OptimizationObjectives(min_quality=0.8),
        journal=ExperimentJournal(tmp_path / "journals"),
        optimizer_agent=optimizer,
    )
    return experiment, histories


def test_optimizer_learns_from_outcomes_and_recommendation_is_loadable(tmp_path):
    experiment, histories = configured(tmp_path, ["bad", "cheap", None])
    result = experiment.run()
    assert histories[0] == ""
    assert "quality_below_floor:1.0000" in histories[1]
    assert "cost $2.0000" in histories[2]
    assert result.recommendation is not None
    assert load_agent(result.recommendation.configuration.yaml).chat_generator.model == "cheap"
    assert experiment.reference.chat_generator.model == "reference"
    assert result.candidates[1].configuration.parent_id == result.candidates[0].candidate_id
    assert (result.artifact_directory / "recommended.yaml").read_text() == result.recommendation.configuration.yaml


def test_runtime_failure_is_journaled_and_supplied_to_next_turn(tmp_path):
    experiment, histories = configured(tmp_path, ["bad", "cheap", None], ModelEvaluator(failing=("bad",)))
    result = experiment.run()
    assert result.candidates[0].failure == "RuntimeError: provider unavailable for bad"
    assert "evaluation_failed" in histories[1]
    rows = [json.loads(line) for line in experiment.journal.path_for(result.run_id).read_text().splitlines()]
    assert rows[1]["failure"] == result.candidates[0].failure


def test_raw_usage_is_journaled_before_pricing(tmp_path):
    evaluator = ModelEvaluator(
        metrics={
            name: EvaluationMetrics(
                quality=1, latency_ms=100, model_usage={name: ModelTokenUsage(input_tokens=1_000_000)}
            )
            for name in ("reference", "cheap")
        }
    )
    experiment, _ = configured(tmp_path, ["cheap", None], evaluator)
    result = experiment.run()
    rows = [json.loads(line) for line in experiment.journal.path_for(result.run_id).read_text().splitlines()]
    assert rows[1]["metrics"]["cost"] is None
    assert result.recommendation.evaluation.metrics.cost == 2


def unmeasurable(details):
    """A candidate whose usage cannot be accounted for, for the stated reason."""
    return ModelEvaluator(
        metrics={
            "reference": EvaluationMetrics(quality=1, cost=10, latency_ms=100),
            "unknown": EvaluationMetrics(
                quality=1, latency_ms=90, details=details, model_usage={"unknown": ModelTokenUsage(input_tokens=10)}
            ),
        }
    )


def test_an_unpriced_model_cannot_win_on_cost(tmp_path):
    experiment, _ = configured(tmp_path, ["unknown", None], unmeasurable({}))

    result = experiment.run()

    assert result.recommendation is None
    assert result.gate_failures[result.candidates[0].candidate_id] == ("cost_unavailable",)


def test_incomplete_usage_cannot_win_on_any_objective(tmp_path):
    """Usage a harness could not account for is what a silently swallowed component failure looks like."""
    for primary in ("cost", "quality", "latency"):
        experiment, _ = configured(
            tmp_path,
            ["unknown", None],
            unmeasurable({"usage_complete": False}),
            objectives=OptimizationObjectives(min_quality=0.8, primary=primary),
        )

        result = experiment.run()

        assert result.recommendation is None
        assert result.gate_failures[result.candidates[0].candidate_id] == ("usage_incomplete",)


def test_runs_are_numbered_in_the_order_they_happened(tmp_path):
    """A directory of experiments should read in order, and a second run must not reuse the first one's name."""
    first, _ = configured(tmp_path, ["cheap", None])
    second, _ = configured(tmp_path, ["cheap", None])

    one = first.run()
    two = second.run()

    assert one.run_id == "run-1"
    assert two.run_id == "run-2"
    assert one.artifact_directory.name == "run-1"
    assert first.journal.path_for("run-1").exists()
    # The editable draft is scratch, so a completed run leaves only its record behind.
    assert not (one.artifact_directory / "candidate.yaml").exists()
    assert (one.artifact_directory / "reference.yaml").exists()


def test_a_supplied_draft_is_left_alone(tmp_path):
    """A file the caller pointed at is theirs, not an artifact the experiment cleans up."""
    draft = tmp_path / "mine.yaml"
    experiment, _ = configured(tmp_path, ["cheap", None])
    experiment.config_path = draft

    experiment.run()

    assert draft.exists()


def test_runs_measuring_the_same_thing_share_a_measurement_context(tmp_path):
    """The context answers whether two runs' numbers can be compared, and a changed reference does not break that."""
    first, _ = configured(tmp_path, ["cheap", None])
    second, _ = configured(tmp_path, ["cheap", None])
    # A different reference configuration, measured against the same runs, cases and evaluator.
    second.reference = Agent(chat_generator=MockChatGenerator(model="reference"), max_agent_steps=7)

    one = first.run()
    two = second.run()

    assert one.measurement_context == two.measurement_context
    # The configurations themselves are still distinguishable, and still recorded.
    assert (one.artifact_directory / "reference.yaml").read_text() != (
        two.artifact_directory / "reference.yaml"
    ).read_text()


def test_a_different_evaluation_set_is_not_comparable(tmp_path):
    class OtherCases(ModelEvaluator):
        def fingerprint(self):
            return {"kind": "different-cases"}

    first, _ = configured(tmp_path, ["cheap", None])
    second, _ = configured(tmp_path, ["cheap", None], evaluator=OtherCases())

    assert first.run().measurement_context != second.run().measurement_context


def test_repeated_experiments_have_separate_journals(tmp_path):
    first, _ = configured(tmp_path, [None])
    second, _ = configured(tmp_path, [None])
    a, b = first.run(), second.run()
    assert a.measurement_context == b.measurement_context
    assert a.run_id != b.run_id
    assert first.journal.path_for(a.run_id).exists()
    assert second.journal.path_for(b.run_id).exists()


def test_empty_store_rejected(tmp_path):
    experiment, _ = configured(tmp_path, [None])
    experiment.run_store = LocalRunStore()
    with pytest.raises(ValueError, match="no successful reference runs"):
        experiment.run()


def test_quality_objective_prefers_better_answers(tmp_path):
    evaluator = ModelEvaluator(
        metrics={
            "reference": EvaluationMetrics(quality=0.3, cost=10, latency_ms=100),
            "cheap": EvaluationMetrics(quality=0.4, cost=1, latency_ms=50),
            "better": EvaluationMetrics(quality=1, cost=8, latency_ms=90),
        }
    )
    experiment, _ = configured(
        tmp_path, ["cheap", "better", None], evaluator, OptimizationObjectives(primary="quality")
    )
    assert load_agent(experiment.run().recommendation.configuration.yaml).chat_generator.model == "better"


def test_ending_the_search_early_records_why(tmp_path):
    """Ending the search is the one decision an experiment cannot revisit and leaves no artifact of its own."""
    experiment, _ = configured(tmp_path, ["cheap", None], objectives=OptimizationObjectives(min_quality=0.8))

    result = experiment.run()

    # Two of the eight allowed evaluations were used; the rest were given up deliberately.
    assert len(result.candidates) == 1
    assert experiment.max_iterations > len(result.candidates)


def test_the_search_reports_what_it_spent_on_itself(tmp_path):
    """An experiment prices the configurations it measures; the optimizer's own calls are the other half."""
    experiment, _ = configured(tmp_path, ["cheap", None])

    result = experiment.run()

    # Four scripted model calls across two turns: edit, validate, submit, then finish.
    assert result.optimizer_usage["optimizer"] == ModelTokenUsage(input_tokens=80, output_tokens=16)
    assert result.optimizer_cost == pytest.approx(80 * 5 / 1_000_000)
    context = json.loads((result.artifact_directory / "context.json").read_text())
    assert context["optimizer_cost"] == result.optimizer_cost


def test_a_candidate_exactly_on_the_quality_tolerance_is_not_gated_out(tmp_path):
    """A tolerance of one case in twenty is 0.05, and 0.2 - 0.05 is 0.15000000000000002 in binary floating point."""
    experiment, _ = configured(
        tmp_path,
        ["cheap", None],
        evaluator=ModelEvaluator(
            metrics={
                "reference": EvaluationMetrics(quality=4 / 20, cost=10, latency_ms=100),
                "cheap": EvaluationMetrics(quality=3 / 20, cost=2, latency_ms=90),
            }
        ),
        objectives=OptimizationObjectives(min_quality=0.0, max_quality_loss=0.05, primary="quality"),
    )

    result = experiment.run()

    assert result.gate_failures[result.candidates[0].candidate_id] == ()


def test_a_regression_is_not_inherited_by_the_next_candidate(tmp_path):
    """The experiment hands each turn the best candidate so far, so a bad branch is not built on."""
    evaluator = ModelEvaluator(
        metrics={
            "reference": EvaluationMetrics(quality=0.5, cost=10, latency_ms=100),
            "good": EvaluationMetrics(quality=0.9, cost=5, latency_ms=90),
            "worse": EvaluationMetrics(quality=0.6, cost=5, latency_ms=90),
        }
    )
    experiment, _ = configured(
        tmp_path,
        ["good", "worse", None],
        evaluator=evaluator,
        objectives=OptimizationObjectives(min_quality=0.0, primary="quality"),
    )

    result = experiment.run()

    scored = {c.candidate_id: c.metrics.quality for c in result.candidates}
    best = max(scored, key=lambda cid: scored[cid])
    # The third turn follows the regression, and it is handed the best candidate rather than the regression.
    assert result.candidates[-1].configuration is None or result.candidates[-1].configuration.parent_id == best


def test_iteration_budget_counts_evaluations(tmp_path):
    evaluator = ModelEvaluator()
    experiment, _ = configured(tmp_path, ["bad", "cheap", None], evaluator)
    experiment.max_iterations = 1
    result = experiment.run()
    assert evaluator.calls == ["reference", "bad"]
    assert len(result.candidates) == 1
