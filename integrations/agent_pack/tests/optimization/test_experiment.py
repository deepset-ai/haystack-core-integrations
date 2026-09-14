import json
from collections import deque
from dataclasses import dataclass

import pytest
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import AdvancedRAGHarnessEvaluator
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    load_agent,
    load_pipeline,
)
from haystack_integrations.agent_pack.optimization.experiment import _fingerprint_eval_cases
from haystack_integrations.evaluation.dataclasses import EvalMetrics, ModelTokenUsage


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


@dataclass
class StubEvalCase:
    """The least an eval case has to be for the experiment to fingerprint the set."""

    question: str

    def to_dict(self):
        return {"question": self.question}


class ModelEvaluator:
    def __init__(self, metrics=None, failing=()):
        self.metrics = metrics or {
            "reference": EvalMetrics(quality=1, cost=10, latency_ms=100),
            "cheap": EvalMetrics(quality=1, cost=2, latency_ms=90),
            "bad": EvalMetrics(quality=0.5, cost=1, latency_ms=50),
        }
        self.failing = failing
        self.calls = []

    def evaluate(self, target, eval_cases):
        assert eval_cases
        model = target.chat_generator.model
        self.calls.append(model)
        if model in self.failing:
            msg = f"provider unavailable for {model}"
            raise RuntimeError(msg)
        return self.metrics[model]


def configured(tmp_path, models, evaluator=None, objectives=None, eval_cases=None):
    optimizer, histories = optimizer_agent_for(models)
    experiment = HarnessOptimizationExperiment(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        eval_cases=eval_cases or [StubEvalCase("a question")],
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


def unmeasurable(details):
    """A candidate whose usage cannot be accounted for, for the stated reason."""
    return ModelEvaluator(
        metrics={
            "reference": EvalMetrics(quality=1, cost=10, latency_ms=100),
            "unknown": EvalMetrics(
                quality=1, latency_ms=90, details=details, model_usage={"unknown": ModelTokenUsage(input_tokens=10)}
            ),
        }
    )


class TestRun:
    def test_validates_before_measuring(self, tmp_path):
        """The experiment finds this hook by name, so a validator it does not find is silently never run."""

        class RejectingEvaluator(ModelEvaluator):
            def __init__(self):
                super().__init__()
                self.seen = []

            def validate(self, target: Agent) -> None:
                self.seen.append(target.chat_generator.model)
                msg = "this evaluator cannot measure that configuration"
                raise ValueError(msg)

        evaluator = RejectingEvaluator()
        experiment, _ = configured(tmp_path, ["cheap", None], evaluator=evaluator)
        result = experiment.run()
        # Reached during the optimizer's validate_config step, so the candidate never costs a measurement.
        assert evaluator.seen == ["cheap"]
        assert result.recommendation is None

    def test_recommendation_is_loadable(self, tmp_path):
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

    def test_runtime_failure_reaches_the_next_turn(self, tmp_path):
        experiment, histories = configured(tmp_path, ["bad", "cheap", None], ModelEvaluator(failing=("bad",)))
        result = experiment.run()
        assert result.candidates[0].failure == "RuntimeError: provider unavailable for bad"
        assert "evaluation_failed" in histories[1]
        rows = [json.loads(line) for line in experiment.journal.path_for(result.run_id).read_text().splitlines()]
        assert rows[1]["failure"] == result.candidates[0].failure

    def test_supplied_draft_is_left_alone(self, tmp_path):
        """A file the caller pointed at is theirs, not an artifact the experiment cleans up."""
        draft = tmp_path / "mine.yaml"
        experiment, _ = configured(tmp_path, ["cheap", None])
        experiment.config_path = draft
        experiment.run()
        assert draft.exists()

    def test_no_eval_cases_raises_error(self):
        """Quality is a fraction over the eval cases, so an empty set has to be refused before it is computed."""
        with pytest.raises(ValueError, match="no eval cases to score"):
            AdvancedRAGHarnessEvaluator().evaluate(target=Agent(chat_generator=MockChatGenerator()), eval_cases=[])

    def test_iteration_budget_counts_evaluations(self, tmp_path):
        evaluator = ModelEvaluator()
        experiment, _ = configured(tmp_path, ["bad", "cheap", None], evaluator)
        experiment.max_iterations = 1
        result = experiment.run()
        assert evaluator.calls == ["reference", "bad"]
        assert len(result.candidates) == 1

    def test_optimizes_a_plain_pipeline(self, tmp_path):
        """
        An Agent is serialized by wrapping it in a one-component Pipeline; a Pipeline is already one. The experiment
        picks the pair to use from the reference, so a candidate comes back as the same kind of thing it started as.
        """
        reference = Pipeline()
        reference.add_component("generator", MockChatGenerator(model="reference"))

        class PipelineEvaluator:
            def __init__(self):
                self.measured = []

            def evaluate(self, target, eval_cases):
                assert eval_cases
                # A Pipeline reference must arrive as a Pipeline, not wrapped in an Agent.
                assert isinstance(target, Pipeline)
                model = target.get_component("generator").model
                self.measured.append(model)
                return EvalMetrics(quality=1.0 if model == "cheap" else 0.5, cost=1, latency_ms=1)

        # A scripted optimizer of its own: the shared one reads the model back through `load_agent`, which by
        # design refuses anything that is not a one-component Agent wrapper.
        stage = 0

        def respond(_messages, tools):
            nonlocal stage
            current = next(tool for tool in tools if tool.name == "read_config").function()
            call = [
                ToolCall(
                    "edit_config",
                    {"old": "model: reference", "new": "model: cheap", "expected_revision": current["revision"]},
                    id="edit",
                ),
                ToolCall("validate_config", {}, id="validate"),
                ToolCall(
                    "submit_candidate",
                    {"expected_revision": current["revision"], "rationale": "cheaper model"},
                    id="submit",
                ),
            ][stage]
            stage += 1
            return ChatMessage.from_assistant(tool_calls=[call])

        optimizer = create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=respond))
        evaluator = PipelineEvaluator()
        result = HarnessOptimizationExperiment(
            reference=reference,
            eval_cases=[StubEvalCase("a question")],
            evaluator=evaluator,
            pricing=ModelPriceCatalog([]),
            # Ranked on quality: both configurations cost the same, so only the better answer can win.
            objectives=OptimizationObjectives(primary="quality", min_quality=0.0),
            journal=ExperimentJournal(tmp_path / "journals"),
            optimizer_agent=optimizer,
            max_iterations=1,
        ).run()
        assert evaluator.measured == ["reference", "cheap"]
        assert result.recommendation is not None
        approved = load_pipeline(result.recommendation.configuration.yaml)
        assert isinstance(approved, Pipeline)
        assert approved.get_component("generator").model == "cheap"
        # The reference itself is never edited in place.
        assert reference.get_component("generator").model == "reference"

    def test_records_why_it_stopped_early(self, tmp_path):
        """Ending the search is the one decision an experiment cannot revisit and leaves no artifact of its own."""
        experiment, _ = configured(tmp_path, ["cheap", None], objectives=OptimizationObjectives(min_quality=0.8))
        result = experiment.run()
        # Two of the eight allowed evaluations were used; the rest were given up deliberately.
        assert len(result.candidates) == 1
        assert experiment.max_iterations > len(result.candidates)

    def test_reports_what_the_search_itself_spent(self, tmp_path):
        """An experiment prices the configurations it measures; the optimizer's own calls are the other half."""
        experiment, _ = configured(tmp_path, ["cheap", None])
        result = experiment.run()
        # Four scripted model calls across two turns: edit, validate, submit, then finish.
        assert result.optimizer_usage["optimizer"] == ModelTokenUsage(input_tokens=80, output_tokens=16)
        assert result.optimizer_cost == pytest.approx(80 * 5 / 1_000_000)
        context = json.loads((result.artifact_directory / "context.json").read_text())
        assert context["optimizer_cost"] == result.optimizer_cost


class TestGating:
    def test_unpriced_model_cannot_win_on_cost(self, tmp_path):
        experiment, _ = configured(tmp_path, ["unknown", None], unmeasurable({}))
        result = experiment.run()
        assert result.recommendation is None
        assert result.gate_failures[result.candidates[0].candidate_id] == ("cost_unavailable",)

    def test_incomplete_usage_cannot_win(self, tmp_path):
        """Usage a harness could not account for is what a silently swallowed component failure looks like."""
        for primary in ("cost", "quality", "latency"):
            experiment, _ = configured(
                tmp_path,
                ["unknown", None],
                unmeasurable({"all_tokens_reported": False}),
                objectives=OptimizationObjectives(min_quality=0.8, primary=primary),
            )
            result = experiment.run()
            assert result.recommendation is None
            assert result.gate_failures[result.candidates[0].candidate_id] == ("usage_incomplete",)

    def test_quality_objective_prefers_better_answers(self, tmp_path):
        evaluator = ModelEvaluator(
            metrics={
                "reference": EvalMetrics(quality=0.3, cost=10, latency_ms=100),
                "cheap": EvalMetrics(quality=0.4, cost=1, latency_ms=50),
                "better": EvalMetrics(quality=1, cost=8, latency_ms=90),
            }
        )
        experiment, _ = configured(
            tmp_path, ["cheap", "better", None], evaluator, OptimizationObjectives(primary="quality")
        )
        assert load_agent(experiment.run().recommendation.configuration.yaml).chat_generator.model == "better"

    def test_candidate_on_the_tolerance(self, tmp_path):
        """A tolerance of one eval case in twenty is 0.05, and 0.2 - 0.05 is 0.15000000000000002 in binary float."""
        experiment, _ = configured(
            tmp_path,
            ["cheap", None],
            evaluator=ModelEvaluator(
                metrics={
                    "reference": EvalMetrics(quality=4 / 20, cost=10, latency_ms=100),
                    "cheap": EvalMetrics(quality=3 / 20, cost=2, latency_ms=90),
                }
            ),
            objectives=OptimizationObjectives(min_quality=0.0, max_quality_loss=0.05, primary="quality"),
        )
        result = experiment.run()
        assert result.gate_failures[result.candidates[0].candidate_id] == ()

    def test_regression_is_not_inherited(self, tmp_path):
        """The experiment hands each turn the best candidate so far, so a bad branch is not built on."""
        evaluator = ModelEvaluator(
            metrics={
                "reference": EvalMetrics(quality=0.5, cost=10, latency_ms=100),
                "good": EvalMetrics(quality=0.9, cost=5, latency_ms=90),
                "worse": EvalMetrics(quality=0.6, cost=5, latency_ms=90),
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


class TestJournal:
    def test_records_raw_usage_before_pricing(self, tmp_path):
        evaluator = ModelEvaluator(
            metrics={
                name: EvalMetrics(
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

    def test_runs_are_numbered_in_order(self, tmp_path):
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

    def test_repeated_experiments_stay_separate(self, tmp_path):
        first, _ = configured(tmp_path, [None])
        second, _ = configured(tmp_path, [None])
        a, b = first.run(), second.run()
        assert a.measurement_context == b.measurement_context
        assert a.run_id != b.run_id
        assert first.journal.path_for(a.run_id).exists()
        assert second.journal.path_for(b.run_id).exists()


class TestMeasurementContext:
    def test_same_inputs_share_a_context(self, tmp_path):
        """The context answers whether two runs' numbers compare, which a changed reference must not break."""
        first, _ = configured(tmp_path, ["cheap", None])
        second, _ = configured(tmp_path, ["cheap", None])
        # A different reference configuration, measured against the same runs, eval cases and evaluator.
        second.reference = Agent(chat_generator=MockChatGenerator(model="reference"), max_agent_steps=7)
        one = first.run()
        two = second.run()
        assert one.measurement_context == two.measurement_context
        # The configurations themselves are still distinguishable, and still recorded.
        assert (one.artifact_directory / "reference.yaml").read_text() != (
            two.artifact_directory / "reference.yaml"
        ).read_text()

    def test_fingerprint_ignores_eval_case_order(self):
        """Two callers listing the same questions differently are measuring the same thing."""
        one = [StubEvalCase("zeta"), StubEvalCase("alpha")]
        other = [StubEvalCase("alpha"), StubEvalCase("zeta")]
        assert _fingerprint_eval_cases(eval_cases=one) == _fingerprint_eval_cases(eval_cases=other)
        assert [entry["question"] for entry in _fingerprint_eval_cases(eval_cases=one)] == ["alpha", "zeta"]
        assert _fingerprint_eval_cases(eval_cases=one) != _fingerprint_eval_cases(eval_cases=[StubEvalCase("alpha")])

    def test_a_different_eval_set_is_not_comparable(self, tmp_path):
        """Measurements only compare when they were taken against the same questions."""
        first, _ = configured(tmp_path, ["cheap", None])
        second, _ = configured(tmp_path, ["cheap", None], eval_cases=[StubEvalCase("a different question")])
        assert first.run().measurement_context != second.run().measurement_context
