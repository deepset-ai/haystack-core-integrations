from collections import deque

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.optimization import (
    ApplyPatchRecipe,
    ApprovedAssetCatalog,
    EvaluationMetrics,
    ExperimentJournal,
    HarnessOptimizationExperiment,
    HarnessPatch,
    ModelAsset,
    ModelSubstitutionRecipe,
    ModelTokenUsage,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.runs import AgentRunRecord, LocalRunStore


def reference_run(question="question"):
    return AgentRunRecord(
        run_id="reference-run",
        inputs={"messages": [ChatMessage.from_user(question)]},
        outputs={"last_message": ChatMessage.from_assistant("answer")},
    )


class SequenceProposer:
    def __init__(self, recipes):
        self.recipes = deque(recipes)
        self.histories = []

    def propose(self, **kwargs):
        self.histories.append(list(kwargs["history"]))
        return self.recipes.popleft() if self.recipes else None


class ModelEvaluator:
    def __init__(self, metrics_by_model, *, failing=()):
        self.metrics_by_model = metrics_by_model
        self.failing = set(failing)
        self.calls = []

    def evaluate(self, agent, reference_runs):
        assert reference_runs[0].run_id == "reference-run"
        model = agent.chat_generator.model
        self.calls.append(model)
        if model in self.failing:
            msg = f"provider unavailable for {model}"
            raise RuntimeError(msg)
        return self.metrics_by_model[model]

    def fingerprint(self):
        return {"kind": "model-evaluator"}


def fixed_metrics():
    return {
        "reference": EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
        "cheap": EvaluationMetrics(quality=1.0, cost=2.0, latency_ms=90),
        "bad": EvaluationMetrics(quality=0.5, cost=1.0, latency_ms=50),
    }


def catalog(*, cheap_price=2.0, patches=None):
    return ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference", input_cost_per_million=10.0),
            ModelAsset(model_id="cheap", input_cost_per_million=cheap_price),
            ModelAsset(model_id="bad", input_cost_per_million=1.0),
        ],
        patches=patches,
    )


def experiment(tmp_path, evaluator, proposer, *, assets=None, objectives=None, store=None, journal=None):
    store = store or LocalRunStore()
    if not store.list():
        store.add(reference_run())
    return HarnessOptimizationExperiment(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        run_source=store,
        evaluator=evaluator,
        assets=assets or catalog(),
        objectives=objectives or OptimizationObjectives(min_quality=0.8),
        journal=journal or ExperimentJournal(tmp_path / "experiment.jsonl"),
        proposer=proposer,
    )


def test_optimizer_observes_each_outcome_before_choosing_the_next(tmp_path):
    proposer = SequenceProposer(
        [ModelSubstitutionRecipe(model_id="bad"), ModelSubstitutionRecipe(model_id="cheap"), None]
    )
    result = experiment(tmp_path, ModelEvaluator(fixed_metrics()), proposer).run()

    assert proposer.histories[0] == []
    assert proposer.histories[1][0]["recipe"]["model_id"] == "bad"
    assert proposer.histories[1][0]["gate_failures"] == ("quality_below_floor:1.0000",)
    assert proposer.histories[2][1]["metrics"]["cost"] == 2.0
    assert result.recommendation is not None
    assert result.recommendation.recipe == ModelSubstitutionRecipe(model_id="cheap")


def test_completed_measurements_seed_history_on_resume(tmp_path):
    journal = ExperimentJournal(tmp_path / "experiment.jsonl")
    first_evaluator = ModelEvaluator(fixed_metrics())
    first = experiment(
        tmp_path,
        first_evaluator,
        SequenceProposer([ModelSubstitutionRecipe(model_id="cheap"), None]),
        journal=journal,
    )
    first.run()

    resumed_evaluator = ModelEvaluator(fixed_metrics())
    resumed_proposer = SequenceProposer([None])
    result = experiment(tmp_path, resumed_evaluator, resumed_proposer, journal=journal).run()
    assert resumed_evaluator.calls == []
    assert resumed_proposer.histories[0][0]["recipe"]["model_id"] == "cheap"
    assert result.recommendation is not None


def test_patch_definition_is_part_of_candidate_identity(tmp_path):
    class StepsEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate(self, agent, reference_runs):  # noqa: ARG002
            self.calls.append(agent.max_agent_steps)
            return EvaluationMetrics(quality=1.0, cost=float(agent.max_agent_steps), latency_ms=10)

        def fingerprint(self):
            return {"kind": "steps"}

    journal = ExperimentJournal(tmp_path / "experiment.jsonl")
    first_assets = catalog(patches=[HarnessPatch(name="steps", patch={"max_agent_steps": 2})])
    first = StepsEvaluator()
    experiment(
        tmp_path,
        first,
        SequenceProposer([ApplyPatchRecipe(patch="steps"), None]),
        assets=first_assets,
        journal=journal,
    ).run()

    second_assets = catalog(patches=[HarnessPatch(name="steps", patch={"max_agent_steps": 3})])
    second = StepsEvaluator()
    experiment(
        tmp_path,
        second,
        SequenceProposer([ApplyPatchRecipe(patch="steps"), None]),
        assets=second_assets,
        journal=journal,
    ).run()
    assert second.calls == [3]


def test_objective_changes_rerank_without_remeasuring(tmp_path):
    journal = ExperimentJournal(tmp_path / "experiment.jsonl")
    experiment(
        tmp_path,
        ModelEvaluator(fixed_metrics()),
        SequenceProposer([ModelSubstitutionRecipe(model_id="cheap"), None]),
        objectives=OptimizationObjectives(min_quality=0.8),
        journal=journal,
    ).run()

    second = ModelEvaluator(fixed_metrics())
    result = experiment(
        tmp_path,
        second,
        SequenceProposer([None]),
        objectives=OptimizationObjectives(min_quality=1.0),
        journal=journal,
    ).run()
    assert second.calls == []
    assert result.recommendation is not None


def test_run_content_changes_invalidate_measurements(tmp_path):
    journal = ExperimentJournal(tmp_path / "experiment.jsonl")
    first_store = LocalRunStore()
    first_store.add(reference_run("first"))
    experiment(
        tmp_path,
        ModelEvaluator(fixed_metrics()),
        SequenceProposer([None]),
        store=first_store,
        journal=journal,
    ).run()

    second_store = LocalRunStore()
    second_store.add(reference_run("second"))
    second = ModelEvaluator(fixed_metrics())
    experiment(tmp_path, second, SequenceProposer([None]), store=second_store, journal=journal).run()
    assert second.calls == ["reference"]


def test_current_prices_apply_to_raw_journaled_usage(tmp_path):
    metrics = {
        "reference": EvaluationMetrics(
            quality=1.0, latency_ms=100, model_usage={"reference": ModelTokenUsage(input_tokens=1_000_000)}
        ),
        "cheap": EvaluationMetrics(
            quality=1.0, latency_ms=90, model_usage={"cheap": ModelTokenUsage(input_tokens=1_000_000)}
        ),
        "bad": fixed_metrics()["bad"],
    }
    journal = ExperimentJournal(tmp_path / "experiment.jsonl")
    experiment(
        tmp_path,
        ModelEvaluator(metrics),
        SequenceProposer([ModelSubstitutionRecipe(model_id="cheap"), None]),
        assets=catalog(cheap_price=2.0),
        journal=journal,
    ).run()

    resumed = ModelEvaluator(metrics)
    result = experiment(
        tmp_path,
        resumed,
        SequenceProposer([None]),
        assets=catalog(cheap_price=1.0),
        journal=journal,
    ).run()
    assert resumed.calls == []
    assert result.recommendation.evaluation.metrics.cost == 1.0


def test_failed_candidates_retry_and_duplicate_or_noop_recipes_do_not_run(tmp_path):
    journal = ExperimentJournal(tmp_path / "experiment.jsonl")
    failing = ModelEvaluator(fixed_metrics(), failing={"cheap"})
    first_proposer = SequenceProposer(
        [
            ModelSubstitutionRecipe(model_id="reference"),
            ModelSubstitutionRecipe(model_id="cheap"),
            ModelSubstitutionRecipe(model_id="cheap"),
            None,
        ]
    )
    first = experiment(tmp_path, failing, first_proposer, journal=journal).run()
    assert failing.calls == ["reference", "cheap"]
    assert first.candidates[0].failure is not None

    recovered = ModelEvaluator(fixed_metrics())
    result = experiment(
        tmp_path,
        recovered,
        SequenceProposer([ModelSubstitutionRecipe(model_id="cheap"), None]),
        journal=journal,
    ).run()
    assert recovered.calls == ["cheap"]
    assert result.recommendation is not None
