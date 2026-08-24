# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Constrained champion/challenger optimization campaigns for Haystack Agents."""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Protocol

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.optimization.policy import ApprovedAssetCatalog, AssetValidation
from haystack_integrations.agent_pack.optimization.recipes import (
    CandidateRecipe,
    ModelSubstitutionRecipe,
    StructuralRecipeRegistry,
    recipe_fingerprint,
    recipe_from_dict,
)
from haystack_integrations.agent_pack.optimization.tracing import (
    TraceArtifact,
    TraceSelection,
    TraceSource,
    extract_agent_replay_inputs,
)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Comparable quality, cost, and latency metrics for one harness."""

    quality: float
    cost: float
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationMetrics:
        """Deserialize metrics from a campaign journal."""
        return cls(
            quality=float(data["quality"]),
            cost=float(data["cost"]),
            latency_ms=float(data["latency_ms"]),
            details=data.get("details") or {},
        )


class HarnessEvaluator(Protocol):
    """Evaluate one materialized candidate against selected reference traces."""

    def evaluate(self, agent: Agent, reference_traces: list[TraceArtifact]) -> EvaluationMetrics:
        """Return comparable campaign metrics."""
        ...


def _evaluate_serialized_agent(
    serialized_agent: dict[str, Any], evaluator: HarnessEvaluator, reference_traces: list[TraceArtifact]
) -> EvaluationMetrics:
    agent = Agent.from_dict(serialized_agent)
    return evaluator.evaluate(agent, reference_traces)


class IsolatedHarnessEvaluator:
    """Run a serializable evaluator and Agent in a fresh spawned process."""

    def __init__(self, evaluator: HarnessEvaluator, *, timeout_seconds: float = 600.0) -> None:
        self.evaluator = evaluator
        self.timeout_seconds = timeout_seconds

    def evaluate(self, agent: Agent, reference_traces: list[TraceArtifact]) -> EvaluationMetrics:
        """Serialize the Agent and execute the evaluation outside the campaign process."""
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            future = executor.submit(_evaluate_serialized_agent, agent.to_dict(), self.evaluator, reference_traces)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FutureTimeoutError as error:
                future.cancel()
                msg = f"Candidate evaluation exceeded {self.timeout_seconds} seconds."
                raise TimeoutError(msg) from error


@dataclass(frozen=True)
class OptimizationObjectives:
    """Campaign gates and ranking preferences."""

    min_quality: float = 0.0
    max_quality_loss: float = 0.0
    require_sovereign: bool = False
    primary: Literal["cost", "latency"] = "cost"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_quality <= 1.0:
            msg = "min_quality must be between 0 and 1."
            raise ValueError(msg)
        if self.max_quality_loss < 0:
            msg = "max_quality_loss cannot be negative."
            raise ValueError(msg)


class RecipeProposer(Protocol):
    """Generate typed candidate recipes from approved inputs."""

    def propose(
        self,
        *,
        reference: Agent,
        reference_traces: list[TraceArtifact],
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
    ) -> list[CandidateRecipe]:
        """Return a finite list of constrained recipes."""
        ...


class ApprovedModelRecipeProposer:
    """Deterministically evaluate every approved alternative model."""

    def propose(
        self,
        *,
        reference: Agent,
        reference_traces: list[TraceArtifact],  # noqa: ARG002
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,  # noqa: ARG002
    ) -> list[CandidateRecipe]:
        """Return one substitution recipe per approved non-reference model."""
        reference_model = getattr(reference.chat_generator, "model", None)
        return [
            ModelSubstitutionRecipe(model_id=model_id)
            for model_id in sorted(assets.models)
            if model_id != reference_model
        ]


class HarnessOptimizerAgentProposer:
    """Ask a skill-guided Agent for JSON recipes and validate them against the closed recipe schemas."""

    def __init__(
        self,
        optimizer_agent: Agent,
        *,
        registry: StructuralRecipeRegistry | None = None,
        max_recipes: int = 8,
    ) -> None:
        self.optimizer_agent = optimizer_agent
        self.registry = registry
        self.max_recipes = max_recipes

    def propose(
        self,
        *,
        reference: Agent,
        reference_traces: list[TraceArtifact],
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
    ) -> list[CandidateRecipe]:
        """Request proposals and reject any response outside the typed recipe language."""
        sample_inputs = [extract_agent_replay_inputs(trace) for trace in reference_traces[:3]]
        request = {
            "reference": {
                "model": getattr(reference.chat_generator, "model", None),
                "system_prompt": reference.system_prompt,
                "tools": [tool.name for tool in flatten_tools_or_toolsets(reference.tools)],
            },
            "approved_models": [
                {
                    "model_id": asset.model_id,
                    "provider": asset.provider,
                    "deployment": asset.deployment,
                    "sovereign": asset.sovereign,
                    "input_cost_per_million": asset.input_cost_per_million,
                    "output_cost_per_million": asset.output_cost_per_million,
                }
                for asset in assets.models.values()
            ],
            "approved_tools": sorted(assets.tools),
            "objectives": asdict(objectives),
            "successful_trace_inputs": sample_inputs,
            "maximum_recipes": self.max_recipes,
        }
        result = self.optimizer_agent.run(messages=[ChatMessage.from_user(json.dumps(request, default=str))])
        text = result["last_message"].text
        if not text:
            msg = "Harness optimizer Agent returned no recipe JSON."
            raise ValueError(msg)
        try:
            proposals = json.loads(text)
        except json.JSONDecodeError as error:
            msg = "Harness optimizer Agent must return only a JSON array of typed recipes."
            raise ValueError(msg) from error
        if not isinstance(proposals, list) or len(proposals) > self.max_recipes:
            msg = f"Harness optimizer Agent must return at most {self.max_recipes} recipes."
            raise ValueError(msg)
        return [recipe_from_dict(proposal, registry=self.registry) for proposal in proposals]


@dataclass(frozen=True)
class CandidateEvaluation:
    """Journaled outcome for one recipe."""

    candidate_id: str
    recipe: dict[str, Any]
    valid: bool
    metrics: EvaluationMetrics | None
    asset_validation: AssetValidation | None = None
    failure: str | None = None
    policy_decisions: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation for JSON-lines persistence."""
        return {
            "candidate_id": self.candidate_id,
            "recipe": self.recipe,
            "valid": self.valid,
            "metrics": asdict(self.metrics) if self.metrics is not None else None,
            "asset_validation": asdict(self.asset_validation) if self.asset_validation is not None else None,
            "failure": self.failure,
            "policy_decisions": list(self.policy_decisions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateEvaluation:
        """Deserialize a prior evaluation from the campaign journal."""
        validation = data.get("asset_validation")
        return cls(
            candidate_id=data["candidate_id"],
            recipe=data["recipe"],
            valid=bool(data["valid"]),
            metrics=EvaluationMetrics.from_dict(data["metrics"]) if data.get("metrics") is not None else None,
            asset_validation=(
                AssetValidation(
                    allowed=validation["allowed"],
                    model_ids=tuple(validation["model_ids"]),
                    tool_names=tuple(validation["tool_names"]),
                    reason_codes=tuple(validation.get("reason_codes") or ()),
                )
                if validation is not None
                else None
            ),
            failure=data.get("failure"),
            policy_decisions=tuple(data.get("policy_decisions") or ()),
        )


class CampaignJournal:
    """Append-only JSON-lines journal used to resume candidate evaluation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: dict[str, CandidateEvaluation] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = CandidateEvaluation.from_dict(json.loads(line))
                    self._records[record.candidate_id] = record

    def get(self, candidate_id: str) -> CandidateEvaluation | None:
        """Return a previously completed evaluation by content hash."""
        with self._lock:
            return self._records.get(candidate_id)

    def append(self, evaluation: CandidateEvaluation) -> None:
        """Persist an evaluation unless the candidate is already journaled."""
        with self._lock:
            if evaluation.candidate_id in self._records:
                return
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(evaluation.to_dict(), sort_keys=True) + "\n")
            self._records[evaluation.candidate_id] = evaluation


@dataclass(frozen=True)
class CampaignRecommendation:
    """A candidate that met the hard gates and outranked the reference harness."""

    recipe: CandidateRecipe
    evaluation: CandidateEvaluation

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Explicitly materialize the recommended candidate for human inspection or approval."""
        candidate = self.recipe.materialize(reference, assets)
        assets.require_valid_agent(candidate)
        return candidate


@dataclass(frozen=True)
class CampaignResult:
    """Baseline, candidate outcomes, and optional recommendation."""

    baseline: EvaluationMetrics
    candidates: tuple[CandidateEvaluation, ...]
    recommendation: CampaignRecommendation | None


class HarnessOptimizationCampaign:
    """Evaluate typed recipes and recommend the best candidate that satisfies every hard gate."""

    def __init__(
        self,
        *,
        reference: Agent,
        trace_source: TraceSource,
        evaluator: HarnessEvaluator,
        assets: ApprovedAssetCatalog,
        objectives: OptimizationObjectives,
        journal: CampaignJournal,
        proposer: RecipeProposer | None = None,
        trace_selection: TraceSelection | None = None,
        isolate_evaluations: bool = True,
        evaluation_timeout_seconds: float = 600.0,
    ) -> None:
        self.reference = reference
        self.trace_source = trace_source
        self.evaluator = (
            IsolatedHarnessEvaluator(evaluator, timeout_seconds=evaluation_timeout_seconds)
            if isolate_evaluations
            else evaluator
        )
        self.assets = assets
        self.objectives = objectives
        self.journal = journal
        self.proposer = proposer or ApprovedModelRecipeProposer()
        self.trace_selection = trace_selection or TraceSelection()

    def run(self) -> CampaignResult:
        """Evaluate the reference and every proposed candidate, resuming completed work from the journal."""
        reference_traces = self.trace_source.list(self.trace_selection)
        if not reference_traces:
            msg = "The selected TraceSource contains no successful reference traces."
            raise ValueError(msg)
        for trace in reference_traces:
            extract_agent_replay_inputs(trace)

        baseline = self.evaluator.evaluate(self.reference, reference_traces)
        recipes = self.proposer.propose(
            reference=self.reference,
            reference_traces=reference_traces,
            assets=self.assets,
            objectives=self.objectives,
        )
        outcomes: list[CandidateEvaluation] = []
        recipe_by_id: dict[str, CandidateRecipe] = {}
        for recipe in recipes:
            candidate_id = recipe_fingerprint(recipe)
            recipe_by_id[candidate_id] = recipe
            prior = self.journal.get(candidate_id)
            if prior is not None:
                outcomes.append(prior)
                continue
            outcome = self._evaluate_candidate(recipe, candidate_id, reference_traces, baseline)
            self.journal.append(outcome)
            outcomes.append(outcome)

        eligible = [outcome for outcome in outcomes if outcome.valid and outcome.metrics is not None]
        eligible.sort(key=self._candidate_rank)
        best = eligible[0] if eligible else None
        recommendation = None
        if best is not None and self._outranks_baseline(best, baseline):
            recommendation = CampaignRecommendation(recipe=recipe_by_id[best.candidate_id], evaluation=best)
        return CampaignResult(baseline=baseline, candidates=tuple(outcomes), recommendation=recommendation)

    def _evaluate_candidate(
        self,
        recipe: CandidateRecipe,
        candidate_id: str,
        reference_traces: list[TraceArtifact],
        baseline: EvaluationMetrics,
    ) -> CandidateEvaluation:
        try:
            candidate = recipe.materialize(self.reference, self.assets)
            validation = self.assets.require_valid_agent(candidate)
            metrics = self.evaluator.evaluate(candidate, reference_traces)
            valid = self._passes_gates(metrics, validation, baseline)
            decisions = metrics.details.get("policy_decisions") or []
            return CandidateEvaluation(
                candidate_id=candidate_id,
                recipe=recipe.to_dict(),
                valid=valid,
                metrics=metrics,
                asset_validation=validation,
                policy_decisions=tuple(decisions) if isinstance(decisions, list) else (),
            )
        except Exception as error:
            return CandidateEvaluation(
                candidate_id=candidate_id,
                recipe=recipe.to_dict(),
                valid=False,
                metrics=None,
                failure=f"{type(error).__name__}: {error}",
            )

    def _passes_gates(
        self, metrics: EvaluationMetrics, validation: AssetValidation, baseline: EvaluationMetrics
    ) -> bool:
        quality_floor = max(self.objectives.min_quality, baseline.quality - self.objectives.max_quality_loss)
        if metrics.quality < quality_floor:
            return False
        if self.objectives.require_sovereign:
            return all(self.assets.model(model_id).sovereign for model_id in validation.model_ids)
        return True

    def _rank(self, metrics: EvaluationMetrics) -> tuple[float, float]:
        if self.objectives.primary == "latency":
            return metrics.latency_ms, metrics.cost
        return metrics.cost, metrics.latency_ms

    def _candidate_rank(self, candidate: CandidateEvaluation) -> tuple[float, float]:
        if candidate.metrics is None:
            return float("inf"), float("inf")
        return self._rank(candidate.metrics)

    def _outranks_baseline(self, candidate: CandidateEvaluation, baseline: EvaluationMetrics) -> bool:
        if candidate.metrics is None:
            return False
        if self.objectives.require_sovereign and candidate.asset_validation is not None:
            reference_model = getattr(self.reference.chat_generator, "model", None)
            reference_is_sovereign = (
                isinstance(reference_model, str)
                and reference_model in self.assets.models
                and self.assets.models[reference_model].sovereign
            )
            if not reference_is_sovereign:
                return True
        return self._rank(candidate.metrics) < self._rank(baseline)
