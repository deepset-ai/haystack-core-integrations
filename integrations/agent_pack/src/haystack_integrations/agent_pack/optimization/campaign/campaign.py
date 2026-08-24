# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Constrained champion/challenger optimization campaigns for Haystack Agents."""

import hashlib
import json
import traceback
from typing import Any

from haystack import logging
from haystack.components.agents import Agent
from haystack.core.serialization import component_to_dict
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.campaign.dataclasses import (
    CampaignRecommendation,
    CampaignResult,
    CandidateEvaluation,
    EvaluationMetrics,
    OptimizationObjectives,
)
from haystack_integrations.agent_pack.optimization.campaign.journal import CampaignJournal
from haystack_integrations.agent_pack.optimization.campaign.proposers import ApprovedModelRecipeProposer
from haystack_integrations.agent_pack.optimization.campaign.types.protocol import HarnessEvaluator, RecipeProposer
from haystack_integrations.agent_pack.optimization.recipes.serialization import recipe_fingerprint
from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe
from haystack_integrations.agent_pack.optimization.tracing.dataclasses import TraceArtifact, TraceSelection
from haystack_integrations.agent_pack.optimization.tracing.extraction import is_replayable
from haystack_integrations.agent_pack.optimization.tracing.types.protocol import TraceSource

logger = logging.getLogger(__name__)


class HarnessOptimizationCampaign:
    """
    Evaluate typed recipes and recommend the best candidate that satisfies every hard gate.

    Candidates are evaluated in this process. Evaluating them in a spawned subprocess is deliberately not offered:
    the only way to move a materialized Agent across a process boundary is `to_dict`/`from_dict`, which rebuilds
    document stores and other live resources empty, so an isolated candidate would be scored against a different
    world than the reference.
    """

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
        configuration_key: str | None = None,
    ) -> None:
        """
        Create a campaign.

        :param reference: The champion harness, never mutated.
        :param trace_source: Where reference traces come from.
        :param evaluator: Scores a materialized candidate.
        :param assets: The approved model and tool allowlist.
        :param objectives: Gates and ranking preference.
        :param journal: Where measurements are persisted for resume.
        :param proposer: Generates candidate recipes. Defaults to enumerating approved models.
        :param trace_selection: Which reference traces to use.
        :param configuration_key: Optional discriminator folded into the configuration hash. Use it to invalidate
            journaled results after changing something the campaign cannot see, such as the contents of the document
            store the harness retrieves from.
        """
        self.reference = reference
        self.trace_source = trace_source
        self.evaluator = evaluator
        self.assets = assets
        self.objectives = objectives
        self.journal = journal
        self.proposer = proposer or ApprovedModelRecipeProposer()
        self.trace_selection = trace_selection or TraceSelection()
        self.configuration_key = configuration_key

    def run(self) -> CampaignResult:
        """
        Evaluate the reference and every proposed candidate, resuming completed work from the journal.

        :returns: The baseline measurement, every candidate outcome, and a recommendation to approve if one candidate
            cleared the gates and improved on the reference.
        :raises ValueError: If no successful reference trace is selected, or a selected trace cannot be replayed.
        """
        reference_traces = self.trace_source.list(selection=self.trace_selection)
        if not reference_traces:
            msg = "The selected TraceSource contains no successful reference traces."
            raise ValueError(msg)
        unusable = [artifact.run_id for artifact in reference_traces if not is_replayable(artifact=artifact)]
        if unusable:
            msg = (
                f"Selected reference traces cannot be replayed: {', '.join(unusable)}. Capture them with a "
                "LocalTraceCollector configured with capture_content=True."
            )
            raise ValueError(msg)

        configuration_hash = self._configuration_hash(reference_traces=reference_traces)
        baseline = self._baseline(configuration_hash=configuration_hash, reference_traces=reference_traces)
        recipes = self.proposer.propose(
            reference=self.reference,
            reference_traces=reference_traces,
            assets=self.assets,
            objectives=self.objectives,
        )

        outcomes: list[CandidateEvaluation] = []
        recipe_by_id: dict[str, CandidateRecipe] = {}
        for recipe in recipes:
            candidate_id = self._candidate_id(configuration_hash=configuration_hash, recipe=recipe)
            recipe_by_id[candidate_id] = recipe
            prior = self.journal.get(candidate_id=candidate_id)
            if prior is not None:
                outcomes.append(prior)
                continue
            outcome = self._evaluate_candidate(
                recipe=recipe,
                candidate_id=candidate_id,
                configuration_hash=configuration_hash,
                reference_traces=reference_traces,
            )
            self.journal.append(evaluation=outcome)
            outcomes.append(outcome)

        gate_failures = {
            outcome.candidate_id: self._gate_failures(candidate=outcome, baseline=baseline) for outcome in outcomes
        }
        eligible = [outcome for outcome in outcomes if not gate_failures[outcome.candidate_id]]
        eligible.sort(key=self._candidate_rank)

        recommendation = None
        for candidate in eligible:
            reasons = self._recommendation_reasons(candidate=candidate, baseline=baseline)
            if reasons is not None:
                recommendation = CampaignRecommendation(
                    recipe=recipe_by_id[candidate.candidate_id], evaluation=candidate, reasons=reasons
                )
                break
        return CampaignResult(
            baseline=baseline,
            candidates=tuple(outcomes),
            recommendation=recommendation,
            configuration_hash=configuration_hash,
            gate_failures=gate_failures,
            reference_validation=self.assets.validate_agent(agent=self.reference),
        )

    def _configuration_hash(self, reference_traces: list[TraceArtifact]) -> str:
        """
        Identify everything that affects what a measurement means.

        :param reference_traces: The selected reference traces.
        :returns: A hex SHA-256 digest covering the reference harness, the asset catalog, the objectives, the
            evaluator, the selected traces, and any `configuration_key`.
        """
        payload = {
            "reference": self._reference_fingerprint(),
            "models": sorted(
                [
                    asset.model_id,
                    asset.provider,
                    asset.deployment,
                    asset.input_cost_per_million,
                    asset.output_cost_per_million,
                ]
                for asset in self.assets.models.values()
            ),
            "tools": sorted([asset.name, asset.provider] for asset in self.assets.tools.values()),
            "strict_identification": self.assets.strict_identification,
            "objectives": self.objectives.to_dict(),
            "evaluator": self._evaluator_fingerprint(),
            "traces": sorted(artifact.run_id for artifact in reference_traces),
            "configuration_key": self.configuration_key,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _baseline(self, *, configuration_hash: str, reference_traces: list[TraceArtifact]) -> EvaluationMetrics:
        """
        Measure the reference harness, reusing a journaled measurement of the same configuration.

        The baseline is journaled like a candidate so resuming a campaign costs nothing when everything is already
        measured. Without it, every resume pays for a full evaluation pass over the reference.
        """
        baseline_id = f"baseline:{configuration_hash}"
        prior = self.journal.get(candidate_id=baseline_id)
        if prior is not None and prior.metrics is not None:
            return prior.metrics
        metrics = self.evaluator.evaluate(agent=self.reference, reference_traces=reference_traces, assets=self.assets)
        self.journal.append(
            evaluation=CandidateEvaluation(
                candidate_id=baseline_id,
                configuration_hash=configuration_hash,
                recipe={"kind": "reference"},
                metrics=metrics,
            )
        )
        return metrics

    @staticmethod
    def _candidate_id(*, configuration_hash: str, recipe: CandidateRecipe) -> str:
        """
        Identify a candidate by both its recipe and the configuration it was measured under.

        A recipe-only identity would let a resumed campaign replay metrics recorded against a different reference
        harness, evaluation set, price table, or set of reference traces.

        :param configuration_hash: The campaign configuration hash.
        :param recipe: The transformation being measured.
        :returns: A hex SHA-256 digest identifying this measurement.
        """
        return hashlib.sha256(f"{configuration_hash}:{recipe_fingerprint(recipe=recipe)}".encode()).hexdigest()

    def _reference_fingerprint(self) -> dict[str, Any]:
        """
        Describe what about the reference harness determines its behaviour.

        Deliberately not `Agent.to_dict()`: a full serialization pulls in the configuration of everything the harness
        holds, and some of that is regenerated per process. `InMemoryDocumentStore`, for one, serializes a random
        `index` UUID, so hashing the full form would change the campaign's identity on every run and make resume
        impossible. What is described here is stable across processes; anything the campaign cannot see belongs in
        `configuration_key`.
        """
        try:
            generator: Any = component_to_dict(obj=self.reference.chat_generator, name="chat_generator")
        except Exception:
            generator = {"type": type(self.reference.chat_generator).__name__}
        return {
            "generator": generator,
            # Every model in the harness, hooks and delegated agents included.
            "models": list(self.assets.validate_agent(agent=self.reference).model_ids),
            "tools": sorted(
                [configured.name, configured.description]
                for configured in flatten_tools_or_toolsets(tools=self.reference.tools)
            ),
            "system_prompt": self.reference.system_prompt,
            "user_prompt": self.reference.user_prompt,
            "exit_conditions": list(self.reference.exit_conditions or []),
            "max_agent_steps": self.reference.max_agent_steps,
            "tool_concurrency_limit": self.reference.tool_concurrency_limit,
        }

    def _evaluator_fingerprint(self) -> dict[str, Any]:
        """Describe the evaluator, including its own fingerprint when it offers one."""
        identity: dict[str, Any] = {"type": f"{type(self.evaluator).__module__}.{type(self.evaluator).__qualname__}"}
        describe = getattr(self.evaluator, "fingerprint", None)
        if callable(describe):
            identity["configuration"] = describe()
        return identity

    def _evaluate_candidate(
        self,
        *,
        recipe: CandidateRecipe,
        candidate_id: str,
        configuration_hash: str,
        reference_traces: list[TraceArtifact],
    ) -> CandidateEvaluation:
        """Materialize, validate, and score one candidate, recording a failure rather than aborting the campaign."""
        try:
            candidate = recipe.materialize(reference=self.reference, assets=self.assets)
            validation = self.assets.require_valid_agent(agent=candidate)
            metrics = self.evaluator.evaluate(agent=candidate, reference_traces=reference_traces, assets=self.assets)
            return CandidateEvaluation(
                candidate_id=candidate_id,
                configuration_hash=configuration_hash,
                recipe=recipe.to_dict(),
                metrics=metrics,
                asset_validation=validation,
            )
        except Exception as error:
            logger.warning(
                "Candidate evaluation failed for recipe {recipe}: {error}",
                recipe=recipe.to_dict(),
                error=f"{type(error).__name__}: {error}",
            )
            return CandidateEvaluation(
                candidate_id=candidate_id,
                configuration_hash=configuration_hash,
                recipe=recipe.to_dict(),
                metrics=None,
                failure="".join(traceback.format_exception_only(type(error), error)).strip(),
            )

    def _gate_failures(self, *, candidate: CandidateEvaluation, baseline: EvaluationMetrics) -> tuple[str, ...]:
        """Return every hard gate this candidate misses. An empty tuple means it is eligible for ranking."""
        if candidate.failure is not None or candidate.metrics is None:
            return ("evaluation_failed",)
        # An asset violation is not gated here: `require_valid_agent` raises while the candidate is being
        # materialized, so a candidate that reaches this point has already cleared the catalog.
        failures: list[str] = []
        quality_floor = max(self.objectives.min_quality, baseline.gating_quality - self.objectives.max_quality_loss)
        if candidate.metrics.gating_quality < quality_floor:
            failures.append(f"quality_below_floor:{quality_floor:.4f}")
        return tuple(failures)

    def _rank(self, metrics: EvaluationMetrics) -> tuple[float, float]:
        """Order a measurement by the primary objective, breaking ties on the secondary one."""
        if self.objectives.primary == "latency":
            return metrics.latency_ms, metrics.cost
        return metrics.cost, metrics.latency_ms

    def _candidate_rank(self, candidate: CandidateEvaluation) -> tuple[float, float]:
        """Rank a candidate, sorting one that produced no measurements last."""
        if candidate.metrics is None:
            return float("inf"), float("inf")
        return self._rank(metrics=candidate.metrics)

    def _recommendation_reasons(
        self, *, candidate: CandidateEvaluation, baseline: EvaluationMetrics
    ) -> tuple[str, ...] | None:
        """Return why this candidate is recommended, or None if it should not be."""
        if candidate.metrics is None:
            return None
        reasons: list[str] = []
        if candidate.metrics.details.get("validated") is False:
            reasons.append("quality_unvalidated")
        if candidate.metrics.quality_lower_bound is None:
            # The evaluator scored each case once, so quality is a single sample of a non-deterministic run.
            reasons.append("single_sample")
        if self._rank(metrics=candidate.metrics) >= self._rank(metrics=baseline):
            return None
        reasons.append("cost_improvement" if self.objectives.primary == "cost" else "latency_improvement")
        return tuple(reasons)
