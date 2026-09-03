# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Objectives and pricing context used by harness optimization."""

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from haystack_integrations.agent_pack.dataclasses import EvaluationMetrics


@dataclass(frozen=True, kw_only=True)
class ModelPrice:
    """Informational token prices for one model deployment."""

    model_id: str
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


class ModelPriceCatalog:
    """Known prices used to explain choices and rank measured candidates."""

    def __init__(self, prices: list[ModelPrice]) -> None:
        """
        Create a catalog from unique model identifiers.

        :param prices: Informational token prices keyed by each entry's model identifier.
        :raises ValueError: If multiple entries use the same model identifier.
        """
        self.prices = {price.model_id: price for price in prices}
        if len(self.prices) != len(prices):
            msg = "Model price identifiers must be unique."
            raise ValueError(msg)

    def get(self, model_id: str) -> ModelPrice | None:
        """Return known pricing for a model without restricting model selection."""
        return self.prices.get(model_id)

    def price(self, metrics: EvaluationMetrics) -> EvaluationMetrics:
        """
        Apply known prices to raw model usage.

        :param metrics: Raw harness evaluation metrics to price.
        :returns: A copy with calculated cost, or unavailable cost and the unknown model identifiers in its details.
        """
        if metrics.cost is not None:
            return metrics
        unknown = sorted(model_id for model_id in metrics.model_usage if self.get(model_id=model_id) is None)
        if unknown:
            return replace(metrics, cost=None, details={**metrics.details, "unpriced_models": unknown})
        total = 0.0
        for model_id, usage in metrics.model_usage.items():
            price = self.get(model_id=model_id)
            if price is None:
                continue
            total += (
                usage.input_tokens * price.input_cost_per_million + usage.output_tokens * price.output_cost_per_million
            ) / 1_000_000
        return replace(metrics, cost=total)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return a JSON-compatible representation for the optimizer Agent."""
        return [asdict(obj=price) for price in self.prices.values()]


@dataclass(frozen=True, kw_only=True)
class OptimizationObjectives:
    """
    Hard quality gates and the primary ranking measurement.

    :param min_quality: Absolute minimum normalized quality in `[0.0, 1.0]` required of a candidate.
    :param max_quality_loss: Maximum absolute quality-point decrease from the reference, in `[0.0, 1.0]`.
    :param primary: Measurement minimized after candidates pass the quality gates.
    """

    min_quality: float = 0.0
    max_quality_loss: float = 0.0
    primary: Literal["cost", "latency"] = "cost"

    def __post_init__(self) -> None:
        """Validate normalized quality thresholds."""
        if not 0.0 <= self.min_quality <= 1.0:
            msg = "min_quality must be between 0.0 and 1.0."
            raise ValueError(msg)
        if not 0.0 <= self.max_quality_loss <= 1.0:
            msg = "max_quality_loss must be between 0.0 and 1.0."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(obj=self)


__all__ = [
    "ModelPrice",
    "ModelPriceCatalog",
    "OptimizationObjectives",
]
