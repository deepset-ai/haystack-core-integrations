# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from haystack_integrations.evaluation.dataclasses import EvalMetrics, ModelTokenUsage


@dataclass(kw_only=True)
class OptimizationObjectives:
    """
    Hard quality gates and the primary ranking measurement.

    :param min_quality: Absolute minimum normalized quality in `[0.0, 1.0]` required of a candidate.
    :param max_quality_loss: Maximum absolute quality-point decrease from the reference, in `[0.0, 1.0]`.
    :param primary: What candidates are ranked by. "cost" and "latency" are minimized among candidates that clear
        the quality gates; "quality" is maximized directly, with cost breaking ties, which needs no quality
        threshold to be chosen in advance and cannot prefer a cheaper configuration that answers worse.
    """

    min_quality: float = 0.0
    max_quality_loss: float = 0.0
    primary: Literal["cost", "latency", "quality"] = "cost"

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


@dataclass(kw_only=True)
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
        """Return known pricing for a model."""
        return self.prices.get(model_id)

    def cost_of(self, model_usage: dict[str, ModelTokenUsage]) -> float | None:
        """
        Calculate what raw token usage costs at known prices.

        Every input token is charged at full price. A provider that discounts tokens served from its prompt cache
        charges less than this, so the result is an upper bound wherever caching is in play, and comparisons
        between configurations that cache alike stay fair.

        :param model_usage: Raw token usage keyed by model identifier.
        :returns: The total cost, or `None` when any model in the usage has no known price.
        """
        if any(self.get(model_id=model_id) is None for model_id in model_usage):
            return None
        total = 0.0
        for model_id, usage in model_usage.items():
            price = self.get(model_id=model_id)
            if price is None:  # pragma: no cover - excluded by the check above
                continue
            total += (
                usage.input_tokens * price.input_cost_per_million + usage.output_tokens * price.output_cost_per_million
            ) / 1_000_000
        return total

    def price(self, metrics: EvalMetrics) -> EvalMetrics:
        """
        Apply known prices to raw model usage.

        :param metrics: Raw harness evaluation metrics to price.
        :returns: A copy with calculated cost, or unavailable cost and the unknown model identifiers in its details.
        """
        if metrics.details.get("all_tokens_reported") is False:
            return replace(metrics, cost=None)
        if metrics.cost is not None:
            return metrics
        unknown = sorted(model_id for model_id in metrics.model_usage if self.get(model_id=model_id) is None)
        if unknown:
            return replace(metrics, cost=None, details={**metrics.details, "unpriced_models": unknown})
        return replace(metrics, cost=self.cost_of(model_usage=metrics.model_usage))

    def to_dict(self) -> list[dict[str, Any]]:
        """Return a JSON-compatible representation for the optimizer Agent."""
        return [asdict(obj=price) for price in self.prices.values()]
