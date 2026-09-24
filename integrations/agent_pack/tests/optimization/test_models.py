import pytest

from haystack_integrations.agent_pack.optimization import ModelPrice, ModelPriceCatalog, OptimizationObjectives
from haystack_integrations.evaluation.dataclasses import EvalMetrics, ModelTokenUsage


def pricing():
    """Return a small informational pricing catalog."""
    return ModelPriceCatalog(
        prices=[ModelPrice(model_id="known", input_cost_per_million=2.0, output_cost_per_million=4.0)]
    )


class TestModelPriceCatalog:
    def test_prices_raw_usage(self):
        """Raw measurements are repriced from the current informational catalog."""
        metrics = pricing().price(
            metrics=EvalMetrics(
                quality=1.0,
                latency_ms=10,
                model_usage={"known": ModelTokenUsage(input_tokens=100, output_tokens=20)},
            )
        )
        assert metrics.cost == (100 * 2.0 + 20 * 4.0) / 1_000_000

    def test_unknown_model_is_unpriced(self):
        """Pricing context does not act as a model-selection allowlist."""
        metrics = pricing().price(
            metrics=EvalMetrics(
                quality=1.0,
                latency_ms=10,
                model_usage={"unknown": ModelTokenUsage(input_tokens=100)},
            )
        )
        assert metrics.cost is None
        assert metrics.details["unpriced_models"] == ["unknown"]

    def test_init_duplicate_model_ids(self):
        """Ambiguous price declarations fail at experiment construction time."""
        with pytest.raises(ValueError, match="unique"):
            ModelPriceCatalog(prices=[ModelPrice(model_id="same"), ModelPrice(model_id="same")])


class TestOptimizationObjectives:
    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"min_quality": -0.1}, "min_quality"),
            ({"min_quality": 1.1}, "min_quality"),
            ({"max_quality_loss": -0.1}, "max_quality_loss"),
            ({"max_quality_loss": 1.1}, "max_quality_loss"),
        ],
    )
    def test_init_invalid_quality(self, arguments, message):
        """Optimization quality gates use the same normalized scale as harness measurements."""
        with pytest.raises(ValueError, match=message):
            OptimizationObjectives(**arguments)
