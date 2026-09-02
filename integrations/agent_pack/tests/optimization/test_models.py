import pytest
from haystack.components.generators.chat import MockChatGenerator

from haystack_integrations.agent_pack.optimization import (
    EvaluationMetrics,
    ModelPrice,
    ModelPriceCatalog,
    ModelTokenUsage,
)
from haystack_integrations.agent_pack.optimization.models import generator_model_id, serialized_model_id


def pricing():
    """Return a small informational pricing catalog."""
    return ModelPriceCatalog(
        prices=[ModelPrice(model_id="known", input_cost_per_million=2.0, output_cost_per_million=4.0)]
    )


def test_pricing_catalog_prices_raw_usage():
    """Raw measurements are repriced from the current informational catalog."""
    metrics = EvaluationMetrics(
        quality=1.0,
        latency_ms=10,
        model_usage={"known": ModelTokenUsage(input_tokens=100, output_tokens=20)},
    ).price(pricing=pricing())
    assert metrics.cost == (100 * 2.0 + 20 * 4.0) / 1_000_000


def test_unknown_model_is_allowed_but_explicitly_unpriced():
    """Pricing context does not act as a model-selection allowlist."""
    metrics = EvaluationMetrics(
        quality=1.0,
        latency_ms=10,
        model_usage={"unknown": ModelTokenUsage(input_tokens=100)},
    ).price(pricing=pricing())
    assert metrics.cost is None
    assert metrics.details["unpriced_models"] == ["unknown"]


def test_price_identifiers_must_be_unique():
    """Ambiguous price declarations fail at experiment construction time."""
    with pytest.raises(ValueError, match="unique"):
        ModelPriceCatalog(prices=[ModelPrice(model_id="same"), ModelPrice(model_id="same")])


def test_model_identifier_helpers_support_live_and_serialized_generators():
    """Usage attribution recognizes common live and serialized model fields."""
    generator = MockChatGenerator(model="known")
    assert generator_model_id(generator=generator) == "known"
    assert serialized_model_id(serialized_component=generator.to_dict()) == "known"
