# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Objectives, pricing context, and raw measurements used by harness optimization."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from haystack.core.serialization import component_to_dict

_MODEL_KEYS = ("model", "azure_deployment", "model_name")
_NESTED_MODEL_CONTAINERS = ("api_params",)
_NESTED_MODEL_KEYS = ("model", "repo_id")


def _model_id_path(init_parameters: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Locate a recognized model identifier in serialized generator parameters."""
    for key in _MODEL_KEYS:
        if isinstance(init_parameters.get(key), str):
            return (key,)
    for container in _NESTED_MODEL_CONTAINERS:
        nested = init_parameters.get(container)
        if isinstance(nested, Mapping):
            for key in _NESTED_MODEL_KEYS:
                if isinstance(nested.get(key), str):
                    return (container, key)
    return None


def serialized_model_id(serialized_component: Mapping[str, Any]) -> str | None:
    """Return the model identifier in a serialized chat generator, if recognizable."""
    parameters = serialized_component.get("init_parameters") or serialized_component.get("data") or {}
    if not isinstance(parameters, Mapping) or (path := _model_id_path(init_parameters=parameters)) is None:
        return None
    value: Any = parameters
    for key in path:
        value = value[key]
    return value if isinstance(value, str) else None


def generator_model_id(generator: Any) -> str | None:
    """Return the configured model identifier of a live chat generator."""
    for key in _MODEL_KEYS:
        if isinstance(value := getattr(generator, key, None), str):
            return value
    for container in _NESTED_MODEL_CONTAINERS:
        nested = getattr(generator, container, None)
        if isinstance(nested, Mapping):
            for key in _NESTED_MODEL_KEYS:
                if isinstance(value := nested.get(key), str):
                    return value
    try:
        return serialized_model_id(serialized_component=component_to_dict(obj=generator, name="chat_generator"))
    except Exception:
        return None


@dataclass(frozen=True, kw_only=True)
class ModelPrice:
    """Informational token prices for one model deployment."""

    model_id: str
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


class ModelPriceCatalog:
    """Known prices used to explain choices and rank measured candidates."""

    def __init__(self, prices: list[ModelPrice]) -> None:
        """Create a catalog from unique model identifiers."""
        self.prices = {price.model_id: price for price in prices}
        if len(self.prices) != len(prices):
            msg = "Model price identifiers must be unique."
            raise ValueError(msg)

    def get(self, model_id: str) -> ModelPrice | None:
        """Return known pricing for a model without restricting model selection."""
        return self.prices.get(model_id)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return a JSON-compatible representation for the optimizer Agent."""
        return [asdict(price) for price in self.prices.values()]


@dataclass(frozen=True, kw_only=True)
class ModelTokenUsage:
    """Raw token usage attributable to one model deployment."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, kw_only=True)
class EvaluationMetrics:
    """Quality, latency, and raw cost inputs measured for one Agent configuration."""

    quality: float
    latency_ms: float
    model_usage: dict[str, ModelTokenUsage] = field(default_factory=dict)
    cost: float | None = None
    quality_lower_bound: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def gating_quality(self) -> float:
        """Return the pessimistic quality value used by hard gates."""
        return self.quality if self.quality_lower_bound is None else self.quality_lower_bound

    def price(self, pricing: ModelPriceCatalog) -> "EvaluationMetrics":
        """Apply known prices to raw usage, leaving unknown model usage explicitly unpriced."""
        if self.cost is not None:
            return self
        unknown = sorted(model_id for model_id in self.model_usage if pricing.get(model_id=model_id) is None)
        if unknown:
            return replace(self, cost=None, details={**self.details, "unpriced_models": unknown})
        total = 0.0
        for model_id, usage in self.model_usage.items():
            price = pricing.get(model_id=model_id)
            if price is None:
                continue
            total += (
                usage.input_tokens * price.input_cost_per_million + usage.output_tokens * price.output_cost_per_million
            ) / 1_000_000
        return replace(self, cost=total)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "quality": self.quality,
            "latency_ms": self.latency_ms,
            "model_usage": {model: asdict(usage) for model, usage in self.model_usage.items()},
            "cost": self.cost,
            "quality_lower_bound": self.quality_lower_bound,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationMetrics":
        """Restore metrics written by :meth:`to_dict`."""
        lower_bound = data.get("quality_lower_bound")
        cost = data.get("cost")
        return cls(
            quality=float(data["quality"]),
            latency_ms=float(data["latency_ms"]),
            model_usage={model: ModelTokenUsage(**usage) for model, usage in (data.get("model_usage") or {}).items()},
            cost=None if cost is None else float(cost),
            quality_lower_bound=None if lower_bound is None else float(lower_bound),
            details=data.get("details") or {},
        )


@dataclass(frozen=True, kw_only=True)
class OptimizationObjectives:
    """Hard quality gates and the primary ranking measurement."""

    min_quality: float = 0.0
    max_quality_loss: float = 0.0
    primary: Literal["cost", "latency"] = "cost"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


__all__ = [
    "EvaluationMetrics",
    "ModelPrice",
    "ModelPriceCatalog",
    "ModelTokenUsage",
    "OptimizationObjectives",
    "generator_model_id",
    "serialized_model_id",
]
