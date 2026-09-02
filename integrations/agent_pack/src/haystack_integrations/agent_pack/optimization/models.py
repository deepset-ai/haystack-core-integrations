# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Assets, objectives, and raw measurements used by harness optimization."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from haystack.core.serialization import component_to_dict

_MODEL_KEYS = ("model", "azure_deployment", "model_name")
_NESTED_MODEL_CONTAINERS = ("api_params",)
_NESTED_MODEL_KEYS = ("model", "repo_id")


def _model_id_path(init_parameters: Mapping[str, Any]) -> tuple[str, ...] | None:
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
    if not isinstance(parameters, Mapping) or (path := _model_id_path(parameters)) is None:
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
        return serialized_model_id(component_to_dict(obj=generator, name="chat_generator"))
    except Exception:
        return None


@dataclass(frozen=True, kw_only=True)
class ModelAsset:
    """An approved model deployment and the prices used when experiment results are ranked."""

    model_id: str
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    generator: dict[str, Any] | None = None

    def substitution_patch(self, *, serialized_generator: dict[str, Any]) -> dict[str, Any]:
        """Return the serialized Agent patch that selects this model."""
        if self.generator is not None:
            specification = deepcopy(self.generator)
            if "type" not in specification:
                msg = f"Model asset {self.model_id!r} declares a generator without a 'type'."
                raise ValueError(msg)
            declared = serialized_model_id(specification)
            if declared != self.model_id:
                msg = (
                    f"Model asset {self.model_id!r} must declare a generator whose serialized model identifier "
                    "matches the catalog identifier."
                )
                raise ValueError(msg)
            return {"chat_generator": specification}

        parameters = serialized_generator.get("init_parameters")
        path = _model_id_path(parameters) if isinstance(parameters, Mapping) else None
        if path is None:
            msg = (
                f"Model asset {self.model_id!r} needs an explicit generator because the reference generator does "
                "not serialize a model identifier."
            )
            raise ValueError(msg)
        return {".".join(("chat_generator", "init_parameters", *path)): self.model_id}

    def identity(self) -> dict[str, Any]:
        """Return everything that changes the candidate this asset materializes."""
        return {"model_id": self.model_id, "generator": self.generator}


@dataclass(frozen=True, kw_only=True)
class HarnessPatch:
    """A named, approved patch over an Agent's serialized initialization parameters."""

    name: str
    patch: dict[str, Any]
    description: str | None = None

    def identity(self) -> dict[str, Any]:
        """Return everything that changes the candidate this patch materializes."""
        return {"name": self.name, "patch": self.patch}


class ApprovedAssetCatalog:
    """Models and named configuration changes the optimizer may select."""

    def __init__(self, *, models: list[ModelAsset], patches: list[HarnessPatch] | None = None) -> None:
        self.models = {asset.model_id: asset for asset in models}
        self.patches = {declared.name: declared for declared in patches or []}
        if len(self.models) != len(models):
            msg = "Model asset IDs must be unique."
            raise ValueError(msg)
        if len(self.patches) != len(patches or []):
            msg = "Patch names must be unique."
            raise ValueError(msg)

    def model(self, model_id: str) -> ModelAsset:
        """Return an approved model or fail closed."""
        try:
            return self.models[model_id]
        except KeyError as error:
            msg = f"Model {model_id!r} is not in the approved asset catalog."
            raise ValueError(msg) from error

    def patch(self, name: str) -> HarnessPatch:
        """Return an approved patch or fail closed."""
        try:
            return self.patches[name]
        except KeyError as error:
            msg = f"Patch {name!r} is not in the approved asset catalog."
            raise ValueError(msg) from error


@dataclass(frozen=True, kw_only=True)
class ModelTokenUsage:
    """Raw token usage attributable to one model deployment."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, kw_only=True)
class EvaluationMetrics:
    """Quality, latency, and raw cost inputs measured for one harness."""

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

    def price(self, assets: ApprovedAssetCatalog) -> "EvaluationMetrics":
        """Return metrics with current catalog prices applied to raw model usage."""
        if self.cost is not None:
            return self
        total = 0.0
        for model_id, usage in self.model_usage.items():
            asset = assets.model(model_id)
            total += (
                usage.input_tokens * asset.input_cost_per_million + usage.output_tokens * asset.output_cost_per_million
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
    "ApprovedAssetCatalog",
    "EvaluationMetrics",
    "HarnessPatch",
    "ModelAsset",
    "ModelTokenUsage",
    "OptimizationObjectives",
    "generator_model_id",
]
