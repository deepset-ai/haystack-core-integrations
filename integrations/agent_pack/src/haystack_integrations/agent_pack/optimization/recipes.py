# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Typed, allowlisted transformations for candidate Agent harnesses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Protocol

from haystack.components.agents import Agent
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.tools import AgentTool, Tool, flatten_tools_or_toolsets

from haystack_integrations.agent_pack.optimization.policy import ApprovedAssetCatalog


class CandidateRecipe(Protocol):
    """A constrained, serializable transformation from one Agent to another."""

    kind: ClassVar[str]

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Create a candidate without mutating the reference Agent."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the recipe for hashing and campaign journaling."""
        ...


def recipe_fingerprint(recipe: CandidateRecipe) -> str:
    """Return a stable content hash for a typed candidate recipe."""
    payload = json.dumps(recipe.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _tools_by_name(agent: Agent) -> dict[str, Tool]:
    tools = flatten_tools_or_toolsets(agent.tools)
    return {tool.name: tool for tool in tools}


def _select_tools(agent: Agent, names: tuple[str, ...]) -> list[Tool]:
    available = _tools_by_name(agent)
    missing = sorted(set(names) - available.keys())
    if missing:
        msg = f"Recipe references tools not configured on the Agent: {', '.join(missing)}."
        raise ValueError(msg)
    return [available[name] for name in names]


def _clone_generator_with_generation_kwargs(reference_generator: Any, overrides: dict[str, Any]) -> Any:
    serialized = deepcopy(component_to_dict(reference_generator, "chat_generator"))
    init_parameters = serialized.get("init_parameters")
    if not isinstance(init_parameters, dict):
        msg = f"{type(reference_generator).__name__} has no serializable init_parameters."
        raise ValueError(msg)
    current = init_parameters.get("generation_kwargs") or {}
    if not isinstance(current, dict):
        msg = f"{type(reference_generator).__name__}.generation_kwargs is not a mapping."
        raise ValueError(msg)
    init_parameters["generation_kwargs"] = {**current, **overrides}
    return component_from_dict(type(reference_generator), serialized, "chat_generator")


@dataclass(frozen=True)
class ModelSubstitutionRecipe:
    """Replace the coordinator model with an approved model deployment."""

    model_id: str
    kind: ClassVar[str] = "model_substitution"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Clone the Agent with a generator built by the selected model asset."""
        generator = assets.model(self.model_id).build_generator(reference.chat_generator)
        return reference.clone(chat_generator=generator)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model substitution."""
        return {"kind": self.kind, **asdict(self)}


@dataclass(frozen=True)
class PromptAndGenerationRecipe:
    """Change the system prompt and/or generator parameters."""

    system_prompt: str | None = None
    generation_kwargs: dict[str, Any] | None = None
    kind: ClassVar[str] = "prompt_and_generation"

    def __post_init__(self) -> None:
        if self.system_prompt is None and not self.generation_kwargs:
            msg = "PromptAndGenerationRecipe requires at least one change."
            raise ValueError(msg)

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:  # noqa: ARG002
        """Clone the Agent with the requested prompt and generation settings."""
        overrides: dict[str, Any] = {}
        if self.system_prompt is not None:
            overrides["system_prompt"] = self.system_prompt
        if self.generation_kwargs:
            overrides["chat_generator"] = _clone_generator_with_generation_kwargs(
                reference.chat_generator, self.generation_kwargs
            )
        return reference.clone(**overrides)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prompt/generation change."""
        return {"kind": self.kind, **asdict(self)}


@dataclass(frozen=True)
class ToolSelectionRecipe:
    """Restrict an Agent to a named subset of its configured tools."""

    tool_names: tuple[str, ...]
    kind: ClassVar[str] = "tool_selection"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:  # noqa: ARG002
        """Clone the Agent with only the selected tools."""
        return reference.clone(tools=_select_tools(reference, self.tool_names))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool selection."""
        return {"kind": self.kind, "tool_names": list(self.tool_names)}


@dataclass(frozen=True)
class SpecialistDelegationRecipe:
    """Delegate selected tools to a cloned specialist exposed through ``AgentTool``."""

    name: str
    description: str
    specialist_tool_names: tuple[str, ...]
    specialist_system_prompt: str
    coordinator_tool_names: tuple[str, ...] = ()
    specialist_model_id: str | None = None
    kind: ClassVar[str] = "specialist_delegation"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Create a coordinator and specialist without mutating the reference Agent."""
        specialist_overrides: dict[str, Any] = {
            "tools": _select_tools(reference, self.specialist_tool_names),
            "system_prompt": self.specialist_system_prompt,
        }
        if self.specialist_model_id is not None:
            model = assets.model(self.specialist_model_id)
            specialist_overrides["chat_generator"] = model.build_generator(reference.chat_generator)
        specialist = reference.clone(**specialist_overrides)
        specialist_tool = AgentTool(agent=specialist, name=self.name, description=self.description)

        coordinator_tools = _select_tools(reference, self.coordinator_tool_names) if self.coordinator_tool_names else []
        return reference.clone(tools=[*coordinator_tools, specialist_tool])

    def to_dict(self) -> dict[str, Any]:
        """Serialize the specialist topology."""
        data = asdict(self)
        data["specialist_tool_names"] = list(self.specialist_tool_names)
        data["coordinator_tool_names"] = list(self.coordinator_tool_names)
        return {"kind": self.kind, **data}


@dataclass(frozen=True)
class CompositeRecipe:
    """Apply several typed recipes in a declared order."""

    recipes: tuple[CandidateRecipe, ...]
    kind: ClassVar[str] = "composite"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Apply each recipe to the previous candidate."""
        candidate = reference
        for recipe in self.recipes:
            candidate = recipe.materialize(candidate, assets)
        return candidate

    def to_dict(self) -> dict[str, Any]:
        """Serialize all constituent recipes."""
        return {"kind": self.kind, "recipes": [recipe.to_dict() for recipe in self.recipes]}


StructuralRecipeFactory = Callable[[Agent, ApprovedAssetCatalog, dict[str, Any]], Agent]


class StructuralRecipeRegistry:
    """Explicit allowlist of named structural transformations."""

    def __init__(self) -> None:
        self._factories: dict[str, StructuralRecipeFactory] = {}

    def register(self, name: str, factory: StructuralRecipeFactory) -> None:
        """Register one trusted transformation by stable name."""
        if name in self._factories:
            msg = f"Structural recipe {name!r} is already registered."
            raise ValueError(msg)
        self._factories[name] = factory

    def materialize(
        self, name: str, reference: Agent, assets: ApprovedAssetCatalog, parameters: dict[str, Any]
    ) -> Agent:
        """Execute one registered transformation or reject the recipe."""
        try:
            factory = self._factories[name]
        except KeyError as error:
            msg = f"Structural recipe {name!r} is not registered."
            raise ValueError(msg) from error
        return factory(reference, assets, parameters)


@dataclass(frozen=True)
class RegisteredStructuralRecipe:
    """Reference a trusted structural transformation and JSON-compatible parameters."""

    name: str
    parameters: dict[str, Any]
    registry: StructuralRecipeRegistry
    kind: ClassVar[str] = "registered_structure"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """Materialize through the explicit structural recipe registry."""
        return self.registry.materialize(self.name, reference, assets, self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trusted recipe reference, excluding the runtime registry."""
        return {"kind": self.kind, "name": self.name, "parameters": self.parameters}


def recipe_from_dict(data: dict[str, Any], *, registry: StructuralRecipeRegistry | None = None) -> CandidateRecipe:
    """Parse one optimizer proposal through the closed set of supported recipe schemas."""
    kind = data.get("kind")
    if kind == ModelSubstitutionRecipe.kind:
        return ModelSubstitutionRecipe(model_id=data["model_id"])
    if kind == PromptAndGenerationRecipe.kind:
        return PromptAndGenerationRecipe(
            system_prompt=data.get("system_prompt"), generation_kwargs=data.get("generation_kwargs")
        )
    if kind == ToolSelectionRecipe.kind:
        return ToolSelectionRecipe(tool_names=tuple(data["tool_names"]))
    if kind == SpecialistDelegationRecipe.kind:
        return SpecialistDelegationRecipe(
            name=data["name"],
            description=data["description"],
            specialist_tool_names=tuple(data["specialist_tool_names"]),
            specialist_system_prompt=data["specialist_system_prompt"],
            coordinator_tool_names=tuple(data.get("coordinator_tool_names") or ()),
            specialist_model_id=data.get("specialist_model_id"),
        )
    if kind == CompositeRecipe.kind:
        return CompositeRecipe(recipes=tuple(recipe_from_dict(item, registry=registry) for item in data["recipes"]))
    if kind == RegisteredStructuralRecipe.kind:
        if registry is None:
            msg = "A StructuralRecipeRegistry is required for registered structural recipes."
            raise ValueError(msg)
        return RegisteredStructuralRecipe(name=data["name"], parameters=data.get("parameters") or {}, registry=registry)
    msg = f"Unsupported candidate recipe kind: {kind!r}."
    raise ValueError(msg)
