# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The closed set of typed, allowlisted transformations for candidate Agent harnesses."""

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from haystack.components.agents import Agent
from haystack.tools import AgentTool

from haystack_integrations.agent_pack.optimization.policy.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe
from haystack_integrations.agent_pack.optimization.recipes.utils import (
    TEXT_EXIT_CONDITION,
    clone_agent,
    clone_generator_with_generation_kwargs,
    normalized_names,
    satisfiable_exit_conditions,
    select_tools,
)


@dataclass(frozen=True, kw_only=True)
class ModelSubstitutionRecipe:
    """
    Replace the coordinator model with an approved model deployment.

    :param model_id: The approved model to switch to.
    """

    model_id: str
    kind: ClassVar[str] = "model_substitution"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Clone the Agent with a generator built by the selected model asset.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist.
        :returns: The new candidate Agent.
        """
        generator = assets.model(model_id=self.model_id).build_generator(reference_generator=reference.chat_generator)
        return clone_agent(reference=reference, chat_generator=generator)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ModelSubstitutionRecipe into a dictionary.

        :returns: A dictionary with keys 'kind' and 'model_id'.
        """
        return {"kind": self.kind, **asdict(self)}


@dataclass(frozen=True, kw_only=True)
class PromptAndGenerationRecipe:
    """
    Change the system prompt and/or generator parameters.

    :param system_prompt: Replacement system prompt.
    :param generation_kwargs: Generation parameters merged over the reference generator's own.
    """

    system_prompt: str | None = None
    generation_kwargs: dict[str, Any] | None = None
    kind: ClassVar[str] = "prompt_and_generation"

    def __post_init__(self) -> None:
        if self.system_prompt is None and not self.generation_kwargs:
            msg = "PromptAndGenerationRecipe requires at least one change."
            raise ValueError(msg)

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:  # noqa: ARG002
        """
        Clone the Agent with the requested prompt and generation settings.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist. Unused: neither change introduces a new asset.
        :returns: The new candidate Agent.
        """
        overrides: dict[str, Any] = {}
        if self.system_prompt is not None:
            overrides["system_prompt"] = self.system_prompt
        if self.generation_kwargs:
            overrides["chat_generator"] = clone_generator_with_generation_kwargs(
                reference_generator=reference.chat_generator, overrides=self.generation_kwargs
            )
        return clone_agent(reference=reference, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the PromptAndGenerationRecipe into a dictionary.

        :returns: A dictionary with keys 'kind', 'system_prompt', and 'generation_kwargs'.
        """
        return {"kind": self.kind, **asdict(self)}


@dataclass(frozen=True, kw_only=True)
class ToolSelectionRecipe:
    """
    Restrict an Agent to a named subset of its configured tools.

    :param tool_names: The tools to keep. Sorted and de-duplicated, so two orderings of the same set describe one
        candidate rather than two.
    """

    tool_names: tuple[str, ...]
    kind: ClassVar[str] = "tool_selection"

    def __post_init__(self) -> None:
        if not self.tool_names:
            msg = "ToolSelectionRecipe requires at least one tool name."
            raise ValueError(msg)
        object.__setattr__(self, "tool_names", normalized_names(names=self.tool_names))

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:  # noqa: ARG002
        """
        Clone the Agent with only the selected tools.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist. Unused: a subset introduces no new asset.
        :returns: The new candidate Agent.
        """
        tools = select_tools(agent=reference, names=self.tool_names)
        return clone_agent(
            reference=reference,
            tools=tools,
            exit_conditions=satisfiable_exit_conditions(reference=reference, tools=tools),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ToolSelectionRecipe into a dictionary.

        :returns: A dictionary with keys 'kind' and 'tool_names'.
        """
        return {"kind": self.kind, "tool_names": list(self.tool_names)}


@dataclass(frozen=True, kw_only=True)
class SpecialistDelegationRecipe:
    """
    Delegate selected tools to a cloned specialist exposed through `AgentTool`.

    The specialist is a clone of the reference restricted to `specialist_tool_names`. It gets its own system prompt,
    no user prompt, and the text exit condition, because inheriting the coordinator's exit conditions and prompt
    variables would make it fail for reasons unrelated to the topology being tested.

    :param name: Name of the `AgentTool` the coordinator calls.
    :param description: Tool description shown to the coordinator.
    :param specialist_tool_names: Tools moved to the specialist.
    :param specialist_system_prompt: The specialist's own instructions.
    :param coordinator_tool_names: Tools the coordinator keeps in addition to the specialist tool.
    :param specialist_model_id: Optional approved model for the specialist, defaulting to the reference model.
    :param coordinator_system_prompt: Optional replacement coordinator prompt. When omitted, the reference prompt is
        kept and a short instruction naming the specialist tool is appended, because a coordinator whose prompt still
        describes the tools it no longer has will not reliably delegate.
    """

    name: str
    description: str
    specialist_tool_names: tuple[str, ...]
    specialist_system_prompt: str
    coordinator_tool_names: tuple[str, ...] = ()
    specialist_model_id: str | None = None
    coordinator_system_prompt: str | None = None
    kind: ClassVar[str] = "specialist_delegation"

    def __post_init__(self) -> None:
        if not self.specialist_tool_names:
            msg = "SpecialistDelegationRecipe requires at least one specialist tool."
            raise ValueError(msg)
        object.__setattr__(self, "specialist_tool_names", normalized_names(names=self.specialist_tool_names))
        object.__setattr__(self, "coordinator_tool_names", normalized_names(names=self.coordinator_tool_names))

    def _coordinator_prompt(self, reference_prompt: str | None) -> str:
        """Keep the reference prompt and append an instruction to route the delegated tools to the specialist."""
        delegation = (
            f"Use the `{self.name}` tool for anything that needs "
            f"{', '.join(self.specialist_tool_names)}; do not call those tools yourself."
        )
        return f"{reference_prompt}\n\n{delegation}" if reference_prompt else delegation

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Create a coordinator and specialist without mutating the reference Agent.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist.
        :returns: The new coordinator Agent, exposing the specialist as a tool.
        """
        specialist_overrides: dict[str, Any] = {
            "tools": select_tools(agent=reference, names=self.specialist_tool_names),
            "system_prompt": self.specialist_system_prompt,
            "user_prompt": None,
            "exit_conditions": [TEXT_EXIT_CONDITION],
        }
        if self.specialist_model_id is not None:
            model = assets.model(model_id=self.specialist_model_id)
            specialist_overrides["chat_generator"] = model.build_generator(reference_generator=reference.chat_generator)
        specialist = clone_agent(reference=reference, **specialist_overrides)
        specialist_tool = AgentTool(agent=specialist, name=self.name, description=self.description)

        coordinator_tools = [
            *select_tools(agent=reference, names=self.coordinator_tool_names),
            specialist_tool,
        ]
        return clone_agent(
            reference=reference,
            tools=coordinator_tools,
            system_prompt=self.coordinator_system_prompt
            or self._coordinator_prompt(reference_prompt=reference.system_prompt),
            exit_conditions=satisfiable_exit_conditions(reference=reference, tools=coordinator_tools),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the SpecialistDelegationRecipe into a dictionary.

        :returns: A dictionary with one key per field plus 'kind', with the tool name tuples as lists.
        """
        data = asdict(self)
        data["specialist_tool_names"] = list(self.specialist_tool_names)
        data["coordinator_tool_names"] = list(self.coordinator_tool_names)
        return {"kind": self.kind, **data}


@dataclass(frozen=True, kw_only=True)
class CompositeRecipe:
    """
    Apply several typed recipes in a declared order.

    :param recipes: The transformations to apply, each to the result of the previous one.
    """

    recipes: tuple[CandidateRecipe, ...]
    kind: ClassVar[str] = "composite"

    def __post_init__(self) -> None:
        if not self.recipes:
            msg = "CompositeRecipe requires at least one recipe."
            raise ValueError(msg)

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Apply each recipe to the previous candidate.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist.
        :returns: The new candidate Agent.
        """
        candidate = reference
        for recipe in self.recipes:
            candidate = recipe.materialize(reference=candidate, assets=assets)
        return candidate

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the CompositeRecipe into a dictionary.

        :returns: A dictionary with keys 'kind' and 'recipes'.
        """
        return {"kind": self.kind, "recipes": [recipe.to_dict() for recipe in self.recipes]}
