# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The closed set of typed, allowlisted transformations for candidate Agent harnesses."""

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from haystack.components.agents import Agent

from haystack_integrations.agent_pack.optimization.assets.catalog import ApprovedAssetCatalog
from haystack_integrations.agent_pack.optimization.recipes.utils import (
    _clone_agent,
    _clone_generator_with_generation_kwargs,
    _normalized_names,
    _satisfiable_exit_conditions,
    _select_tools,
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
        return _clone_agent(reference=reference, chat_generator=generator)

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
            overrides["chat_generator"] = _clone_generator_with_generation_kwargs(
                reference_generator=reference.chat_generator, overrides=self.generation_kwargs
            )
        return _clone_agent(reference=reference, **overrides)

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
        object.__setattr__(self, "tool_names", _normalized_names(names=self.tool_names))

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:  # noqa: ARG002
        """
        Clone the Agent with only the selected tools.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist. Unused: a subset introduces no new asset.
        :returns: The new candidate Agent.
        """
        tools = _select_tools(agent=reference, names=self.tool_names)
        return _clone_agent(
            reference=reference,
            tools=tools,
            exit_conditions=_satisfiable_exit_conditions(reference=reference, tools=tools),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ToolSelectionRecipe into a dictionary.

        :returns: A dictionary with keys 'kind' and 'tool_names'.
        """
        return {"kind": self.kind, "tool_names": list(self.tool_names)}
