# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the candidate recipes."""

from copy import deepcopy
from typing import Any

from haystack.components.agents import Agent
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.tools import Tool, flatten_tools_or_toolsets

TEXT_EXIT_CONDITION = "text"


def normalized_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """
    Sort and de-duplicate tool names so equivalent selections share one fingerprint.

    :param names: The names to normalize.
    :returns: The sorted, de-duplicated names.
    """
    return tuple(sorted(set(names)))


def clone_agent(reference: Agent, **overrides: Any) -> Agent:
    """
    Clone an Agent without sharing the mutable containers `Agent.clone` copies by reference.

    `Agent.clone` re-reads each init parameter off the instance, so a clone shares the reference's `hooks` dict,
    `state_schema` dict, and `tools` list. Registering a hook or appending a tool on a candidate would then mutate
    the reference harness. Fresh containers are substituted here; the hook and tool objects themselves are still
    shared, so a stateful hook instance remains shared between reference and candidate.

    :param reference: The Agent to clone.
    :param overrides: Init parameters to replace on the clone.
    :returns: The new Agent.
    """
    if "hooks" not in overrides and reference.hooks:
        overrides["hooks"] = {point: list(hooks) for point, hooks in reference.hooks.items()}
    if "state_schema" not in overrides:
        schema = getattr(reference, "state_schema", None)
        if isinstance(schema, dict):
            overrides["state_schema"] = dict(schema)
    if "tools" not in overrides and isinstance(reference.tools, list):
        overrides["tools"] = list(reference.tools)
    return reference.clone(**overrides)


def tools_by_name(agent: Agent) -> dict[str, Tool]:
    """
    Return an Agent's tools keyed by name, flattening any toolsets.

    :param agent: The Agent to inspect.
    :returns: The configured tools, by name.
    """
    return {configured.name: configured for configured in flatten_tools_or_toolsets(tools=agent.tools)}


def select_tools(agent: Agent, names: tuple[str, ...]) -> list[Tool]:
    """
    Return the named subset of an Agent's tools.

    :param agent: The Agent to select from.
    :param names: The tool names to keep.
    :returns: The selected tools, in the order the names were given.
    :raises ValueError: If a name is not configured on the Agent.
    """
    available = tools_by_name(agent=agent)
    missing = sorted(set(names) - available.keys())
    if missing:
        msg = f"Recipe references tools not configured on the Agent: {', '.join(missing)}."
        raise ValueError(msg)
    return [available[name] for name in names]


def satisfiable_exit_conditions(reference: Agent, tools: list[Tool]) -> list[str]:
    """
    Drop exit conditions naming tools the candidate no longer exposes.

    Keeping them would make the candidate raise at construction for a reason unrelated to the transformation under
    test. The text exit condition always survives, and is the fallback when nothing else does.

    :param reference: The Agent whose exit conditions are being carried over.
    :param tools: The tools the candidate will expose.
    :returns: The exit conditions the candidate can satisfy.
    """
    tool_names = {configured.name for configured in tools}
    kept = [
        condition
        for condition in (reference.exit_conditions or [TEXT_EXIT_CONDITION])
        if condition == TEXT_EXIT_CONDITION or condition in tool_names
    ]
    return kept or [TEXT_EXIT_CONDITION]


def clone_generator_with_generation_kwargs(reference_generator: Any, overrides: dict[str, Any]) -> Any:
    """
    Build a copy of a chat generator with merged generation parameters.

    :param reference_generator: The generator to copy.
    :param overrides: Generation parameters to merge over the generator's own.
    :returns: A new generator, leaving the reference generator untouched.
    :raises ValueError: If the generator does not accept a generation parameter mapping.
    """
    serialized = deepcopy(component_to_dict(obj=reference_generator, name="chat_generator"))
    init_parameters = serialized.get("init_parameters")
    if not isinstance(init_parameters, dict):
        msg = f"{type(reference_generator).__name__} has no serializable init_parameters."
        raise ValueError(msg)
    if "generation_kwargs" not in init_parameters:
        # Checked up front so an unsupported change is reported as such, rather than as a deserialization TypeError
        # from deep inside the generator's constructor.
        msg = f"{type(reference_generator).__name__} does not accept generation_kwargs."
        raise ValueError(msg)
    current = init_parameters.get("generation_kwargs") or {}
    if not isinstance(current, dict):
        msg = f"{type(reference_generator).__name__}.generation_kwargs is not a mapping."
        raise ValueError(msg)
    init_parameters["generation_kwargs"] = {**current, **overrides}
    return component_from_dict(cls=type(reference_generator), data=serialized, name="chat_generator")
