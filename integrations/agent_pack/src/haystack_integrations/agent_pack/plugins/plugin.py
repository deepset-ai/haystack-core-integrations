# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from haystack.components.agents import Agent
from haystack.hooks import Hook, HookPoint
from haystack.tools import Tool, Toolset, ToolsType, flatten_tools_or_toolsets

_MESSAGE_BLOCK_START = re.compile(r"{%\s*message\b")
_MESSAGE_BLOCK_END = re.compile(r"{%\s*endmessage\s*%}")


@dataclass(frozen=True)
class AgentPlugin:
    """
    A declarative bundle of capabilities that can be added to an Agent.

    AgentPlugin groups the tools, State requirements, hooks, and system-prompt instructions needed to realize one
    reusable Agent capability. Plugins do not mutate or wrap an Agent themselves. Pass them to `apply_plugins`,
    which validates all contributions and returns one cloned Agent while leaving the source Agent unchanged.

    Plugin contributions are applied in declaration order. Tools and hooks are appended after the Agent's existing
    configuration. Identical State entries are shared, while incompatible State definitions, duplicate plugin names,
    and duplicate known tool names raise an error. Prompt instructions are appended to the existing system prompt in
    clearly delimited plugin sections.

    :param name: Non-empty name identifying the plugin during composition and in prompt-section headings. Names must
        be unique within one `apply_plugins` call.
    :param tools: Tools or Toolsets contributed by the plugin.
    :param state_schema: Agent State entries required by the plugin. Each key maps to the standard Agent State
        definition containing a `type` and, optionally, a merge `handler`.
    :param hooks: Hooks contributed at each Agent hook point. Hooks are appended after existing hooks and after hooks
        from earlier plugins at the same hook point.
    :param prompt_instructions: Optional system-prompt fragment describing how the Agent should use the capability.
        The fragment must not contain Jinja message blocks; `apply_plugins` places it correctly inside a plain
        prompt or an existing system-message block.
    """

    name: str
    tools: ToolsType | None = None
    state_schema: dict[str, dict[str, Any]] = field(default_factory=dict)
    hooks: dict[HookPoint, list[Hook]] = field(default_factory=dict)
    prompt_instructions: str | None = None


def _normalize_tools(tools: ToolsType | None) -> list[Tool | Toolset]:
    """
    Normalize tools into a list while preserving Toolsets.

    :param tools: Tools to normalize. Accepts a Toolset, a sequence of Tools and Toolsets, or None.
    :returns: A new list containing the supplied Tools and Toolsets in their original order. A single Toolset remains
        intact as one list item.
    :raises TypeError: If a sequence contains an item that is neither a Tool nor a Toolset.
    """
    if tools is None:
        return []
    if isinstance(tools, Toolset):
        return [tools]

    items = list(tools)
    if not all(isinstance(item, (Tool, Toolset)) for item in items):
        msg = "AgentPlugin tools must contain only Tool or Toolset instances."
        raise TypeError(msg)
    return items


def _merge_prompt(system_prompt: str | None, plugins: Sequence[AgentPlugin]) -> str | None:
    """
    Append plugin instructions to a plain prompt or inside its single system-message block.

    :param system_prompt: Existing Agent system prompt, either plain text, one Jinja system-message block, or None.
    :param plugins: Plugins whose non-empty prompt instructions should be appended in declaration order.
    :returns: The combined system prompt, or the original None when no prompt or instructions are present.
    :raises ValueError: If plugin instructions contain Jinja message blocks, or if an existing message-block prompt
        does not contain exactly one closing block.
    """
    sections: list[str] = []
    for plugin in plugins:
        instructions = (plugin.prompt_instructions or "").strip()
        if not instructions:
            continue
        if _MESSAGE_BLOCK_START.search(instructions) or _MESSAGE_BLOCK_END.search(instructions):
            msg = f"Plugin '{plugin.name}' prompt instructions must not contain Jinja message blocks."
            raise ValueError(msg)
        sections.append(f"## Plugin: {plugin.name}\n{instructions}")

    if not sections:
        return system_prompt

    addition = "\n\n" + "\n\n".join(sections)
    if system_prompt is None:
        return addition.lstrip()

    if not _MESSAGE_BLOCK_START.search(system_prompt):
        return system_prompt.rstrip() + addition

    end_blocks = list(_MESSAGE_BLOCK_END.finditer(system_prompt))
    if len(end_blocks) != 1:
        msg = "The Agent system prompt must contain exactly one complete Jinja message block."
        raise ValueError(msg)
    end = end_blocks[0]
    return system_prompt[: end.start()].rstrip() + addition + "\n" + system_prompt[end.start() :]


def apply_plugins(agent: Agent, plugins: Sequence[AgentPlugin]) -> Agent:
    """
    Clone an Agent with declarative plugin contributions.

    The input Agent is never modified. Existing tools, state entries, hooks, and prompt instructions are preserved,
    with plugin contributions appended in declaration order. Identical state entries are shared; incompatible state
    definitions and duplicate plugin or known tool names fail before the Agent is cloned.

    :param agent: Agent to extend.
    :param plugins: Plugins to apply in declaration order.
    :returns: A cloned Agent containing all plugin contributions.
    """
    # Validate plugin names and collect them into a list for repeated iteration
    plugin_list = list(plugins)
    names: set[str] = set()
    for plugin in plugin_list:
        if not plugin.name.strip():
            msg = "AgentPlugin names must not be empty."
            raise ValueError(msg)
        if plugin.name in names:
            msg = f"Duplicate AgentPlugin name: '{plugin.name}'."
            raise ValueError(msg)
        names.add(plugin.name)

    # Merge tools while checking for duplicate known tool names
    merged_tools = _normalize_tools(tools=agent.tools)
    tool_owners: dict[str, str] = {}
    for tool in flatten_tools_or_toolsets(tools=merged_tools):
        tool_name = tool.name
        if tool_name in tool_owners:
            msg = f"The Agent already contains duplicate known tool name '{tool_name}'."
            raise ValueError(msg)
        tool_owners[tool_name] = "the Agent"
    for plugin in plugin_list:
        plugin_tools = _normalize_tools(tools=plugin.tools)
        for tool in flatten_tools_or_toolsets(tools=plugin_tools):
            tool_name = tool.name
            if owner := tool_owners.get(tool_name):
                msg = (
                    f"Plugin '{plugin.name}' contributes duplicate tool name '{tool_name}', "
                    f"already provided by {owner}."
                )
                raise ValueError(msg)
            tool_owners[tool_name] = f"plugin '{plugin.name}'"
        merged_tools.extend(plugin_tools)

    # Merge state schema while checking for incompatible definitions
    merged_schema = {key: dict(config) for key, config in agent.state_schema.items()}
    state_owners = dict.fromkeys(merged_schema, "the Agent")
    for plugin in plugin_list:
        for key, config in plugin.state_schema.items():
            if key not in merged_schema:
                merged_schema[key] = dict(config)
                state_owners[key] = f"plugin '{plugin.name}'"
            elif merged_schema[key] != config:
                msg = (
                    f"Plugin '{plugin.name}' contributes an incompatible definition for state key '{key}', "
                    f"already provided by {state_owners[key]}."
                )
                raise ValueError(msg)

    # Merge hooks by appending plugin hooks after existing hooks at each hook point
    merged_hooks = {hook_point: list(hooks) for hook_point, hooks in agent.hooks.items()}
    for plugin in plugin_list:
        for hook_point, hooks in plugin.hooks.items():
            merged_hooks.setdefault(hook_point, []).extend(hooks)

    # Merge system prompt with plugin instructions
    system_prompt = _merge_prompt(system_prompt=agent.system_prompt, plugins=plugin_list)

    # Clone the Agent with the merged configuration
    return agent.clone(tools=merged_tools, state_schema=merged_schema, hooks=merged_hooks, system_prompt=system_prompt)
