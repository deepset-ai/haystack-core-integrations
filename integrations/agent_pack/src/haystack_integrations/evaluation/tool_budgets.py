# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .dataclasses import ToolNames, ToolRunStats

# The key an eval case uses to budget every tool it did not name.
ANY_TOOL = "*"

# What one uncovered tool may be called when the eval case names no `*` allowance of its own.
DEFAULT_TOOL_BUDGET = 5


def _group(tools: ToolNames) -> tuple[str, ...]:
    """
    Normalize one tool name or a group of them to the key a resolved budget is reported under.

    :param tools: A tool name, or several of them.
    :returns: The names as a tuple, in a fixed order so the same group is always the same key.
    """
    return (tools,) if isinstance(tools, str) else tuple(sorted(tools))


def resolve_tool_budgets(
    budgets: dict[ToolNames, int],
    tool_names: tuple[str, ...] | list[str],
    default: int = DEFAULT_TOOL_BUDGET,
) -> dict[tuple[str, ...], int]:
    """
    Give every tool an agent has an allowance, whether or not the eval case named it.

    A tool the eval case named keeps its group's allowance, which the whole group shares. Every other tool the
    agent has gets an allowance of its own: the `*` entry when the eval case sets one, and `default` otherwise.
    A tool that appears in a candidate configuration after the eval case was written is capped that way rather
    than running unbudgeted.

    :param budgets: What the eval case declared, as `{tool name or names: limit}`.
    :param tool_names: The tools the agent under evaluation actually has.
    :param default: The allowance for an uncovered tool when the eval case names no `*` entry.
    :returns: One entry per group, keyed by the tool names it covers.
    """
    # Named groups first, so the `*` allowance only reaches what they leave over.
    named = {_group(tools=tools): limit for tools, limit in budgets.items() if tools != ANY_TOOL}
    covered = {name for group in named for name in group}
    uncovered = {(name,): budgets.get(ANY_TOOL, default) for name in tool_names if name not in covered}
    return named | uncovered


def budgets_exceeded(
    stats: ToolRunStats, budgets: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], tuple[int, int]]:
    """
    Report the groups a run called more often than its eval case allows.

    :param stats: The tool calls the run made.
    :param budgets: Resolved allowances, as `resolve_tool_budgets` returns them.
    :returns: The groups that went over, as `{tool names: (calls made, allowance)}`.
    """
    spent = {group: stats.calls_to(tools=group) for group in budgets}
    return {group: (calls, budgets[group]) for group, calls in spent.items() if calls > budgets[group]}
