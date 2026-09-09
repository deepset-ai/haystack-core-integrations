# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from haystack.dataclasses import ChatMessage

# One tool name, or a group of them sharing a question: "how often did the run search for anything at all?"
ToolNames = str | tuple[str, ...] | list[str]


@dataclass
class ToolRunStats:
    """
    The tool calls one agent run made, and what they add up to.

    Which tools an agent has is up to the agent, so nothing here names one. A harness asks its questions by
    passing the names it cares about, which lets the same statistics score any agent's run.

    :param calls: Every call the run made, in the order it made them, as `(tool name, the arguments it passed)`:

            [("list_metadata_fields", {}), ("search_documents", {"query": "CRISPR", "filters": None})]

    :param errors: The calls that came back an error, as `(tool name, what it said)`:

            [("get_metadata_field_values", "field 'nope' does not exist in the store")]
    """

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @staticmethod
    def _named(tools: ToolNames) -> set[str]:
        """
        Normalize one tool name or a group of them to a set.

        :param tools: A tool name, or several of them.
        :returns: The names as a set.
        """
        return {tools} if isinstance(tools, str) else set(tools)

    def calls_to(self, tools: ToolNames) -> int:
        """
        Count the calls made to one tool, or to any of a group of them.

        :param tools: A tool name, or several of them.
        :returns: How many calls the run made to them.
        """
        wanted = self._named(tools=tools)
        return sum(1 for name, _ in self.calls if name in wanted)

    def calls_with_argument(self, tools: ToolNames, argument: str) -> int:
        """
        Count the calls that passed a non-empty value for one argument, such as a retrieval carrying a filter.

        :param tools: A tool name, or several of them.
        :param argument: The argument that has to carry a value.
        :returns: How many of those calls passed something for it.
        """
        wanted = self._named(tools=tools)
        return sum(1 for name, arguments in self.calls if name in wanted and arguments.get(argument))

    def called_before(self, tools: ToolNames, other: ToolNames) -> bool:
        """
        Whether the run reached for one tool before it reached for another.

        :param tools: The tool, or tools, that should come first.
        :param other: The tool, or tools, they should come before.
        :returns: True when one of `tools` was called and none of `other` was called before it. False when
            `other` came first, and when neither was called at all.
        """
        wanted, after = self._named(tools=tools), self._named(tools=other)
        for name, _ in self.calls:
            if name in wanted:
                return True
            if name in after:
                return False
        return False


def extract_tool_run_stats(messages: list[ChatMessage]) -> ToolRunStats:
    """
    Extract tool calls and error results from an agent run.

    :param messages: The messages returned by `agent.run(...)`.
    :returns: The extracted statistics.
    """
    return ToolRunStats(
        calls=[(call.tool_name, call.arguments or {}) for message in messages for call in message.tool_calls],
        errors=[
            (result.origin.tool_name, str(result.result))
            for message in messages
            for result in message.tool_call_results
            if result.error
        ],
    )
