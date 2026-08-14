# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from dbos import DBOS, DBOSConfig
from haystack import component
from haystack.core.serialization import allow_deserialization_module
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

# The fakes below live in this module, so deserialization tests need it on the trusted-module allowlist.
allow_deserialization_module("tests.conftest")


@pytest.fixture
def dbos_app(tmp_path) -> Iterator[DBOS]:
    """A DBOS instance backed by a throwaway SQLite database. No service required."""
    DBOS.destroy(destroy_registry=True)
    config: DBOSConfig = {
        "name": "haystack-dbos-tests",
        "system_database_url": f"sqlite:///{tmp_path / 'dbos.sqlite'}",
        "run_admin_server": False,
    }
    instance = DBOS(config=config)
    yield instance
    DBOS.destroy(destroy_registry=True)


@component
class FakeChatGenerator:
    """A Chat Generator that replays a scripted list of replies and counts how often it was called."""

    def __init__(
        self,
        replies: list[ChatMessage] | None = None,
        on_call: Callable[[list[ChatMessage]], None] | None = None,
    ) -> None:
        self.replies = replies if replies is not None else [ChatMessage.from_assistant("done")]
        self.on_call = on_call
        self.calls: list[list[ChatMessage]] = []
        self.warmed_up = 0
        self.closed = 0

    # The generator accepts the full Chat Generator input set so the Agent treats it like a real one, but only
    # `messages` affects the scripted replies.
    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: Any = None,  # noqa: ARG002
        *,
        generation_kwargs: dict[str, Any] | None = None,  # noqa: ARG002
        tools: Any = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.calls.append(list(messages))
        if self.on_call is not None:
            self.on_call(messages)
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return {"replies": [self.replies[index]]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: Any = None,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        tools: Any = None,
    ) -> dict[str, Any]:
        return self.run(messages, streaming_callback, generation_kwargs=generation_kwargs, tools=tools)

    def warm_up(self) -> None:
        self.warmed_up += 1

    def close(self) -> None:
        self.closed += 1

    def to_dict(self) -> dict[str, Any]:
        return {"type": "tests.conftest.FakeChatGenerator", "init_parameters": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FakeChatGenerator":  # noqa: ARG003
        return cls()


@component
class ToollessChatGenerator:
    """A Chat Generator whose run method does not accept tools."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:  # noqa: ARG002
        return {"replies": [ChatMessage.from_assistant("no tools here")]}


def counting_tool(counter: dict[str, int], name: str = "record") -> Tool:
    """A Tool that increments `counter` every time it runs, so replay behaviour is observable."""

    def _run(value: str) -> str:
        counter[name] = counter.get(name, 0) + 1
        return f"{name}:{value}"

    return Tool(
        name=name,
        description=f"Record {name}",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        function=_run,
    )


def tool_call_reply(tool_name: str = "record", value: str = "x", call_id: str = "call-1") -> ChatMessage:
    """An assistant message requesting a single tool call."""
    return ChatMessage.from_assistant(
        tool_calls=[ToolCall(id=call_id, tool_name=tool_name, arguments={"value": value})]
    )
