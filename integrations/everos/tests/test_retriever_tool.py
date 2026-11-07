# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import MagicMock

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.memory_stores.everos import EverOSMemoryStore
from haystack_integrations.tools.everos import EverOSMemoryRetrieverTool


def test_retriever_tool_formats_memories():
    store = MagicMock(spec=EverOSMemoryStore)
    store.search_memories.return_value = [
        ChatMessage.from_system("Alice prefers concise answers."),
        ChatMessage.from_system("Alice works on developer tooling."),
    ]
    tool = EverOSMemoryRetrieverTool(memory_store=store, top_k=6, method="hybrid", include_profile=True)

    result = tool.retrieve("What does Alice prefer?", user_id="alice", top_k=2)

    assert "Alice prefers concise answers" in result
    assert "Alice works on developer tooling" in result
    store.search_memories.assert_called_once_with(
        query="What does Alice prefer?",
        top_k=2,
        method="hybrid",
        include_profile=True,
        user_id="alice",
        agent_id=None,
        app_id="default",
        project_id="default",
        session_id=None,
    )


def test_retriever_tool_returns_no_memories_message():
    store = MagicMock(spec=EverOSMemoryStore)
    store.search_memories.return_value = []
    tool = EverOSMemoryRetrieverTool(memory_store=store)
    assert tool.retrieve("query", user_id="alice") == "No memories found."


def test_retriever_tool_round_trip_serialization(configured_store):
    tool = EverOSMemoryRetrieverTool(memory_store=configured_store, top_k=8, method="keyword", include_profile=True)
    restored = EverOSMemoryRetrieverTool.from_dict(tool.to_dict())
    assert isinstance(restored.memory_store, EverOSMemoryStore)
    assert restored.memory_store.to_dict() == configured_store.to_dict()
    assert restored.top_k == 8
    assert restored.method == "keyword"
    assert restored.include_profile is True


def test_retriever_tool_warm_up_is_idempotent():
    store = MagicMock(spec=EverOSMemoryStore)
    tool = EverOSMemoryRetrieverTool(memory_store=store)
    tool.warm_up()
    tool.warm_up()
    store.warm_up.assert_called_once()


def test_agent_injects_memory_scope_without_network():
    class ScriptedGenerator:
        def __init__(self):
            self.calls = 0

        def run(self, messages, tools=None, **kwargs):  # noqa: ARG002 - Agent generator protocol
            self.calls += 1
            if self.calls == 1:
                return {
                    "replies": [
                        ChatMessage.from_assistant(
                            tool_calls=[
                                ToolCall(tool_name="retrieve_memories", arguments={"query": "preferences"}, id="call-1")
                            ]
                        )
                    ]
                }
            return {"replies": [ChatMessage.from_assistant("Done.")]}

    store = MagicMock(spec=EverOSMemoryStore)
    store.search_memories.return_value = [ChatMessage.from_system("Prefers concise replies.")]
    scope = {"user_id": "test-user", "app_id": "test-app", "project_id": "test-project", "session_id": "test-session"}
    tool = EverOSMemoryRetrieverTool(memory_store=store, inputs_from_state={key: key for key in scope})
    agent = Agent(chat_generator=ScriptedGenerator(), tools=[tool], state_schema={key: {"type": str} for key in scope})
    result = agent.run(messages=[ChatMessage.from_user("Recall preferences")], **scope)
    assert result["last_message"].text == "Done."
    for key, value in scope.items():
        assert store.search_memories.call_args.kwargs[key] == value
    assert "session_id" not in tool.parameters["properties"]


def test_retriever_serialization_does_not_mutate_input():
    tool = EverOSMemoryRetrieverTool(memory_store=EverOSMemoryStore(), inputs_from_state={"session": "session_id"})
    data = tool.to_dict()
    original = deepcopy(data)
    for _ in range(2):
        restored = EverOSMemoryRetrieverTool.from_dict(data)
        assert restored.inputs_from_state == {"session": "session_id"}
    assert data == original
