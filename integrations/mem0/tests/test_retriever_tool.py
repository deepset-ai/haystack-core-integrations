# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore
from haystack_integrations.tools.mem0.retriever_tool import Mem0MemoryRetrieverTool


@pytest.fixture
def store(monkeypatch, mock_mem0_client):  # noqa: ARG001
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    return Mem0MemoryStore()


class TestMem0MemoryRetrieverTool:
    def test_init_default(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        assert isinstance(tool, Tool)
        assert tool.top_k == 5
        assert tool.name == "retrieve_memories"
        assert tool.description == (
            "Search long-term memories relevant to a query. "
            "Use this tool whenever you need to recall information from past conversations or stored facts."
        )
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to find relevant memories."},
                "top_k": {"type": "integer", "description": "Maximum number of memories to return."},
            },
            "required": ["query"],
        }
        assert tool.inputs_from_state == {"user_id": "user_id"}

    def test_init_custom(self, store):
        tool = Mem0MemoryRetrieverTool(
            memory_store=store,
            name="my_tool",
            description="Custom desc",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Custom query."},
                    "top_k": {"type": "integer", "description": "Custom limit."},
                },
                "required": ["query", "top_k"],
            },
            inputs_from_state={"test_id": "user_id"},
        )
        assert tool.name == "my_tool"
        assert tool.description == "Custom desc"
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Custom query."},
                "top_k": {"type": "integer", "description": "Custom limit."},
            },
            "required": ["query", "top_k"],
        }
        assert tool.inputs_from_state == {"test_id": "user_id"}

    def test_default_dicts_are_copied(self, store):
        first = Mem0MemoryRetrieverTool(memory_store=store)
        second = Mem0MemoryRetrieverTool(memory_store=store)

        first.inputs_from_state["session_id"] = "user_id"
        first.parameters["properties"]["extra"] = {"type": "string"}

        assert second.inputs_from_state == {"user_id": "user_id"}
        assert "extra" not in second.parameters["properties"]

    def test_invoke_passes_ids_to_store(self, store):
        store.search_memories = Mock(return_value=[ChatMessage.from_system("remembered fact")])
        tool = Mem0MemoryRetrieverTool(memory_store=store, top_k=3)
        tool.invoke(query="what do I like?", user_id="alice", run_id="r1", agent_id="a1", app_id="app1")
        store.search_memories.assert_called_once_with(
            query="what do I like?",
            top_k=3,
            user_id="alice",
            run_id="r1",
            agent_id="a1",
            app_id="app1",
        )

    def test_invoke_returns_string(self, store):
        store.search_memories = Mock(return_value=[ChatMessage.from_system("I like Python")])
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        result = tool.invoke(query="programming", user_id="alice")
        assert isinstance(result, str)
        assert "I like Python" in result

    def test_invoke_returns_no_memories_message(self, store):
        store.search_memories = Mock(return_value=[])
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        result = tool.invoke(query="nothing", user_id="alice")
        assert "No memories found" in result

    def test_warm_up(self, store):
        store.warm_up = Mock()
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        tool.warm_up()
        store.warm_up.assert_called_once()

    def test_warm_up_is_idempotent(self, store):
        store.warm_up = Mock()
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        tool.warm_up()
        tool.warm_up()
        tool.warm_up()
        store.warm_up.assert_called_once()

    def test_to_dict(self):
        store = Mem0MemoryStore()
        tool = Mem0MemoryRetrieverTool(memory_store=store, top_k=4)
        tool_dict = tool.to_dict()
        assert tool_dict == {
            "type": "haystack_integrations.tools.mem0.retriever_tool.Mem0MemoryRetrieverTool",
            "data": {
                "memory_store": {
                    "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
                    "init_parameters": {"api_key": {"type": "env_var", "env_vars": ["MEM0_API_KEY"], "strict": True}},
                },
                "top_k": 4,
                "name": "retrieve_memories",
                "description": (
                    "Search long-term memories relevant to a query. Use this tool whenever you need to recall "
                    "information from past conversations or stored facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to find relevant memories."},
                        "top_k": {"type": "integer", "description": "Maximum number of memories to return."},
                    },
                    "required": ["query"],
                },
                "inputs_from_state": {"user_id": "user_id"},
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.tools.mem0.retriever_tool.Mem0MemoryRetrieverTool",
            "data": {
                "memory_store": {
                    "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
                    "init_parameters": {"api_key": {"type": "env_var", "env_vars": ["MEM0_API_KEY"], "strict": True}},
                },
                "top_k": 4,
                "name": "retrieve_memories",
                "description": (
                    "Search long-term memories relevant to a query. Use this tool whenever you need to recall "
                    "information from past conversations or stored facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to find relevant memories."},
                        "top_k": {"type": "integer", "description": "Maximum number of memories to return."},
                    },
                    "required": ["query"],
                },
                "inputs_from_state": {"user_id": "user_id"},
            },
        }
        tool = Mem0MemoryRetrieverTool.from_dict(data)
        assert isinstance(tool, Mem0MemoryRetrieverTool)
        assert tool.top_k == 4
        assert isinstance(tool.memory_store, Mem0MemoryStore)
