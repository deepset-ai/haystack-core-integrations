# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.utils import Secret

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore
from haystack_integrations.tools.mem0.retriever_tool import Mem0MemoryRetrieverTool
from haystack_integrations.tools.mem0.writer_tool import Mem0MemoryWriterTool


@pytest.fixture
def store(monkeypatch, mock_mem0_client):  # noqa: ARG001
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    return Mem0MemoryStore()


class TestMem0MemoryRetrieverTool:
    def test_returns_tool_instance(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="u1")
        assert isinstance(tool, Tool)

    def test_default_name(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="u1")
        assert tool.name == "retrieve_memories"

    def test_custom_name_and_description(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="u1", name="my_tool", description="Custom desc")
        assert tool.name == "my_tool"
        assert tool.description == "Custom desc"

    def test_ids_not_exposed_to_llm(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="u1")
        props = tool.parameters.get("properties", {})
        assert "user_id" not in props
        assert "run_id" not in props
        assert "agent_id" not in props

    def test_query_is_required(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="u1")
        assert "query" in tool.parameters.get("required", [])

    def test_invoke_calls_store_with_bound_ids(self, store):
        store.search_memories = Mock(return_value=[ChatMessage.from_system("remembered fact")])
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="bound-user", run_id="r1", agent_id="a1", top_k=3)
        tool.invoke(query="what do I like?")
        kwargs = store.search_memories.call_args[1]
        assert kwargs["user_id"] == "bound-user"
        assert kwargs["run_id"] == "r1"
        assert kwargs["agent_id"] == "a1"
        assert kwargs["top_k"] == 3

    def test_invoke_returns_string(self, store):
        store.search_memories = Mock(return_value=[ChatMessage.from_system("I like Python")])
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="u1")
        result = tool.invoke(query="programming")
        assert isinstance(result, str)
        assert "I like Python" in result

    def test_invoke_returns_no_memories_message(self, store):
        store.search_memories = Mock(return_value=[])
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="u1")
        result = tool.invoke(query="nothing")
        assert "No memories found" in result

    def test_serialization_roundtrip(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MY_MEM0_KEY", "test-key")
        store = Mem0MemoryStore(api_key=Secret.from_env_var("MY_MEM0_KEY"))
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="alice", run_id="r1", agent_id="a1", top_k=3)
        serialized = serialize_tools_or_toolset([tool])
        assert isinstance(serialized, list)
        assert serialized[0]["type"].endswith("Mem0MemoryRetrieverTool")

        data = {"tools": serialized}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        assert isinstance(restored, Mem0MemoryRetrieverTool)
        assert restored.name == tool.name
        assert restored.user_id == "alice"
        assert restored.run_id == "r1"
        assert restored.agent_id == "a1"
        assert restored.top_k == 3
        assert isinstance(restored.memory_store, Mem0MemoryStore)

    def test_serialized_tool_invokes_correctly_after_roundtrip(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        tool = Mem0MemoryRetrieverTool(memory_store=store, user_id="alice")

        data = {"tools": serialize_tools_or_toolset([tool])}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        restored.memory_store.search_memories = Mock(return_value=[ChatMessage.from_system("Python fan")])
        result = restored.invoke(query="hobbies")
        assert "Python fan" in result
        assert restored.memory_store.search_memories.call_args[1]["user_id"] == "alice"


class TestMem0MemoryWriterTool:
    def test_returns_tool_instance(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store, user_id="u1")
        assert isinstance(tool, Tool)

    def test_default_name(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store, user_id="u1")
        assert tool.name == "store_memory"

    def test_ids_not_exposed_to_llm(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store, user_id="u1")
        props = tool.parameters.get("properties", {})
        assert "user_id" not in props

    def test_text_is_required(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store, user_id="u1")
        assert "text" in tool.parameters.get("required", [])

    def test_invoke_calls_store_with_bound_ids(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store, user_id="bound-user", run_id="r1", agent_id="a1")
        tool.invoke(text="I enjoy hiking")
        kwargs = store.add_memories.call_args[1]
        assert kwargs["user_id"] == "bound-user"
        assert kwargs["run_id"] == "r1"
        assert kwargs["agent_id"] == "a1"

    def test_invoke_returns_count_string(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store, user_id="u1")
        result = tool.invoke(text="I enjoy hiking")
        assert isinstance(result, str)
        assert "1" in result

    def test_serialization_roundtrip(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MY_MEM0_KEY", "test-key")
        store = Mem0MemoryStore(api_key=Secret.from_env_var("MY_MEM0_KEY"))
        tool = Mem0MemoryWriterTool(memory_store=store, user_id="alice", run_id="r1", agent_id="a1")
        serialized = serialize_tools_or_toolset([tool])
        assert isinstance(serialized, list)
        assert serialized[0]["type"].endswith("Mem0MemoryWriterTool")

        data = {"tools": serialized}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        assert isinstance(restored, Mem0MemoryWriterTool)
        assert restored.name == tool.name
        assert restored.user_id == "alice"
        assert restored.run_id == "r1"
        assert restored.agent_id == "a1"
        assert isinstance(restored.memory_store, Mem0MemoryStore)

    def test_serialized_tool_invokes_correctly_after_roundtrip(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        tool = Mem0MemoryWriterTool(memory_store=store, user_id="alice")

        data = {"tools": serialize_tools_or_toolset([tool])}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        restored.memory_store.add_memories = Mock(return_value=[{"memory_id": "m1"}])
        result = restored.invoke(text="I enjoy hiking")
        assert "1" in result
        assert restored.memory_store.add_memories.call_args[1]["user_id"] == "alice"
