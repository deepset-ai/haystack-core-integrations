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
from haystack_integrations.tools.mem0.writer_tool import Mem0MemoryWriterTool, mem0_memory_writer_tool


@pytest.fixture
def store(monkeypatch, mock_mem0_client):  # noqa: ARG001
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    return Mem0MemoryStore()


class TestMem0MemoryRetrieverTool:
    def test_returns_tool_instance(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        assert isinstance(tool, Tool)

    def test_default_name(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        assert tool.name == "retrieve_memories"

    def test_custom_name_and_description(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store, name="my_tool", description="Custom desc")
        assert tool.name == "my_tool"
        assert tool.description == "Custom desc"

    def test_ids_not_exposed_to_llm(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        props = tool.parameters.get("properties", {})
        assert "user_id" not in props
        assert "run_id" not in props
        assert "agent_id" not in props

    def test_default_inputs_from_state(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        assert tool.inputs_from_state == {"user_id": "user_id", "run_id": "run_id", "agent_id": "agent_id"}

    def test_custom_inputs_from_state(self, store):
        custom = {"session_id": "user_id"}  # map state key "session_id" to param "user_id"
        tool = Mem0MemoryRetrieverTool(memory_store=store, inputs_from_state=custom)
        assert tool.inputs_from_state == custom

    def test_query_is_required(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        assert "query" in tool.parameters.get("required", [])

    def test_invoke_passes_ids_to_store(self, store):
        store.search_memories = Mock(return_value=[ChatMessage.from_system("remembered fact")])
        tool = Mem0MemoryRetrieverTool(memory_store=store, top_k=3)
        tool.invoke(query="what do I like?", user_id="alice", run_id="r1", agent_id="a1")
        kwargs = store.search_memories.call_args[1]
        assert kwargs["user_id"] == "alice"
        assert kwargs["run_id"] == "r1"
        assert kwargs["agent_id"] == "a1"
        assert kwargs["top_k"] == 3

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

    def test_warm_up_calls_store(self, store):
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

    def test_serialization_roundtrip(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MY_MEM0_KEY", "test-key")
        store = Mem0MemoryStore(api_key=Secret.from_env_var("MY_MEM0_KEY"))
        tool = Mem0MemoryRetrieverTool(memory_store=store, top_k=3)
        serialized = serialize_tools_or_toolset([tool])
        assert isinstance(serialized, list)
        assert serialized[0]["type"].endswith("Mem0MemoryRetrieverTool")

        data = {"tools": serialized}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        assert isinstance(restored, Mem0MemoryRetrieverTool)
        assert restored.name == tool.name
        assert restored.top_k == 3
        assert restored.inputs_from_state == tool.inputs_from_state
        assert isinstance(restored.memory_store, Mem0MemoryStore)

    def test_serialization_roundtrip_custom_inputs_from_state(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        custom = {"session_id": "user_id"}  # map state key "session_id" to param "user_id"
        tool = Mem0MemoryRetrieverTool(memory_store=store, inputs_from_state=custom)

        data = {"tools": serialize_tools_or_toolset([tool])}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        assert restored.inputs_from_state == custom

    def test_serialized_tool_invokes_correctly_after_roundtrip(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        tool = Mem0MemoryRetrieverTool(memory_store=store)

        data = {"tools": serialize_tools_or_toolset([tool])}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        restored.memory_store.search_memories = Mock(return_value=[ChatMessage.from_system("Python fan")])
        result = restored.invoke(query="hobbies", user_id="alice")
        assert "Python fan" in result
        assert restored.memory_store.search_memories.call_args[1]["user_id"] == "alice"


class TestMem0MemoryWriterTool:
    def test_returns_tool_instance(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        assert isinstance(tool, Tool)

    def test_default_name(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        assert tool.name == "store_memory"

    def test_ids_not_exposed_to_llm(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        props = tool.parameters.get("properties", {})
        assert "user_id" not in props

    def test_default_inputs_from_state(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        assert tool.inputs_from_state == {"user_id": "user_id", "run_id": "run_id", "agent_id": "agent_id"}

    def test_custom_inputs_from_state(self, store):
        custom = {"session_id": "user_id"}  # map state key "session_id" to param "user_id"
        tool = Mem0MemoryWriterTool(memory_store=store, inputs_from_state=custom)
        assert tool.inputs_from_state == custom

    def test_text_is_required(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        assert "text" in tool.parameters.get("required", [])

    def test_invoke_passes_ids_to_store(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.invoke(text="I enjoy hiking", user_id="alice", run_id="r1", agent_id="a1")
        kwargs = store.add_memories.call_args[1]
        assert kwargs["user_id"] == "alice"
        assert kwargs["run_id"] == "r1"
        assert kwargs["agent_id"] == "a1"

    def test_invoke_returns_count_string(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store)
        result = tool.invoke(text="I enjoy hiking", user_id="alice")
        assert isinstance(result, str)
        assert "1" in result

    def test_warm_up_calls_store(self, store):
        store.warm_up = Mock()
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.warm_up()
        store.warm_up.assert_called_once()

    def test_warm_up_is_idempotent(self, store):
        store.warm_up = Mock()
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.warm_up()
        tool.warm_up()
        tool.warm_up()
        store.warm_up.assert_called_once()

    def test_serialization_roundtrip(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MY_MEM0_KEY", "test-key")
        store = Mem0MemoryStore(api_key=Secret.from_env_var("MY_MEM0_KEY"))
        tool = Mem0MemoryWriterTool(memory_store=store)
        serialized = serialize_tools_or_toolset([tool])
        assert isinstance(serialized, list)
        assert serialized[0]["type"].endswith("Mem0MemoryWriterTool")

        data = {"tools": serialized}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        assert isinstance(restored, Mem0MemoryWriterTool)
        assert restored.name == tool.name
        assert restored.inputs_from_state == tool.inputs_from_state
        assert isinstance(restored.memory_store, Mem0MemoryStore)

    def test_serialization_roundtrip_custom_inputs_from_state(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        custom = {"session_id": "user_id"}  # map state key "session_id" to param "user_id"
        tool = Mem0MemoryWriterTool(memory_store=store, inputs_from_state=custom)

        data = {"tools": serialize_tools_or_toolset([tool])}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        assert restored.inputs_from_state == custom

    def test_serialized_tool_invokes_correctly_after_roundtrip(self, monkeypatch, mock_mem0_client):  # noqa: ARG002
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        store = Mem0MemoryStore()
        tool = Mem0MemoryWriterTool(memory_store=store)

        data = {"tools": serialize_tools_or_toolset([tool])}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        restored.memory_store.add_memories = Mock(return_value=[{"memory_id": "m1"}])
        result = restored.invoke(text="I enjoy hiking", user_id="alice")
        assert "1" in result
        assert restored.memory_store.add_memories.call_args[1]["user_id"] == "alice"


class TestMem0MemoryWriterToolFunction:
    def test_is_tool_instance(self):
        assert isinstance(mem0_memory_writer_tool, Tool)

    def test_name(self):
        assert mem0_memory_writer_tool.name == "mem0_memory_writer_tool"

    def test_inputs_from_state(self):
        assert mem0_memory_writer_tool.inputs_from_state == {
            "user_id": "user_id",
            "run_id": "run_id",
            "agent_id": "agent_id",
        }

    def test_text_is_required(self):
        assert "text" in mem0_memory_writer_tool.parameters.get("required", [])

    def test_invoke(self, monkeypatch, mock_mem0_client):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        mock_mem0_client.add.return_value = {"results": [{"id": "m1", "data": {"memory": "hiking"}}]}
        result = mem0_memory_writer_tool.invoke(text="I enjoy hiking", user_id="alice")
        assert isinstance(result, str)

    def test_serialization_roundtrip(self):
        serialized = serialize_tools_or_toolset([mem0_memory_writer_tool])
        assert isinstance(serialized, list)
        assert serialized[0]["type"].endswith("Tool")

        data = {"tools": serialized}
        deserialize_tools_or_toolset_inplace(data)
        restored = data["tools"][0]

        assert isinstance(restored, Tool)
        assert restored.name == mem0_memory_writer_tool.name
        assert restored.inputs_from_state == mem0_memory_writer_tool.inputs_from_state
