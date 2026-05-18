# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
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
        assert "app_id" not in props

    def test_default_inputs_from_state(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        assert tool.inputs_from_state == {"user_id": "user_id"}

    def test_defaults_exposed_in_signature(self):
        signature = inspect.signature(Mem0MemoryRetrieverTool.__init__)
        assert signature.parameters["inputs_from_state"].default == {"user_id": "user_id"}
        assert "query" in signature.parameters["parameters"].default["properties"]

    def test_default_dicts_are_copied(self, store):
        first = Mem0MemoryRetrieverTool(memory_store=store)
        second = Mem0MemoryRetrieverTool(memory_store=store)

        first.inputs_from_state["session_id"] = "user_id"
        first.parameters["properties"]["extra"] = {"type": "string"}

        assert second.inputs_from_state == {"user_id": "user_id"}
        assert "extra" not in second.parameters["properties"]

    def test_custom_inputs_from_state(self, store):
        custom = {
            "session_id": "user_id",
            "run_state": "run_id",
            "agent_state": "agent_id",
            "app_state": "app_id",
        }
        tool = Mem0MemoryRetrieverTool(memory_store=store, inputs_from_state=custom)
        assert tool.inputs_from_state == custom

    def test_custom_parameters(self, store):
        custom = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Custom query."},
                "top_k": {"type": "integer", "description": "Custom limit."},
            },
            "required": ["query", "top_k"],
        }
        tool = Mem0MemoryRetrieverTool(memory_store=store, parameters=custom)
        assert tool.parameters == custom
        assert tool.parameters is not custom

    def test_function_is_public_retrieve_method(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        assert tool.function == tool.retrieve

    def test_query_is_required(self, store):
        tool = Mem0MemoryRetrieverTool(memory_store=store)
        assert "query" in tool.parameters.get("required", [])

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
        assert restored.parameters == tool.parameters
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
        restored.memory_store.search_memories.assert_called_once_with(
            query="hobbies",
            top_k=5,
            user_id="alice",
            run_id=None,
            agent_id=None,
            app_id=None,
        )


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
        assert "run_id" not in props
        assert "agent_id" not in props
        assert "app_id" not in props

    def test_default_inputs_from_state(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        assert tool.inputs_from_state == {"user_id": "user_id"}

    def test_defaults_exposed_in_signature(self):
        signature = inspect.signature(Mem0MemoryWriterTool.__init__)
        assert signature.parameters["inputs_from_state"].default == {"user_id": "user_id"}
        assert "text" in signature.parameters["parameters"].default["properties"]
        assert "infer" in signature.parameters["parameters"].default["properties"]

    def test_default_dicts_are_copied(self, store):
        first = Mem0MemoryWriterTool(memory_store=store)
        second = Mem0MemoryWriterTool(memory_store=store)

        first.inputs_from_state["session_id"] = "user_id"
        first.parameters["properties"]["extra"] = {"type": "string"}

        assert second.inputs_from_state == {"user_id": "user_id"}
        assert "extra" not in second.parameters["properties"]

    def test_custom_inputs_from_state(self, store):
        custom = {
            "session_id": "user_id",
            "run_state": "run_id",
            "agent_state": "agent_id",
            "app_state": "app_id",
        }
        tool = Mem0MemoryWriterTool(memory_store=store, inputs_from_state=custom)
        assert tool.inputs_from_state == custom

    def test_custom_parameters(self, store):
        custom = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Custom memory."},
                "infer": {"type": "boolean", "description": "Custom inference toggle."},
            },
            "required": ["text"],
        }
        tool = Mem0MemoryWriterTool(memory_store=store, parameters=custom)
        assert tool.parameters == custom
        assert tool.parameters is not custom

    def test_function_is_public_store_method(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        assert tool.function == tool.store

    def test_store_infer_default_false(self):
        signature = inspect.signature(Mem0MemoryWriterTool.store)
        assert signature.parameters["infer"].default is False

    def test_text_is_required(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        assert "text" in tool.parameters.get("required", [])
        assert "infer" not in tool.parameters.get("required", [])

    def test_invoke_passes_ids_to_store(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.invoke(text="I enjoy hiking", user_id="alice", run_id="r1", agent_id="a1", app_id="app1")
        store.add_memories.assert_called_once_with(
            messages=[ChatMessage.from_user("I enjoy hiking")],
            user_id="alice",
            run_id="r1",
            agent_id="a1",
            app_id="app1",
            infer=False,
        )

    def test_configured_infer_passed_to_store(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.invoke(text="I enjoy hiking", user_id="alice", infer=True)
        store.add_memories.assert_called_once_with(
            messages=[ChatMessage.from_user("I enjoy hiking")],
            user_id="alice",
            run_id=None,
            agent_id=None,
            app_id=None,
            infer=True,
        )

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
        assert restored.parameters == tool.parameters
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
        restored.memory_store.add_memories.assert_called_once_with(
            messages=[ChatMessage.from_user("I enjoy hiking")],
            user_id="alice",
            run_id=None,
            agent_id=None,
            app_id=None,
            infer=False,
        )
