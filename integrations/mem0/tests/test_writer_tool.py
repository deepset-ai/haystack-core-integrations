# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore
from haystack_integrations.tools.mem0.writer_tool import Mem0MemoryWriterTool


@pytest.fixture
def store(monkeypatch, mock_mem0_client):  # noqa: ARG001
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    return Mem0MemoryStore()


class TestMem0MemoryWriterTool:
    def test_init_default(self, store):
        tool = Mem0MemoryWriterTool(memory_store=store)
        assert isinstance(tool, Tool)
        assert tool.name == "store_memory"
        assert tool.description == (
            "Store a piece of information as a long-term memory. "
            "Use this tool to persist important facts, preferences, or context for future conversations."
        )
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The information to store as a memory."},
                "infer": {
                    "type": "boolean",
                    "description": (
                        "If true, Mem0 extracts memories from the text. If false, Mem0 stores the text as-is."
                    ),
                    "default": False,
                },
            },
            "required": ["text"],
        }
        assert tool.inputs_from_state == {"user_id": "user_id"}

    def test_init_custom(self, store):
        tool = Mem0MemoryWriterTool(
            memory_store=store,
            name="my_tool",
            description="Custom desc",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Custom memory."},
                    "infer": {"type": "boolean", "description": "Custom inference toggle."},
                },
                "required": ["text"],
            },
            inputs_from_state={"test_id": "user_id"},
        )
        assert tool.name == "my_tool"
        assert tool.description == "Custom desc"
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Custom memory."},
                "infer": {"type": "boolean", "description": "Custom inference toggle."},
            },
            "required": ["text"],
        }
        assert tool.inputs_from_state == {"test_id": "user_id"}

    def test_default_dicts_are_copied(self, store):
        first = Mem0MemoryWriterTool(memory_store=store)
        second = Mem0MemoryWriterTool(memory_store=store)

        first.inputs_from_state["session_id"] = "user_id"
        first.parameters["properties"]["extra"] = {"type": "string"}

        assert second.inputs_from_state == {"user_id": "user_id"}
        assert "extra" not in second.parameters["properties"]

    def test_invoke_passes_ids_and_infer_to_store(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.invoke(text="I enjoy hiking", infer=True, user_id="alice", run_id="r1", agent_id="a1", app_id="app1")
        store.add_memories.assert_called_once_with(
            messages=[ChatMessage.from_user("I enjoy hiking")],
            user_id="alice",
            run_id="r1",
            agent_id="a1",
            app_id="app1",
            infer=True,
        )

    def test_invoke_uses_infer_default(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.invoke(text="I enjoy hiking", user_id="alice")
        store.add_memories.assert_called_once_with(
            messages=[ChatMessage.from_user("I enjoy hiking")],
            user_id="alice",
            run_id=None,
            agent_id=None,
            app_id=None,
            infer=False,
        )

    def test_invoke_returns_count_string(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = Mem0MemoryWriterTool(memory_store=store)
        result = tool.invoke(text="I enjoy hiking", user_id="alice")
        assert isinstance(result, str)
        assert "1" in result

    def test_warm_up(self, store):
        store.warm_up = Mock()
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.warm_up()
        store.warm_up.assert_called_once()

    def test_warm_up_is_idempotent(self, store):
        store.warm_up = Mock()
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool.warm_up()
        tool.warm_up()
        store.warm_up.assert_called_once()

    def test_to_dict(self):
        store = Mem0MemoryStore()
        tool = Mem0MemoryWriterTool(memory_store=store)
        tool_dict = tool.to_dict()
        assert tool_dict == {
            "type": "haystack_integrations.tools.mem0.writer_tool.Mem0MemoryWriterTool",
            "data": {
                "memory_store": {
                    "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
                    "init_parameters": {"api_key": {"type": "env_var", "env_vars": ["MEM0_API_KEY"], "strict": True}},
                },
                "name": "store_memory",
                "description": (
                    "Store a piece of information as a long-term memory. Use this tool to persist important facts, "
                    "preferences, or context for future conversations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The information to store as a memory."},
                        "infer": {
                            "type": "boolean",
                            "description": (
                                "If true, Mem0 extracts memories from the text. If false, Mem0 stores the text as-is."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["text"],
                },
                "inputs_from_state": {"user_id": "user_id"},
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.tools.mem0.writer_tool.Mem0MemoryWriterTool",
            "data": {
                "memory_store": {
                    "type": "haystack_integrations.memory_stores.mem0.memory_store.Mem0MemoryStore",
                    "init_parameters": {"api_key": {"type": "env_var", "env_vars": ["MEM0_API_KEY"], "strict": True}},
                },
                "name": "store_memory",
                "description": (
                    "Store a piece of information as a long-term memory. Use this tool to persist important facts, "
                    "preferences, or context for future conversations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The information to store as a memory."},
                        "infer": {
                            "type": "boolean",
                            "description": (
                                "If true, Mem0 extracts memories from the text. If false, Mem0 stores the text as-is."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["text"],
                },
                "inputs_from_state": {"user_id": "user_id"},
            },
        }
        tool = Mem0MemoryWriterTool.from_dict(data)
        assert isinstance(tool, Mem0MemoryWriterTool)
        assert isinstance(tool.memory_store, Mem0MemoryStore)
