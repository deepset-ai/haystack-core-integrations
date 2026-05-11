# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.memory_stores.mem0.memory_store import Mem0MemoryStore
from haystack_integrations.memory_stores.mem0.tools import (
    create_mem0_memory_retriever_tool,
    create_mem0_memory_writer_tool,
)


@pytest.fixture
def store(monkeypatch, mock_mem0_client):  # noqa: ARG001
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    return Mem0MemoryStore()


class TestCreateMem0MemoryRetrieverTool:
    def test_returns_tool_instance(self, store):
        tool = create_mem0_memory_retriever_tool(store, user_id="u1")
        assert isinstance(tool, Tool)

    def test_default_name(self, store):
        tool = create_mem0_memory_retriever_tool(store, user_id="u1")
        assert tool.name == "retrieve_memories"

    def test_custom_name_and_description(self, store):
        tool = create_mem0_memory_retriever_tool(store, user_id="u1", name="my_tool", description="Custom desc")
        assert tool.name == "my_tool"
        assert tool.description == "Custom desc"

    def test_ids_not_exposed_to_llm(self, store):
        tool = create_mem0_memory_retriever_tool(store, user_id="u1")
        props = tool.parameters.get("properties", {})
        assert "user_id" not in props
        assert "run_id" not in props
        assert "agent_id" not in props

    def test_query_is_required(self, store):
        tool = create_mem0_memory_retriever_tool(store, user_id="u1")
        assert "query" in tool.parameters.get("required", [])

    def test_tool_calls_retriever_with_bound_ids(self, store):
        store.search_memories = Mock(return_value=[ChatMessage.from_system("remembered fact")])
        tool = create_mem0_memory_retriever_tool(store, user_id="bound-user", top_k=3)
        tool.invoke(query="what do I like?")
        kwargs = store.search_memories.call_args[1]
        assert kwargs["user_id"] == "bound-user"

    def test_tool_returns_string(self, store):
        store.search_memories = Mock(return_value=[ChatMessage.from_system("I like Python")])
        tool = create_mem0_memory_retriever_tool(store, user_id="u1")
        result = tool.invoke(query="programming")
        assert isinstance(result, str)
        assert "I like Python" in result

    def test_tool_returns_no_memories_message(self, store):
        store.search_memories = Mock(return_value=[])
        tool = create_mem0_memory_retriever_tool(store, user_id="u1")
        result = tool.invoke(query="nothing")
        assert "No memories found" in result


class TestCreateMem0MemoryWriterTool:
    def test_returns_tool_instance(self, store):
        tool = create_mem0_memory_writer_tool(store, user_id="u1")
        assert isinstance(tool, Tool)

    def test_default_name(self, store):
        tool = create_mem0_memory_writer_tool(store, user_id="u1")
        assert tool.name == "store_memory"

    def test_ids_not_exposed_to_llm(self, store):
        tool = create_mem0_memory_writer_tool(store, user_id="u1")
        props = tool.parameters.get("properties", {})
        assert "user_id" not in props

    def test_text_is_required(self, store):
        tool = create_mem0_memory_writer_tool(store, user_id="u1")
        assert "text" in tool.parameters.get("required", [])

    def test_tool_calls_writer_with_bound_ids(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = create_mem0_memory_writer_tool(store, user_id="bound-user")
        tool.invoke(text="I enjoy hiking")
        kwargs = store.add_memories.call_args[1]
        assert kwargs["user_id"] == "bound-user"

    def test_tool_returns_count_string(self, store):
        store.add_memories = Mock(return_value=[{"memory_id": "m1", "memory": "fact"}])
        tool = create_mem0_memory_writer_tool(store, user_id="u1")
        result = tool.invoke(text="I enjoy hiking")
        assert isinstance(result, str)
        assert "1" in result
