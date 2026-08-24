import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator

from haystack_integrations.agent_pack.plugins import Mem0MemoryPlugin, apply_plugins
from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore
from haystack_integrations.tools.mem0 import Mem0MemoryRetrieverTool, Mem0MemoryWriterTool


def test_mem0_plugin_contributes_tools_state_and_safe_prompt_guidance():
    store = Mem0MemoryStore()

    plugin = Mem0MemoryPlugin(memory_store=store, top_k=3)

    assert [type(tool) for tool in plugin.tools] == [Mem0MemoryRetrieverTool, Mem0MemoryWriterTool]
    assert plugin.tools[0].memory_store is store
    assert plugin.tools[0].top_k == 3
    assert plugin.tools[0].inputs_from_state == {"user_id": "user_id"}
    assert plugin.tools[1].memory_store is store
    assert plugin.state_schema == {"user_id": {"type": str}}
    assert "not authoritative factual evidence" in plugin.prompt_instructions
    assert "document-only or source-only restrictions" in plugin.prompt_instructions


def test_mem0_plugin_registers_custom_scope_state():
    plugin = Mem0MemoryPlugin(
        memory_store=Mem0MemoryStore(),
        inputs_from_state={"customer_id": "user_id", "session_id": "run_id"},
    )

    assert plugin.state_schema == {"customer_id": {"type": str}, "session_id": {"type": str}}
    assert plugin.tools[0].inputs_from_state == {"customer_id": "user_id", "session_id": "run_id"}
    assert plugin.tools[1].inputs_from_state == {"customer_id": "user_id", "session_id": "run_id"}


def test_mem0_plugin_agent_serialization_roundtrip():
    agent = Agent(chat_generator=MockChatGenerator("done"), system_prompt="Base")
    configured = apply_plugins(agent, [Mem0MemoryPlugin(memory_store=Mem0MemoryStore())])

    restored = Agent.from_dict(configured.to_dict())

    assert [type(tool) for tool in restored.tools] == [Mem0MemoryRetrieverTool, Mem0MemoryWriterTool]
    assert restored.state_schema == {"user_id": {"type": str}}
    assert "## Plugin: mem0_memory" in restored.system_prompt


def test_mem0_plugin_missing_dependency_error(monkeypatch):
    class _MissingMem0Import:
        def check(self):
            msg = "Run 'pip install mem0-haystack' to use Mem0MemoryPlugin."
            raise ImportError(msg)

    monkeypatch.setattr("haystack_integrations.agent_pack.plugins.mem0.mem0_import", _MissingMem0Import())

    with pytest.raises(ImportError, match="pip install mem0-haystack"):
        Mem0MemoryPlugin(memory_store=Mem0MemoryStore())
