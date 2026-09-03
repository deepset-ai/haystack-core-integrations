import pytest
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.tools import Toolset, tool

from haystack_integrations.agent_pack.optimization import (
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)
from haystack_integrations.agent_pack.optimization.agent import HARNESS_OPTIMIZER_SYSTEM_PROMPT


def test_system_prompt_grants_full_configuration_control_and_explains_mutations():
    """The optimizer is guided toward evidence-based arbitrary edits rather than named patches."""
    assert "complete serialized reference Agent configuration" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "may change any part" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "RFC 6901" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "rather than an allowlist" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "documentation tools" in HARNESS_OPTIMIZER_SYSTEM_PROMPT


def test_optimizer_agent_defaults_and_optional_docs_toolset(monkeypatch):
    """The factory keeps provider and optional documentation setup compact."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    default = create_harness_optimizer_agent()
    assert isinstance(default.chat_generator, OpenAIResponsesChatGenerator)
    assert default.chat_generator.model == "gpt-5.6-sol"
    assert default.system_prompt == HARNESS_OPTIMIZER_SYSTEM_PROMPT

    @tool
    def search_haystack_docs(query: str) -> str:
        """Search official Haystack documentation."""
        return query

    docs = Toolset([search_haystack_docs])
    with_docs = create_harness_optimizer_agent(chat_generator=MockChatGenerator("{}"), docs_toolset=docs)
    assert with_docs.tools == [docs]


def test_haystack_documentation_mcp_server_is_read_only_and_lazy():
    """The exposed public MCP integration contains only documentation search."""
    pytest.importorskip("haystack_integrations.tools.mcp", reason="mcp-haystack is optional")
    toolset = create_haystack_documentation_mcp_toolset()
    assert toolset.tool_names == ["search_haystack_docs"]
    assert toolset.eager_connect is False
