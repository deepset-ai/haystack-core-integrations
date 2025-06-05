import pytest

from haystack_integrations.tools.mcp import MCPTool, MCPToolset


@pytest.fixture
def mcp_tool_cleanup():
    """Fixture to ensure all MCPTool and MCPToolset instances are properly closed after tests."""
    tools = []
    toolsets = []

    def _register(item):
        """Register an MCP component for cleanup."""
        if isinstance(item, MCPTool):
            tools.append(item)
        elif isinstance(item, MCPToolset):
            toolsets.append(item)
        return item

    yield _register

    # Finalizer to close all tools and toolsets
    for tool in tools:
        tool.close()

    for toolset in toolsets:
        toolset.close()
