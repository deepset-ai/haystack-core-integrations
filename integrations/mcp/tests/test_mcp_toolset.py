import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.tools import Tool

from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo
from haystack_integrations.tools.mcp.mcp_tool import MCPConnectionError


@pytest.fixture
async def mock_mcp_toolset():
    """Fixture to create a pre-configured MCPToolset for testing without server connection."""
    mock_tool1 = MagicMock(spec=Tool)
    mock_tool1.name = "tool1"
    mock_tool1.description = "Test tool 1"
    mock_tool1.inputSchema = {"type": "object", "properties": {}}

    mock_tool2 = MagicMock(spec=Tool)
    mock_tool2.name = "tool2"
    mock_tool2.description = "Test tool 2"
    mock_tool2.inputSchema = {"type": "object", "properties": {}}

    mock_client = AsyncMock()
    mock_client.connect.return_value = [mock_tool1, mock_tool2]
    mock_client.close = AsyncMock()

    with (
        patch("haystack_integrations.tools.mcp.mcp_toolset.AsyncExecutor.get_instance") as mock_executor,
        patch("haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo.create_client") as mock_create_client,
    ):
        mock_create_client.return_value = mock_client
        mock_executor.return_value.run.return_value = [mock_tool1, mock_tool2]

        toolset = MCPToolset(
            server_info=SSEServerInfo(base_url="http://example.com", token="test-token"),
            connection_timeout=45,
            invocation_timeout=60,
        )

        yield toolset

        # Cleanup
        await mock_client.close()


@pytest.fixture
async def mock_mcp_toolset_with_tool_names():
    """Fixture to create an MCPToolset with specific tool_names filtering."""
    mock_tool2 = MagicMock(spec=Tool)
    mock_tool2.name = "tool2"
    mock_tool2.description = "Test tool 2"
    mock_tool2.inputSchema = {"type": "object", "properties": {}}

    mock_client = AsyncMock()
    mock_client.connect.return_value = [mock_tool2]
    mock_client.close = AsyncMock()

    with (
        patch("haystack_integrations.tools.mcp.mcp_toolset.AsyncExecutor.get_instance") as mock_executor,
        patch("haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo.create_client") as mock_create_client,
    ):
        mock_create_client.return_value = mock_client
        mock_executor.return_value.run.return_value = [mock_tool2]

        toolset = MCPToolset(
            server_info=SSEServerInfo(base_url="http://example.com", token="test-token"),
            tool_names=["tool2"],  # Only include tool2
            connection_timeout=45,
            invocation_timeout=60,
        )

        yield toolset

        # Cleanup
        await mock_client.close()


@pytest.mark.asyncio
class TestMCPToolset:
    """Tests for the MCPToolset class."""

    async def test_toolset_initialization(self, mock_mcp_toolset):
        """Test if the MCPToolset initializes correctly and loads tools."""
        toolset = mock_mcp_toolset

        assert isinstance(toolset.server_info, SSEServerInfo)
        assert toolset.connection_timeout == 45
        assert toolset.invocation_timeout == 60
        assert len(toolset) == 2

        tool_names = [tool.name for tool in toolset.tools]
        assert "tool1" in tool_names
        assert "tool2" in tool_names

        tool1 = next(tool for tool in toolset.tools if tool.name == "tool1")
        tool2 = next(tool for tool in toolset.tools if tool.name == "tool2")

        assert tool1.name == "tool1"
        assert tool2.name == "tool2"
        assert tool1.description == "Test tool 1"
        assert tool2.description == "Test tool 2"

    async def test_toolset_with_filtered_tools(self, mock_mcp_toolset_with_tool_names):
        """Test if the MCPToolset correctly filters tools based on tool_names parameter."""
        toolset = mock_mcp_toolset_with_tool_names

        # Verify tool_names parameter was stored
        assert toolset.tool_names == ["tool2"]

        # Verify only the specified tool was added
        assert len(toolset) == 1

        tool_names = [tool.name for tool in toolset.tools]
        assert "tool1" not in tool_names
        assert "tool2" in tool_names

        # Check the tool that was included
        tool = toolset.tools[0]
        assert tool.name == "tool2"
        assert tool.description == "Test tool 2"

    async def test_toolset_serde(self, mock_mcp_toolset):
        """Test serialization and deserialization of MCPToolset."""
        toolset = mock_mcp_toolset

        toolset_dict = toolset.to_dict()
        assert toolset_dict["type"] == "haystack_integrations.tools.mcp.mcp_toolset.MCPToolset"
        assert toolset_dict["data"]["connection_timeout"] == 45
        assert toolset_dict["data"]["invocation_timeout"] == 60
        assert toolset_dict["data"]["server_info"]["base_url"] == "http://example.com"
        assert toolset_dict["data"]["tool_names"] is None

        with patch("haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.__init__", return_value=None) as mock_init:
            MCPToolset.from_dict(toolset_dict)

            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["connection_timeout"] == 45
            assert kwargs["invocation_timeout"] == 60
            assert kwargs["tool_names"] is None
            assert isinstance(kwargs["server_info"], SSEServerInfo)
            assert kwargs["server_info"].base_url == "http://example.com"

    async def test_toolset_serde_with_tool_names(self, mock_mcp_toolset_with_tool_names):
        """Test serialization and deserialization of MCPToolset with tool_names parameter."""
        toolset = mock_mcp_toolset_with_tool_names

        toolset_dict = toolset.to_dict()
        assert toolset_dict["type"] == "haystack_integrations.tools.mcp.mcp_toolset.MCPToolset"
        assert toolset_dict["data"]["tool_names"] == ["tool2"]

        with patch("haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.__init__", return_value=None) as mock_init:
            MCPToolset.from_dict(toolset_dict)

            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["tool_names"] == ["tool2"]

    async def test_toolset_combination(self, mock_mcp_toolset):
        """Test combining MCPToolset with other tools."""
        toolset = mock_mcp_toolset

        def add(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b

        add_tool = Tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        )

        combined_tools = toolset + [add_tool]

        assert len(combined_tools) == 3

        tool_names = [tool.name for tool in combined_tools.tools]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
        assert "add" in tool_names

    async def test_toolset_error_handling(self):
        """Test error handling during toolset initialization."""
        with pytest.raises(MCPConnectionError):
            MCPToolset(
                server_info=SSEServerInfo(base_url="http://example.com"),
                connection_timeout=30.0,
                invocation_timeout=30.0,
            )


@pytest.mark.integration
class TestMCPToolsetIntegration:
    """Integration tests for MCPToolset."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows fails for some reason")
    async def test_toolset_with_sse_connection(self):
        """Test MCPToolset with an SSE connection to a simple server."""
        import socket
        import subprocess
        import tempfile

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(
                f"""
import sys
import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MCP Calculator", host="127.0.0.1", port={port})

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    \"\"\"Subtract b from a\"\"\"
    return a - b

if __name__ == "__main__":
    try:
        asyncio.run(mcp.run_async(transport="sse"))
    except Exception as e:
        sys.exit(1)
""".encode()
            )
            server_script_path = temp_file.name

        server_process = None
        toolset = None
        try:
            server_process = subprocess.Popen(
                ["python", server_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for server to start and be ready
            time.sleep(5)  # Increased from 2 to 5 seconds

            server_info = SSEServerInfo(base_url=f"http://127.0.0.1:{port}")
            toolset = MCPToolset(server_info=server_info)

            assert len(toolset) == 2

            tool_names = [tool.name for tool in toolset.tools]
            assert "add" in tool_names
            assert "subtract" in tool_names

            add_tool = next(tool for tool in toolset.tools if tool.name == "add")
            result = add_tool.invoke(a=5, b=3)
            assert result.content[0].text == "8"

            subtract_tool = next(tool for tool in toolset.tools if tool.name == "subtract")
            result = subtract_tool.invoke(a=10, b=4)
            assert result.content[0].text == "6"

        finally:
            # Clean up toolset first to close connections
            if toolset is not None:
                del toolset

            # Cleanup server process
            if server_process and server_process.poll() is None:
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    server_process.wait(timeout=5)

            # Clean up temporary file
            if os.path.exists(server_script_path):
                os.remove(server_script_path)
