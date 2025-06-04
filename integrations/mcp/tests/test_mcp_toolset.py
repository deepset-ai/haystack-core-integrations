import os
import sys
import time
from unittest.mock import patch

import pytest
import pytest_asyncio
from haystack import logging
from haystack.tools import Tool

from haystack_integrations.tools.mcp import MCPToolset
from haystack_integrations.tools.mcp.mcp_tool import MCPConnectionError, SSEServerInfo

# Import in-memory transport and fixtures
from .mcp_memory_transport import InMemoryServerInfo
from .mcp_servers_fixtures import calculator_mcp, echo_mcp

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def calculator_toolset(mcp_tool_cleanup):
    """Fixture that provides an MCPToolset connected to the in-memory ``calculator_mcp`` server."""

    server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
    toolset = MCPToolset(
        server_info=server_info,
        connection_timeout=45,
        invocation_timeout=60,
    )

    return mcp_tool_cleanup(toolset)


@pytest_asyncio.fixture
async def echo_toolset(mcp_tool_cleanup):
    """Fixture that provides an MCPToolset connected to the in-memory ``echo_mcp`` server."""

    server_info = InMemoryServerInfo(server=echo_mcp._mcp_server)
    toolset = MCPToolset(
        server_info=server_info,
        connection_timeout=45,
        invocation_timeout=60,
    )

    return mcp_tool_cleanup(toolset)


@pytest_asyncio.fixture
async def calculator_toolset_with_tool_filter(mcp_tool_cleanup):
    """Fixture that provides an MCPToolset connected to ``calculator_mcp`` but exposing only the *add* tool."""

    server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
    toolset = MCPToolset(
        server_info=server_info,
        tool_names=["add"],  # Only include the 'add' tool
        connection_timeout=45,
        invocation_timeout=60,
    )

    return mcp_tool_cleanup(toolset)


@pytest.mark.asyncio
class TestMCPToolset:
    """Tests for the MCPToolset class."""

    async def test_toolset_initialization(self, calculator_toolset):
        """Test if the MCPToolset initializes correctly and loads tools."""
        toolset = calculator_toolset

        assert isinstance(toolset.server_info, InMemoryServerInfo)
        assert toolset.connection_timeout == 45
        assert toolset.invocation_timeout == 60
        assert len(toolset) == 3  # add, subtract, divide_by_zero

        tool_names = [tool.name for tool in toolset.tools]
        assert "add" in tool_names
        assert "subtract" in tool_names
        assert "divide_by_zero" in tool_names

        add_tool = next(tool for tool in toolset.tools if tool.name == "add")
        subtract_tool = next(tool for tool in toolset.tools if tool.name == "subtract")

        assert add_tool.name == "add"
        assert subtract_tool.name == "subtract"
        assert "Add two integers." in add_tool.description
        assert "Subtract integer b from integer a." in subtract_tool.description

    async def test_echo_toolset(self, echo_toolset):
        """Test the MCPToolset with echo server."""
        toolset = echo_toolset

        assert len(toolset) == 1  # echo

        tool_names = [tool.name for tool in toolset.tools]
        assert "echo" in tool_names

        echo_tool = toolset.tools[0]
        assert echo_tool.name == "echo"
        assert "Echo the input text." in echo_tool.description

    async def test_toolset_with_filtered_tools(self, calculator_toolset_with_tool_filter):
        """Test if the MCPToolset correctly filters tools based on tool_names parameter."""
        toolset = calculator_toolset_with_tool_filter

        # Verify tool_names parameter was stored
        assert toolset.tool_names == ["add"]

        # Verify only the specified tool was added
        assert len(toolset) == 1

        tool_names = [tool.name for tool in toolset.tools]
        assert "subtract" not in tool_names
        assert "divide_by_zero" not in tool_names
        assert "add" in tool_names

        # Check the tool that was included
        tool = toolset.tools[0]
        assert tool.name == "add"
        assert "Add two integers." in tool.description

    async def test_toolset_serde(self, calculator_toolset):
        """Test serialization and deserialization of MCPToolset."""
        toolset = calculator_toolset

        toolset_dict = toolset.to_dict()
        assert toolset_dict["type"] == "haystack_integrations.tools.mcp.mcp_toolset.MCPToolset"
        assert toolset_dict["data"]["connection_timeout"] == 45
        assert toolset_dict["data"]["invocation_timeout"] == 60
        assert isinstance(toolset_dict["data"]["server_info"], dict)
        assert toolset_dict["data"]["tool_names"] is None

        with patch("haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.__init__", return_value=None) as mock_init:
            MCPToolset.from_dict(toolset_dict)

            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["connection_timeout"] == 45
            assert kwargs["invocation_timeout"] == 60
            assert kwargs["tool_names"] is None

    async def test_toolset_serde_with_tool_names(self, calculator_toolset_with_tool_filter):
        """Test serialization and deserialization of MCPToolset with tool_names parameter."""
        toolset = calculator_toolset_with_tool_filter

        toolset_dict = toolset.to_dict()
        assert toolset_dict["type"] == "haystack_integrations.tools.mcp.mcp_toolset.MCPToolset"
        assert toolset_dict["data"]["tool_names"] == ["add"]

        with patch("haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.__init__", return_value=None) as mock_init:
            MCPToolset.from_dict(toolset_dict)

            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["tool_names"] == ["add"]

    async def test_toolset_combination(self, calculator_toolset):
        """Test combining MCPToolset with other tools."""
        toolset = calculator_toolset

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers"""
            return a * b

        multiply_tool = Tool(
            name="multiply",
            description="Multiply two numbers",
            function=multiply,
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        )

        combined_tools = toolset + [multiply_tool]

        assert len(combined_tools) == 4  # add, subtract, divide_by_zero, multiply

        tool_names = [tool.name for tool in combined_tools.tools]
        assert "add" in tool_names
        assert "subtract" in tool_names
        assert "divide_by_zero" in tool_names
        assert "multiply" in tool_names

    @patch.object(InMemoryServerInfo, "create_client")
    async def test_toolset_error_handling(self, mock_create_client):
        """Test that initialization errors from the underlying client are surfaced as ``MCPConnectionError``."""

        mock_create_client.side_effect = MCPConnectionError(
            message="Test connection error",
            operation="connect",
        )

        server_info = InMemoryServerInfo(server=calculator_mcp)

        with pytest.raises(MCPConnectionError):
            MCPToolset(
                server_info=server_info,
                connection_timeout=1.0,
                invocation_timeout=1.0,
            )


@pytest.mark.integration
class TestMCPToolsetIntegration:
    """Integration tests for MCPToolset."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows fails for some reason")
    def test_toolset_with_sse_connection(self):
        """Test MCPToolset with an SSE connection to a simple server."""
        import socket
        import subprocess
        import tempfile

        # Find an available port
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        port = find_free_port()

        # Create a temporary file for the server script
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(
                f"""
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
        mcp.run(transport="sse")
    except Exception as e:
        sys.exit(1)
""".encode()
            )
            server_script_path = temp_file.name

        server_process = None
        try:
            # Start the server in a separate process
            server_process = subprocess.Popen(
                ["python", server_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Give the server a moment to start
            time.sleep(2)

            # Create the toolset
            server_info = SSEServerInfo(base_url=f"http://127.0.0.1:{port}")
            toolset = MCPToolset(server_info=server_info)
            # Verify we got both tools
            assert len(toolset) == 2

            tool_names = [tool.name for tool in toolset.tools]
            assert "add" in tool_names
            assert "subtract" in tool_names

            # Test the add tool
            add_tool = next(tool for tool in toolset.tools if tool.name == "add")
            result = add_tool.invoke(a=5, b=3)
            assert result.content[0].text == "8"

            # Test the subtract tool
            subtract_tool = next(tool for tool in toolset.tools if tool.name == "subtract")
            result = subtract_tool.invoke(a=10, b=4)
            assert result.content[0].text == "6"

        except Exception:
            # Check server output for clues
            if server_process and server_process.poll() is None:
                server_process.terminate()
            raise

        finally:
            # Explicitly close tools first to prevent SSE connection errors
            try:
                toolset.close()
            except Exception as e:
                logger.debug(f"Error during tool cleanup: {e}")

            # Clean up
            if server_process:
                if server_process.poll() is None:  # Process is still running
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                        server_process.wait(timeout=5)

            # Remove the temporary file
            if os.path.exists(server_script_path):
                os.remove(server_script_path)
