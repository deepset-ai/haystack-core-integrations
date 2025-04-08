import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
from haystack.components.tools import ToolInvoker
from haystack.tools import Tool

from haystack_integrations.tools.mcp import (
    MCPToolset,
    SSEServerInfo
)
from haystack_integrations.tools.mcp.mcp_tool import MCPConnectionError, MCPTool


@pytest.fixture
def mock_mcp_toolset():
    """Fixture to create a pre-configured MCPToolset for testing without server connection."""
    mock_tool1 = MagicMock(spec=MCPTool)
    mock_tool1.name = "tool1"
    mock_tool1.description = "Test tool 1"
    
    mock_tool2 = MagicMock(spec=MCPTool)
    mock_tool2.name = "tool2"
    mock_tool2.description = "Test tool 2"

    with (
        patch("haystack_integrations.tools.mcp.mcp_toolset.MCPToolset._run_sync") as mock_run_sync,
        patch("haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo.create_client") as mock_create_client,
        patch("haystack_integrations.tools.mcp.mcp_toolset.MCPTool", autospec=True) as mock_mcp_tool_class,
    ):
        mock_client = MagicMock()
        
        mock_tool_info1 = MagicMock()
        mock_tool_info1.name = "tool1"
        mock_tool_info1.description = "Test tool 1"
        mock_tool_info1.inputSchema = {"type": "object", "properties": {}}
        
        mock_tool_info2 = MagicMock()
        mock_tool_info2.name = "tool2"
        mock_tool_info2.description = "Test tool 2"
        mock_tool_info2.inputSchema = {"type": "object", "properties": {}}
        
        mock_run_sync.return_value = [mock_tool_info1, mock_tool_info2]
        mock_create_client.return_value = mock_client
        mock_mcp_tool_class.side_effect = [mock_tool1, mock_tool2]
        
        toolset = MCPToolset(
            server_info=SSEServerInfo(base_url="http://example.com", token="test-token"),
            connection_timeout=45,
            invocation_timeout=60
        )
        
        yield toolset


class TestMCPToolset:
    """Tests for the MCPToolset class."""

    def test_toolset_initialization(self, mock_mcp_toolset):
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

    def test_toolset_serde(self, mock_mcp_toolset):
        """Test serialization and deserialization of MCPToolset."""
        toolset = mock_mcp_toolset
        
        toolset_dict = toolset.to_dict()
        assert toolset_dict["type"] == "haystack_integrations.tools.mcp.mcp_toolset.MCPToolset"
        assert toolset_dict["data"]["connection_timeout"] == 45
        assert toolset_dict["data"]["invocation_timeout"] == 60
        assert toolset_dict["data"]["server_info"]["base_url"] == "http://example.com"
        
        with patch("haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.__init__", return_value=None) as mock_init:
            MCPToolset.from_dict(toolset_dict)
            
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["connection_timeout"] == 45
            assert kwargs["invocation_timeout"] == 60
            assert isinstance(kwargs["server_info"], SSEServerInfo)
            assert kwargs["server_info"].base_url == "http://example.com"

    def test_toolset_combination(self, mock_mcp_toolset):
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
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        )
        
        combined_tools = toolset + [add_tool]
        
        assert len(combined_tools) == 3
        
        tool_names = [tool.name for tool in combined_tools.tools]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
        assert "add" in tool_names

    def test_toolset_error_handling(self):
        """Test error handling during toolset initialization."""
        with (
            patch("haystack_integrations.tools.mcp.mcp_toolset.MCPToolset._run_sync") as mock_run_sync,
            patch("haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo.create_client") as mock_create_client,
        ):
            mock_client = MagicMock()
            mock_create_client.return_value = mock_client
            mock_run_sync.side_effect = Exception("Connection failed")
            
            with pytest.raises(MCPConnectionError):
                MCPToolset(
                    server_info=SSEServerInfo(base_url="http://example.com"),
                    connection_timeout=30,
                    invocation_timeout=30
                )

    def test_run_sync_method(self):
        """Test the _run_sync method for handling asyncio coroutines."""
        async def sample_coro():
            return "test_result"
            
        # Test with no running loop
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            result = MCPToolset._run_sync(None, sample_coro())
            assert result == "test_result"
            
        # Test with running loop - key behavior we improved
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop
            
            mock_temp_loop = MagicMock()
            mock_temp_loop.run_until_complete.return_value = "test_result"
            
            with patch("asyncio.new_event_loop", return_value=mock_temp_loop):
                result = MCPToolset._run_sync(None, sample_coro())
                assert result == "test_result"
                mock_temp_loop.close.assert_called_once()


@pytest.mark.integration
class TestMCPToolsetIntegration:
    """Integration tests for MCPToolset."""
    
    @pytest.mark.skipif(sys.platform == "win32", reason="Windows fails for some reason")
    def test_toolset_with_sse_connection(self):
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
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MCP Math Server", host="127.0.0.1", port={port})

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers\"\"\"
    return a * b

if __name__ == "__main__":
    try:
        mcp.run(transport="sse")
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)
""".encode()
            )
            server_script_path = temp_file.name
        
        server_process = None
        try:
            server_process = subprocess.Popen(
                ["python", server_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(2)  # Wait for server to start
            
            server_info = SSEServerInfo(base_url=f"http://127.0.0.1:{port}")
            toolset = MCPToolset(server_info=server_info)
            
            assert len(toolset) == 2
            
            tool_names = [tool.name for tool in toolset.tools]
            assert "add" in tool_names
            assert "multiply" in tool_names
            
            add_tool = next(tool for tool in toolset.tools if tool.name == "add")
            result = add_tool.invoke(a=5, b=3)
            assert result.content[0].text == "8"
            
            multiply_tool = next(tool for tool in toolset.tools if tool.name == "multiply")
            result = multiply_tool.invoke(a=4, b=7)  
            assert result.content[0].text == "28"
            
        finally:
            # Cleanup resources
            if server_process and server_process.poll() is None:
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    server_process.wait(timeout=5)
            
            if os.path.exists(server_script_path):
                os.remove(server_script_path) 