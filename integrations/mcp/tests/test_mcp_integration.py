import json
import os
import socket
import subprocess
import sys
import tempfile
import time

import pytest
from haystack import Pipeline, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole

from haystack_integrations.tools.mcp import (
    MCPConnectionError,
    MCPError,
    MCPTool,
    SSEServerInfo,
    StdioServerInfo,
)

from .mcp_memory_transport import InMemoryServerInfo
from .mcp_servers_fixtures import echo_mcp

logger = logging.getLogger(__name__)


# Keep integration tests separate
@pytest.mark.integration
class TestMCPToolInPipelineWithOpenAI:
    """Integration tests for MCPTool in Haystack pipelines with external dependencies."""

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.skipif(sys.platform == "win32", reason="Windows fails for some reason")
    def test_mcp_tool_with_http_server(self):
        """Test using an MCPTool with a real HTTP server."""

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
import sys
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("MCP Calculator", host="127.0.0.1", port={port})
@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    return a + b

# Add a subtraction tool
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
            # Create an MCPTool that connects to the HTTP server
            server_info = SSEServerInfo(base_url=f"http://127.0.0.1:{port}")

            # Create the tool
            tool = MCPTool(name="add", server_info=server_info)

            # Invoke the tool
            result_json = tool.invoke(a=5, b=3)
            result = json.loads(result_json)

            # Verify the result
            assert not result["isError"]
            assert len(result["content"]) == 1
            assert result["content"][0]["text"] == "8"

            # Try another tool from the same server
            subtract_tool = MCPTool(name="subtract", server_info=server_info)
            result_json = subtract_tool.invoke(a=10, b=4)

            # Parse the JSON result
            result = json.loads(result_json)

            # Verify the result
            assert not result["isError"]
            assert len(result["content"]) == 1
            assert result["content"][0]["text"] == "6"

        except Exception:
            # Check server output for clues
            if server_process and server_process.poll() is None:
                server_process.terminate()
            raise

        finally:
            # Explicitly close tools first to prevent SSE connection errors
            try:
                tool.close()
                subtract_tool.close()
            except Exception as e:
                logger.debug(f"Error during tool cleanup: {e}")

            # Then clean up the server process
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

    @pytest.mark.skip("Brave is temporarily not returning results")
    def test_mcp_brave_search(self, mcp_tool_cleanup):
        """Test using an MCPTool in a pipeline with OpenAI."""

        # Create an MCPTool for the brave_web_search operation
        server_info = StdioServerInfo(
            command="docker",
            args=["run", "-i", "--rm", "-e", "BRAVE_MCP_TRANSPORT", "-e", "BRAVE_API_KEY", "mcp/brave-search"],
            env={
                "BRAVE_MCP_TRANSPORT": "stdio",
                "BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", "YOUR_API_KEY_HERE"),
            },
        )
        try:
            tool = MCPTool(name="brave_web_search", server_info=server_info)
            # Register for cleanup
            mcp_tool_cleanup(tool)

        except MCPError as e:
            if "Could not find docker command" in str(e):
                pytest.skip("Docker is not installed or not in PATH")
            raise

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4.1-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")

        # Create a message that should trigger tool use
        message = ChatMessage.from_user(
            text="Use brave_web_search to search for the latest news about the stock market, use the `query` parameter"
        )

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert any(
            term in tool_message.tool_call_result.result
            for term in ["equity", "market", "stock", "price", "NASDAQ", "S&P 500"]
        ), f"Result should contain information about the stock market\n\nResult: {tool_message.tool_call_result.result}"

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_mcp_tool_in_pipeline_with_multiple_tools(self, mcp_tool_cleanup):
        """Test using multiple MCPTools in a pipeline with OpenAI."""

        # Mix mcp tool with a simple echo tool
        try:
            time_server_info = StdioServerInfo(
                command="uvx", args=["mcp-server-time", "--local-timezone=America/New_York"]
            )
            time_tool = MCPTool(name="get_current_time", server_info=time_server_info)
            # Register for cleanup
            mcp_tool_cleanup(time_tool)

        except MCPError as e:
            if "Could not find uvx command" in str(e):
                pytest.skip("uvx command not found, skipping test")
            raise

        echo_server_info = InMemoryServerInfo(server=echo_mcp._mcp_server)
        echo_tool = MCPTool(name="echo", server_info=echo_server_info)
        # Register for cleanup
        mcp_tool_cleanup(echo_tool)

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4.1-mini", tools=[echo_tool, time_tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[echo_tool, time_tool]))

        pipeline.connect("llm.replies", "tool_invoker.messages")

        # Create a message that should trigger tool use
        message = ChatMessage.from_user(text="What is the current time in New York?")

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "timezone" in tool_message.tool_call_result.result
        assert "datetime" in tool_message.tool_call_result.result
        assert "New_York" in tool_message.tool_call_result.result

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_mcp_tool_error_handling_integration(self):
        """Test error handling with MCPTool connection in a pipeline (integration)."""

        # Use a non-existent server address to force a connection error
        server_info = SSEServerInfo(base_url="http://localhost:9999", timeout=1)  # Short timeout
        with pytest.raises(MCPConnectionError) as exc_info:
            MCPTool(name="non_existent_tool", server_info=server_info, connection_timeout=2, eager_connect=True)

        # Check for platform-agnostic error message patterns
        error_message = str(exc_info.value)
        assert error_message, "Error message should not be empty"
        assert any(text in error_message.lower() for text in ["failed", "connection", "initialize"]), (
            f"Error message '{error_message}' should contain connection failure information"
        )
