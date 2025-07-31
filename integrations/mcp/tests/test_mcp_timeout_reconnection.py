"""
Tests for MCP timeout and reconnection functionality.

This module tests the automatic reconnection logic for MCP SSE connections.
Since we can't wait 10 minutes for real timeouts, we simulate disconnections
by killing the server process and then restarting it.
"""

import logging
import os
import subprocess
import tempfile
import time

from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo
from haystack_integrations.tools.mcp.mcp_tool import SSEClient

logger = logging.getLogger(__name__)


class TestMCPTimeoutReconnection:
    """Test MCP timeout and reconnection functionality."""

    def test_configurable_retry_parameters(self):
        """Test that retry parameters are configurable."""
        # Test SSE client with custom retry parameters
        server_info = SSEServerInfo(url="http://localhost:8000/sse", max_retries=5, base_delay=2.0)
        client = server_info.create_client()

        assert client.max_retries == 5
        assert client.base_delay == 2.0

        # Test that parameters are passed through correctly
        assert isinstance(client, SSEClient)

    def test_real_sse_reconnection_after_server_restart(self):
        """Test real SSE reconnection by killing and restarting server."""
        port = 8001
        server_process = None
        server_script_path = None

        try:
            # Create server script
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                temp_file.write(
                    f"""
import sys
import signal
from mcp.server.fastmcp import FastMCP

# Handle shutdown signals gracefully
def signal_handler(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

mcp = FastMCP("Reconnection Test Server", host="127.0.0.1", port={port})

@mcp.tool()
def test_tool(message: str) -> str:
    return f"Server response: {{message}}"

if __name__ == "__main__":
    try:
        mcp.run(transport="sse")
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {{e}}", file=sys.stderr)
        sys.exit(1)
""".encode()
                )
                server_script_path = temp_file.name

            # Start server
            server_process = subprocess.Popen(
                ["python", server_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(2)  # Wait for server to start

            # Create tool with custom retry parameters
            tool = MCPTool(
                name="test_tool",
                server_info=SSEServerInfo(url=f"http://127.0.0.1:{port}/sse", max_retries=2, base_delay=0.5),
            )

            # First call should work
            result = tool.invoke(message="hello")
            assert "Server response: hello" in result

            # Kill server to simulate timeout
            server_process.terminate()
            try:
                server_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=2)
            time.sleep(1)

            # Restart server
            server_process = subprocess.Popen(
                ["python", server_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(2)

            # Second call should trigger reconnection and work
            result = tool.invoke(message="world")
            assert "Server response: world" in result

        finally:
            if tool:
                try:
                    tool.close()
                except Exception as e:
                    logger.debug(f"Error closing tool: {e}")

            if server_process:
                if server_process.poll() is None:
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                        try:
                            server_process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.warning("Server process still hanging after kill")

            if os.path.exists(server_script_path):
                os.remove(server_script_path)
