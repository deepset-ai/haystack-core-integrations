"""
Tests for MCP timeout and reconnection functionality.

This module tests the automatic reconnection logic for MCP SSE connections.
Since we can't wait 10 minutes for real timeouts, we simulate disconnections
by killing the server process and then restarting it.
"""

import asyncio
import logging
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo
from haystack_integrations.tools.mcp.mcp_tool import SSEClient

logger = logging.getLogger(__name__)


def find_free_port():
    """Find a free port to use for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server_ready(port, max_attempts=20, delay=0.5):
    """Wait for server to accept connections."""
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("127.0.0.1", port))
                if result == 0:
                    logger.debug(f"Server ready on port {port} after {attempt + 1} attempts")
                    return True
        except Exception as e:
            logger.debug(f"Server connection attempt {attempt + 1} failed: {e}")
        time.sleep(delay)

    logger.warning(f"Server not ready on port {port} after {max_attempts} attempts")
    return False


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

    def test_retry_logic_with_mock(self):
        """Test retry logic using mocked SSE client (unit test)."""
        # Create a real SSE client but mock its session and connect method
        server_info = SSEServerInfo(url="http://localhost:8000/sse", max_retries=2, base_delay=0.1)
        client = server_info.create_client()

        # Mock the session and connect method
        mock_session = AsyncMock()
        client.session = mock_session
        client.connect = AsyncMock()

        # Create a successful response mock
        success_response = MagicMock()
        success_response.isError = False
        success_response.model_dump_json.return_value = '{"result": "success"}'

        # First call fails with ConnectionError, second call succeeds
        mock_session.call_tool.side_effect = [ConnectionError("Connection lost"), success_response]

        # Test the retry logic by calling it directly
        async def test_call():
            result = await client.call_tool("test_tool", {"message": "test"})
            return result

        # Run the async test
        result = asyncio.run(test_call())

        # Verify the result
        assert result == '{"result": "success"}'

        # Verify retry behavior
        assert mock_session.call_tool.call_count == 2  # Initial call + retry
        assert client.connect.call_count == 1  # Reconnection attempted once

    def test_real_sse_reconnection_after_server_restart(self):
        """Test real SSE reconnection by killing and restarting server."""
        port = find_free_port()  # Use dynamic port to avoid conflicts
        server_process = None
        server_script_path = None
        tool = None

        try:
            # Create server script with cross-platform signal handling
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                script_content = textwrap.dedent(f"""
                    import sys
                    import signal
                    from mcp.server.fastmcp import FastMCP

                    # Handle shutdown signals gracefully (cross-platform)
                    def signal_handler(signum, frame):
                        sys.exit(0)

                    # Only set up signal handlers that exist on the platform
                    if hasattr(signal, 'SIGTERM'):
                        signal.signal(signal.SIGTERM, signal_handler)
                    if hasattr(signal, 'SIGINT'):
                        signal.signal(signal.SIGINT, signal_handler)

                    mcp = FastMCP("Reconnection Test Server", host="127.0.0.1", port={port})

                    @mcp.tool()
                    def test_tool(message: str) -> str:
                        return f"Server response: {{message}}"

                    if __name__ == "__main__":
                        try:
                            print(f"Starting server on port {port}", flush=True)
                            mcp.run(transport="sse")
                        except (KeyboardInterrupt, SystemExit):
                            print("Server shutting down gracefully", flush=True)
                            sys.exit(0)
                        except Exception as e:
                            print(f"Server error: {{e}}", file=sys.stderr, flush=True)
                            sys.exit(1)
                """).strip()
                temp_file.write(script_content.encode())
                server_script_path = temp_file.name

            # Start server
            logger.debug(f"Starting test server on port {port}")
            server_process = subprocess.Popen(
                [sys.executable, server_script_path],  # Use same Python executable
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for server to be ready
            if not wait_for_server_ready(port):
                # Get server output for debugging
                try:
                    stdout, stderr = server_process.communicate(timeout=2)
                    logger.error(f"Server failed to start. stdout: {stdout}, stderr: {stderr}")
                except subprocess.TimeoutExpired:
                    logger.error("Server startup timed out")
                    server_process.terminate()
                pytest.fail(f"Server failed to start on port {port}")

            # Create tool with custom retry parameters
            logger.debug(f"Creating MCPTool for server on port {port}")
            tool = MCPTool(
                name="test_tool",
                server_info=SSEServerInfo(url=f"http://127.0.0.1:{port}/sse", max_retries=2, base_delay=0.5),
            )

            # First call should work
            result = tool.invoke(message="hello")
            assert "Server response: hello" in result

            # Kill server to simulate timeout
            logger.debug("Terminating server to simulate connection loss")
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logger.debug("Server didn't terminate gracefully, using kill")
                server_process.kill()
                server_process.wait(timeout=2)

            # Restart server
            logger.debug(f"Restarting server on port {port}")
            server_process = subprocess.Popen(
                [sys.executable, server_script_path],  # Use same Python executable
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for restarted server to be ready
            if not wait_for_server_ready(port):
                # Get server output for debugging
                try:
                    stdout, stderr = server_process.communicate(timeout=2)
                    logger.error(f"Restarted server failed. stdout: {stdout}, stderr: {stderr}")
                except subprocess.TimeoutExpired:
                    logger.error("Restarted server startup timed out")
                    server_process.terminate()
                pytest.fail(f"Restarted server failed to start on port {port}")

            # Second call should trigger reconnection and work
            logger.debug("Testing reconnection after server restart")
            result = tool.invoke(message="world")
            assert "Server response: world" in result

        finally:
            # Clean up tool
            if tool:
                try:
                    tool.close()
                except Exception as e:
                    logger.debug(f"Error closing tool: {e}")

            # Clean up server process
            if server_process:
                if server_process.poll() is None:
                    logger.debug("Cleaning up server process")
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.debug("Server didn't terminate gracefully during cleanup, using kill")
                        server_process.kill()
                        try:
                            server_process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.warning("Server process still hanging after kill during cleanup")

            # Clean up temporary script file
            if server_script_path and os.path.exists(server_script_path):
                try:
                    os.remove(server_script_path)
                    logger.debug(f"Removed temporary script: {server_script_path}")
                except Exception as e:
                    logger.debug(f"Error removing script file: {e}")
