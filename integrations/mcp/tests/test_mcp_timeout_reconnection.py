# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Test cases for MCP timeout and reconnection functionality.

Tests automatic reconnection logic for SSE connections that timeout after 5-10 minutes of idle time.
Since we can't wait that long in tests, we simulate timeouts by killing/restarting the server process.
This validates the complete reconnection flow with real SSEClient connections.
"""

import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo

logger = logging.getLogger(__name__)


class TestMCPTimeoutReconnection:
    """Test automatic reconnection logic for SSE connection timeouts."""

    @pytest.mark.integration
    @pytest.mark.skipif(sys.platform == "win32", reason="Windows subprocess handling differs")
    def test_real_sse_reconnection_after_server_restart(self):
        """
        Test reconnection works with real SSE server that gets restarted.

        Simulates SSE timeout (normally 5-10 mins) by killing/restarting server.
        Validates automatic reconnection with real SSEClient (not InMemoryClient).
        """

        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        def wait_for_server_ready(port, max_attempts=10):
            """Wait for server to accept connections."""

            for _attempt in range(max_attempts):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1)
                        result = s.connect_ex(("127.0.0.1", port))
                        if result == 0:
                            return True
                except Exception as e:
                    logger.debug(f"Server connection attempt failed: {e}")
                time.sleep(0.5)
            return False

        port = find_free_port()

        # Create server script
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(
                f"""
import sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Reconnection Test Server", host="127.0.0.1", port={port})

@mcp.tool()
def test_tool(message: str) -> str:
    return f"Server response: {{message}}"

if __name__ == "__main__":
    try:
        mcp.run(transport="sse")
    except Exception as e:
        print(f"Server error: {{e}}", file=sys.stderr)
        sys.exit(1)
""".encode()
            )
            server_script_path = temp_file.name

        server_process = None
        tool = None
        try:
            # Start server
            server_process = subprocess.Popen(
                ["python", server_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if not wait_for_server_ready(port):
                stdout, stderr = server_process.communicate(timeout=2)
                pytest.fail(f"Server failed to start on port {port}. stdout: {stdout}, stderr: {stderr}")

            # Create tool with real SSE connection
            server_info = SSEServerInfo(base_url=f"http://127.0.0.1:{port}")
            tool = MCPTool(name="test_tool", server_info=server_info)

            # Test normal operation
            result = tool.invoke(message="first call")
            result_data = json.loads(result)
            assert "Server response: first call" in result_data["content"][0]["text"]

            # Kill server to simulate timeout
            server_process.terminate()
            server_process.wait(timeout=5)
            time.sleep(1)

            # Restart server
            server_process = subprocess.Popen(
                ["python", server_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if not wait_for_server_ready(port):
                stdout, stderr = server_process.communicate(timeout=2)
                pytest.fail(f"Server failed to restart on port {port}. stdout: {stdout}, stderr: {stderr}")

            # This should trigger reconnection and succeed
            result = tool.invoke(message="after restart")
            result_data = json.loads(result)
            assert "Server response: after restart" in result_data["content"][0]["text"]

        except Exception:
            if server_process and server_process.poll() is None:
                try:
                    stdout, stderr = server_process.communicate(timeout=2)
                    logger.debug(f"Server stdout: {stdout}")
                    logger.debug(f"Server stderr: {stderr}")
                except subprocess.TimeoutExpired:
                    server_process.terminate()
            raise

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
                        server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                        server_process.wait(timeout=5)

            if os.path.exists(server_script_path):
                os.remove(server_script_path)
