import os
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from haystack_integrations.tools.mcp import MCPTool, MCPToolset

# Minimal FastMCP "calculator" server exposing add/subtract tools. `{port}` and `{transport}`
# are filled in per invocation by the mcp_calculator_server fixture below.
_CALCULATOR_SERVER_TEMPLATE = """
import sys
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
        mcp.run(transport="{transport}")
    except Exception:
        sys.exit(1)
"""


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


@pytest.fixture
def mcp_calculator_server():
    """Factory fixture that starts a FastMCP calculator server (add/subtract tools) in a subprocess.

    Call the yielded function with a transport ("sse" or "streamable-http"); it spawns the server,
    waits until it is actually accepting connections, and returns the port it is listening on. All
    spawned servers and their temporary script files are cleaned up automatically when the test ends.

    Build the client URL from the returned port based on the transport:
      - sse:             http://127.0.0.1:{port}/sse
      - streamable-http: http://127.0.0.1:{port}/mcp
    """
    processes = []
    script_paths = []

    def _start(transport: str = "sse") -> int:
        # Find an available port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
            temp_file.write(_CALCULATOR_SERVER_TEMPLATE.format(port=port, transport=transport))
            script_path = temp_file.name
        script_paths.append(script_path)

        # Use sys.executable rather than a bare "python" so we launch the exact interpreter
        # running the tests (more reliable across platforms/venvs).
        process = subprocess.Popen(
            [sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        processes.append(process)

        # Wait until the server is actually accepting connections instead of sleeping a fixed amount.
        # A fixed sleep races with server startup on slow runners and surfaces as "connection refused".
        deadline = time.time() + 30
        while True:
            if process.poll() is not None:
                # Server exited before it became ready - surface its output so the failure is diagnosable.
                stderr = process.stderr.read() if process.stderr else ""
                message = f"MCP server process exited early (code {process.returncode}):\n{stderr}"
                raise RuntimeError(message)
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError as e:
                if time.time() > deadline:
                    message = f"MCP server did not start listening on port {port} within 30s"
                    raise TimeoutError(message) from e
                time.sleep(0.2)

        return port

    yield _start

    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    for script_path in script_paths:
        if os.path.exists(script_path):
            os.remove(script_path)


@pytest.fixture(autouse=True)
def allow_deserialization_of_test_modules(monkeypatch):
    """
    haystack-ai >= 3.0 refuses to deserialize classes and callables from modules outside its
    trusted-module allowlist. Tools and callbacks defined in the test modules live outside that
    allowlist, so trust them explicitly; haystack-ai 2.x ignores this environment variable.
    """
    monkeypatch.setenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", "tests,test_*")
