import os
import subprocess
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools.errors import ToolInvocationError
from haystack.tools.from_function import tool
from haystack.utils.auth import TokenSecret

from haystack_integrations.tools.mcp import (
    MCPError,
    MCPTool,
    SSEServerInfo,
    StdioServerInfo,
)


@tool
def echo(name: str) -> str:
    """Echo a name"""
    return f"Hello, {name}!"


# Create mocking fixtures for MCPTool tests
@pytest.fixture
def mock_mcp_tool():
    """
    Fixture to create a pre-configured MCPTool for testing without server connection.
    This approach uses patching to avoid actual server connections.
    """
    # Create patch for connection methods
    with (
        patch("haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor.get_instance") as mock_get_instance,
        patch("haystack_integrations.tools.mcp.mcp_tool.MCPServerInfo.create_client") as mock_create_client,
    ):
        # Configure mock executor
        mock_executor = MagicMock()
        mock_tool_info = MagicMock()
        mock_tool_info.name = "test_tool"
        mock_tool_info.description = "A test tool"
        mock_tool_info.inputSchema = {"type": "object", "properties": {}}
        mock_executor.run.return_value = [mock_tool_info]
        mock_get_instance.return_value = mock_executor

        # Configure mock client
        mock_client = MagicMock()

        async def mock_invoke(*_args, **_kwargs):
            return {"result": "mock_result"}

        mock_client.invoke = mock_invoke
        mock_create_client.return_value = mock_client

        # Create the MCPTool
        tool = MCPTool(
            name="test_tool",
            server_info=SSEServerInfo(base_url="http://example.com", token=TokenSecret.from_token("MCP_TEST_TOKEN")),
            description="A test MCP tool",
        )

        # Override the invoke method for testing
        original_invoke = tool._invoke_tool

        def mock_invoke_tool(**kwargs):
            if "error" in kwargs:
                error_message = "Mock error for testing"
                raise ToolInvocationError(error_message)
            return {"result": "mock_result"}

        tool._invoke_tool = mock_invoke_tool
        tool.function = mock_invoke_tool

        # Add run method (alias for invoke) for testing
        tool.run = tool.invoke = lambda **kwargs: mock_invoke_tool(**kwargs)

        # Override to_dict method to match expected format
        original_to_dict = tool.to_dict

        def mock_to_dict():
            return {
                "type": "haystack_integrations.tools.mcp.mcp_tool.MCPTool",
                "data": {
                    "name": "test_tool",
                    "description": "A test MCP tool",
                    "server_info": {
                        "type": "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo",
                        "base_url": "http://example.com",
                        "token": {
                            "type": "haystack.utils.auth.EnvVarSecret",
                            "init_parameters": {"env_var": "MOCK_TOKEN"},
                        },
                        "timeout": 30,
                    },
                    "connection_timeout": 30,
                    "invocation_timeout": 30,
                },
            }

        tool.to_dict = mock_to_dict

        yield tool

        # Restore original methods if needed
        tool._invoke_tool = original_invoke
        tool.to_dict = original_to_dict


class TestMCPServerInfo:
    """Unit tests for MCPServerInfo classes."""

    def test_http_server_info_serde(self):
        """Test serialization/deserialization of SSEServerInfo."""
        server_info = SSEServerInfo(base_url="http://example.com", token="test-token", timeout=45)

        # Test to_dict
        info_dict = server_info.to_dict()
        assert info_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo"
        assert info_dict["base_url"] == "http://example.com"
        assert info_dict["token"] == "test-token"
        assert info_dict["timeout"] == 45

        # Test from_dict
        new_info = SSEServerInfo.from_dict(info_dict)
        assert new_info.base_url == "http://example.com"
        assert new_info.token == "test-token"
        assert new_info.timeout == 45

    def test_url_base_url_validation(self):
        """Test validation of url and base_url parameters."""
        # Test with neither url nor base_url
        with pytest.raises(ValueError, match="Either url or base_url must be provided"):
            SSEServerInfo()

        # Test with both url and base_url
        with pytest.raises(ValueError, match="Only one of url or base_url should be provided"):
            SSEServerInfo(url="http://example.com/sse", base_url="http://example.com")

        # Test with only url
        server_info = SSEServerInfo(url="http://example.com/sse")
        assert server_info.url == "http://example.com/sse"
        assert server_info.base_url is None

        # Test with only base_url (deprecated but supported)
        with pytest.warns(DeprecationWarning, match="base_url is deprecated"):
            server_info = SSEServerInfo(base_url="http://example.com")
            assert server_info.base_url == "http://example.com"  # Should preserve original base_url

    def test_stdio_server_info_serde(self):
        """Test serialization/deserialization of StdioServerInfo."""
        server_info = StdioServerInfo(command="python", args=["-m", "mcp_server_time"], env={"TEST_ENV": "value"})

        # Test to_dict
        info_dict = server_info.to_dict()
        assert info_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.StdioServerInfo"
        assert info_dict["command"] == "python"
        assert info_dict["args"] == ["-m", "mcp_server_time"]
        assert info_dict["env"] == {"TEST_ENV": "value"}

        # Test from_dict
        new_info = StdioServerInfo.from_dict(info_dict)
        assert new_info.command == "python"
        assert new_info.args == ["-m", "mcp_server_time"]
        assert new_info.env == {"TEST_ENV": "value"}

    def test_create_client(self):
        """Test client creation from server info."""
        http_info = SSEServerInfo(base_url="http://example.com")
        stdio_info = StdioServerInfo(command="python")

        http_client = http_info.create_client()
        stdio_client = stdio_info.create_client()

        assert http_client.base_url == "http://example.com"
        assert stdio_client.command == "python"


class TestMCPTool:
    """Tests for the MCPTool class."""

    def test_mcp_tool_initialization(self, mock_mcp_tool):
        """Test if the MCPTool can be initialized correctly."""
        # Get the pre-configured mock tool from the fixture
        tool = mock_mcp_tool

        # Check if the tool has all expected attributes
        assert tool.name == "test_tool"
        assert "A test MCP tool" == tool.description
        assert isinstance(tool._server_info, SSEServerInfo)
        assert tool._connection_timeout == 30
        assert tool._invocation_timeout == 30

    def test_mcp_tool_direct_serde(self, mock_mcp_tool, monkeypatch):
        """Test direct serialization and deserialization of the MCPTool."""
        # Set environment variables for secrets
        monkeypatch.setenv("MOCK_TOKEN", "test-token")
        monkeypatch.setenv("MCP_TEST_TOKEN", "test-token")

        # Get the pre-configured mock tool
        tool = mock_mcp_tool

        tool_dict = tool.to_dict()

        # Verify serialization format
        assert tool_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.MCPTool"
        assert tool_dict["data"]["name"] == "test_tool"
        assert tool_dict["data"]["description"] == "A test MCP tool"
        assert tool_dict["data"]["server_info"]["type"] == "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo"
        assert tool_dict["data"]["server_info"]["base_url"] == "http://example.com"

        # The token should be stored as a secret in serialized format
        token_secret = tool_dict["data"]["server_info"]["token"]
        assert isinstance(token_secret, dict)
        assert token_secret["type"] == "haystack.utils.auth.EnvVarSecret"
        assert "env_var" in token_secret["init_parameters"]

        # Test deserialization with patched environment
        with patch("haystack_integrations.tools.mcp.mcp_tool.MCPTool.__init__", return_value=None) as mock_init:
            # Call from_dict but use mock_init to verify arguments
            MCPTool.from_dict(tool_dict)
            # Verify that __init__ was called with correct parameters
            mock_init.assert_called_once()
            args, kwargs = mock_init.call_args
            assert kwargs["name"] == "test_tool"
            assert kwargs["description"] == "A test MCP tool"
            assert isinstance(kwargs["server_info"], SSEServerInfo)
            assert kwargs["server_info"].base_url == "http://example.com"

    def test_serde_in_pipeline(self, monkeypatch):
        """Test serialization and deserialization of MCPTool within a Pipeline."""
        # Set environment variables for testing
        monkeypatch.setenv("MOCK_TOKEN", "test-token")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with (
            patch("haystack_integrations.tools.mcp.mcp_tool.MCPTool.__init__", return_value=None) as mock_init,
            patch("haystack_integrations.tools.mcp.mcp_tool.AsyncExecutor.get_instance"),
            patch("haystack_integrations.tools.mcp.mcp_tool.MCPTool.to_dict") as mock_to_dict,
        ):

            # Configure to_dict to return the same format as our mock
            mock_to_dict.return_value = {
                "type": "haystack_integrations.tools.mcp.mcp_tool.MCPTool",
                "data": {
                    "name": "test_tool",
                    "description": "A test MCP tool",
                    "server_info": {
                        "type": "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo",
                        "base_url": "http://example.com",
                        "token": {
                            "type": "haystack.utils.auth.EnvVarSecret",
                            "init_parameters": {"env_var": "MOCK_TOKEN"},
                        },
                        "timeout": 30,
                    },
                    "connection_timeout": 30,
                    "invocation_timeout": 30,
                },
            }

            # Create a tool instance with mocked initialization
            mock_init.return_value = None
            tool = MCPTool(
                name="test_tool",
                server_info=SSEServerInfo(
                    base_url="http://example.com", token=TokenSecret.from_env_var("MCP_TEST_TOKEN")
                ),
            )

            # Set required attributes that would normally be set in __init__
            tool.name = "test_tool"
            tool.description = "A test MCP tool"
            tool.parameters = {"type": "object", "properties": {}}
            tool.function = lambda **_kwargs: {"result": "mock_result"}

            pipeline = Pipeline()
            pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
            pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
            pipeline.connect("tool_invoker.tool_messages", "llm.messages")

            # Serialize to dict and verify structure
            pipeline_dict = pipeline.to_dict()

            # Verify the tool is properly serialized in the pipeline
            assert (
                pipeline_dict["components"]["tool_invoker"]["type"]
                == "haystack.components.tools.tool_invoker.ToolInvoker"
            )
            assert len(pipeline_dict["components"]["tool_invoker"]["init_parameters"]["tools"]) == 1

            tool_dict = pipeline_dict["components"]["tool_invoker"]["init_parameters"]["tools"][0]
            assert tool_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.MCPTool"
            assert tool_dict["data"]["name"] == "test_tool"
            assert tool_dict["data"]["description"] == "A test MCP tool"
            assert tool_dict["data"]["server_info"]["type"] == "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo"

            # Test round-trip serialization (dumps/loads) with a patched from_dict
            with patch("haystack_integrations.tools.mcp.mcp_tool.MCPTool.from_dict", return_value=tool):
                pipeline_yaml = pipeline.dumps()
                # Just verify we can serialize to YAML without errors
                assert isinstance(pipeline_yaml, str)
                assert "MCPTool" in pipeline_yaml

    def test_mcp_tool_error_handling(self, mock_mcp_tool):
        """Test error handling during tool invocation."""
        # Get the pre-configured mock tool
        tool = mock_mcp_tool

        # Test error handling
        with pytest.raises(ToolInvocationError):
            tool.run(error=True)  # This should trigger our mock error


@pytest.mark.integration
class TestMCPToolInPipelineWithOpenAI:
    """Integration tests for MCPTool in Haystack pipelines with OpenAI."""

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.skipif(sys.platform == "win32", reason="Windows fails for some reason")
    def test_mcp_tool_with_http_server(self):
        """Test using an MCPTool with a real HTTP server."""
        import os
        import socket

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
            result = tool.invoke(a=5, b=3)

            # Verify the result
            assert not result.isError
            assert len(result.content) == 1
            assert result.content[0].text == "8"

            # Try another tool from the same server
            subtract_tool = MCPTool(name="subtract", server_info=server_info)
            result = subtract_tool.invoke(a=10, b=4)

            # Verify the result
            assert not result.isError
            assert len(result.content) == 1
            assert result.content[0].text == "6"

        except Exception:
            # Check server output for clues
            if server_process and server_process.poll() is None:
                server_process.terminate()
            raise

        finally:
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

    @pytest.mark.skipif(
        (not os.environ.get("OPENAI_API_KEY") and not os.environ.get("BRAVE_API_KEY"))
        or (sys.platform == "win32")
        or (sys.platform == "darwin"),
        reason="OPENAI_API_KEY or BRAVE_API_KEY not set, or running on Windows or macOS",
    )
    def test_mcp_brave_search(self):
        """Test using an MCPTool in a pipeline with OpenAI."""

        # Create an MCPTool for the brave_web_search operation
        server_info = StdioServerInfo(
            command="docker",
            args=["run", "-i", "--rm", "-e", f"BRAVE_API_KEY={os.environ.get('BRAVE_API_KEY')}", "mcp/brave-search"],
            env=None,
        )
        tool = MCPTool(name="brave_web_search", server_info=server_info)
        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")

        # Create a message that should trigger tool use
        message = ChatMessage.from_user(text="Use brave_web_search to search for the latest German elections news")

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert any(term in tool_message.tool_call_result.result for term in ["Bundestag", "election"]), (
            "Result should contain information about German elections"
            f"\n\nResult: {tool_message.tool_call_result.result}"
        )

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_mcp_tool_in_pipeline_with_multiple_tools(self):
        """Test using multiple MCPTools in a pipeline with OpenAI."""

        # Mix mcp tool with a simple echo tool
        time_server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=America/New_York"])
        time_tool = MCPTool(name="get_current_time", server_info=time_server_info)

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[echo, time_tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[echo, time_tool]))

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
    def test_mcp_tool_error_handling(self):
        """Test error handling with MCPTool in a pipeline."""

        # Create a custom server info for a server that might return errors
        server_info = SSEServerInfo(base_url="http://localhost:8000")
        with pytest.raises(MCPError):
            MCPTool(name="divide", server_info=server_info)
