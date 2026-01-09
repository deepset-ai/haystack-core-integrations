import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import pytest
import pytest_asyncio
from haystack import logging
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo
from haystack_integrations.tools.mcp.mcp_tool import (
    MCPConnectionError,
    MCPToolNotFoundError,
    SSEServerInfo,
    StreamableHttpServerInfo,
)
from haystack_integrations.tools.mcp.mcp_toolset import (
    _deserialize_state_config,
    _serialize_state_config,
)

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
        eager_connect=True,
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
        eager_connect=True,
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
        eager_connect=True,
    )

    return mcp_tool_cleanup(toolset)


def format_result(result):
    """Sample handler function for testing."""
    return f"FORMATTED: {result}"


@pytest_asyncio.fixture
async def calculator_toolset_with_state_config(mcp_tool_cleanup):
    """Fixture that provides an MCPToolset with state configuration."""

    server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
    toolset = MCPToolset(
        server_info=server_info,
        tool_names=["add", "subtract"],
        connection_timeout=45,
        invocation_timeout=60,
        eager_connect=True,
        inputs_from_state={
            "add": {"first_number": "a"},
            "subtract": {"first_number": "a", "second_number": "b"},
        },
        outputs_to_state={
            "add": {"sum_result": {"source": "content"}},
            "subtract": {"diff_result": {}},
        },
        outputs_to_string={
            "add": {"source": "content", "handler": format_result},
        },
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
                eager_connect=True,
            )

    async def test_toolset_tool_not_found(self):
        """Test that requesting a non-existent tool raises a MCPToolNotFoundError."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)

        with pytest.raises(MCPToolNotFoundError, match=r"The following tools were not found.*"):
            MCPToolset(
                server_info=server_info,
                tool_names=["non_existent_tool"],
                connection_timeout=10,
                invocation_timeout=10,
                eager_connect=True,
            )

    async def test_toolset_with_state_config(self, calculator_toolset_with_state_config):
        """Test that MCPToolset correctly passes state configuration to tools."""
        toolset = calculator_toolset_with_state_config

        # Verify toolset has state configs stored
        assert toolset.inputs_from_state == {
            "add": {"first_number": "a"},
            "subtract": {"first_number": "a", "second_number": "b"},
        }
        assert "add" in toolset.outputs_to_state
        assert "subtract" in toolset.outputs_to_state
        assert "add" in toolset.outputs_to_string

        # Verify tools have correct state configurations
        add_tool = next(tool for tool in toolset.tools if tool.name == "add")
        subtract_tool = next(tool for tool in toolset.tools if tool.name == "subtract")

        assert add_tool.inputs_from_state == {"first_number": "a"}
        assert subtract_tool.inputs_from_state == {"first_number": "a", "second_number": "b"}
        assert add_tool.outputs_to_state == {"sum_result": {"source": "content"}}
        assert subtract_tool.outputs_to_state == {"diff_result": {}}
        assert add_tool.outputs_to_string is not None
        assert subtract_tool.outputs_to_string is None

    async def test_toolset_state_config_serde(self, calculator_toolset_with_state_config):
        """Test serialization and deserialization of MCPToolset with state configuration."""
        toolset = calculator_toolset_with_state_config

        toolset_dict = toolset.to_dict()

        # Verify state configs are serialized
        assert toolset_dict["data"]["inputs_from_state"] == {
            "add": {"first_number": "a"},
            "subtract": {"first_number": "a", "second_number": "b"},
        }
        assert toolset_dict["data"]["outputs_to_state"] is not None
        assert toolset_dict["data"]["outputs_to_string"] is not None
        # Handler should be serialized as a string
        assert isinstance(toolset_dict["data"]["outputs_to_string"]["add"]["handler"], str)

        # Test deserialization
        with patch("haystack_integrations.tools.mcp.mcp_toolset.MCPToolset.__init__", return_value=None) as mock_init:
            MCPToolset.from_dict(toolset_dict)

            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args
            assert kwargs["inputs_from_state"] == {
                "add": {"first_number": "a"},
                "subtract": {"first_number": "a", "second_number": "b"},
            }
            assert "add" in kwargs["outputs_to_state"]
            assert "add" in kwargs["outputs_to_string"]
            # Handler should be deserialized back to a callable
            assert callable(kwargs["outputs_to_string"]["add"]["handler"])

    async def test_toolset_state_config_unknown_tool_warning(self, caplog):
        """Test that a warning is logged when state config references unknown tools."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)

        with caplog.at_level("WARNING"):
            toolset = MCPToolset(
                server_info=server_info,
                tool_names=["add"],  # Only include add
                connection_timeout=10,
                invocation_timeout=10,
                eager_connect=True,
                inputs_from_state={
                    "add": {"first_number": "a"},
                    "unknown_tool": {"some_key": "some_param"},  # This tool doesn't exist
                },
            )

            # The warning should be logged
            assert any("unknown_tool" in record.message for record in caplog.records)
            toolset.close()

    async def test_toolset_no_state_config(self, calculator_toolset):
        """Test that tools have no state config when none is provided."""
        toolset = calculator_toolset

        for tool in toolset.tools:
            assert tool.inputs_from_state is None
            assert tool.outputs_to_state is None
            assert tool.outputs_to_string is None

    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    async def test_pipeline_warmup_with_mcp_toolset(self):
        """Test lazy connection with Pipeline.warm_up() - replicates time_pipeline.py."""

        # Replicate time_pipeline.py using calculator instead of time server
        server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])

        # Create toolset with lazy connection (default behavior)
        toolset = MCPToolset(server_info=server_info)
        try:
            # Build pipeline exactly like time_pipeline.py
            agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"), tools=toolset)
            pipeline = Pipeline()
            pipeline.add_component("agent", agent)

            user_input_msg = ChatMessage.from_user(text="What is the time in New York?")
            result = pipeline.run({"agent": {"messages": [user_input_msg]}})
            assert "New York" in result["agent"]["messages"][3].text
        finally:
            if toolset:
                toolset.close()


@pytest.mark.integration
class TestMCPToolsetIntegration:
    """Integration tests for MCPToolset."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows fails for some reason")
    def test_toolset_with_sse_connection(self):
        """Test MCPToolset with an SSE connection to a simple server."""

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
            toolset = MCPToolset(server_info=server_info, eager_connect=True)
            # Verify we got both tools
            assert len(toolset) == 2

            tool_names = [tool.name for tool in toolset.tools]
            assert "add" in tool_names
            assert "subtract" in tool_names

            # Test the add tool
            add_tool = next(tool for tool in toolset.tools if tool.name == "add")
            result_json = add_tool.invoke(a=5, b=3)

            # Parse the JSON result
            result = json.loads(result_json)
            assert result["content"][0]["text"] == "8"

            # Test the subtract tool
            subtract_tool = next(tool for tool in toolset.tools if tool.name == "subtract")
            result_json = subtract_tool.invoke(a=10, b=4)

            # Parse the JSON result
            result = json.loads(result_json)
            assert result["content"][0]["text"] == "6"

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

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows fails for some reason")
    def test_toolset_with_streamable_http_connection(self):
        """Test MCPToolset with a streamable-http connection to a simple server."""

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
        mcp.run(transport="streamable-http")
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

            # Create the toolset - note the /mcp endpoint for streamable-http
            server_info = StreamableHttpServerInfo(url=f"http://127.0.0.1:{port}/mcp")
            toolset = MCPToolset(server_info=server_info, eager_connect=True)

            # Verify we got both tools
            assert len(toolset) == 2

            tool_names = [tool.name for tool in toolset.tools]
            assert "add" in tool_names
            assert "subtract" in tool_names

            # Test the add tool
            add_tool = next(tool for tool in toolset.tools if tool.name == "add")
            result_json = add_tool.invoke(a=5, b=3)

            # Parse the JSON result
            result = json.loads(result_json)
            assert result["content"][0]["text"] == "8"

            # Test the subtract tool
            subtract_tool = next(tool for tool in toolset.tools if tool.name == "subtract")
            result_json = subtract_tool.invoke(a=10, b=4)

            # Parse the JSON result
            result = json.loads(result_json)
            assert result["content"][0]["text"] == "6"

        except Exception:
            # Check server output for clues
            if server_process and server_process.poll() is None:
                server_process.terminate()
            raise

        finally:
            # Explicitly close tools first to prevent connection errors
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

    def test_pipeline_deserialization_succeeds_with_lazy_connection(self, monkeypatch):
        """
        Test that pipeline deserialization succeeds with lazy connection (eager_connect=False).

        With lazy connection (the default), MCPToolset defers connection until warm_up() is called.
        This allows pipelines to be deserialized even when the server is not available or
        credentials are not yet resolved.

        This test demonstrates that:
        - Pipeline deserialization succeeds even with an invalid token
        - MCPToolset creates a placeholder tool during initialization
        - Actual connection happens later during warm_up()
        """
        pipeline_yaml = """
components:
  agent:
    init_parameters:
      chat_generator:
        init_parameters:
          api_base_url:
          api_key:
            env_vars:
            - OPENAI_API_KEY
            strict: false
            type: env_var
          generation_kwargs: {}
          max_retries:
          model: gpt-4o
          organization:
          streaming_callback:
          timeout:
          tools:
          tools_strict: false
        type: haystack.components.generators.chat.openai.OpenAIChatGenerator
      exit_conditions:
      - text
      max_agent_steps: 100
      raise_on_tool_invocation_failure: false
      state_schema: {}
      streaming_callback:
      system_prompt: |-
        You are an assistant that summarizes latest issues and PRs on a github repository
        that happened within a certain time frame (e.g. last day or last week). Make sure
        that you always use the current date as a basis for the time frame. Iterate over
        issues and PRs where necessary to get a comprehensive overview.
      tools:
        data:
          server_info:
            type: haystack_integrations.tools.mcp.mcp_tool.StreamableHttpServerInfo
            url: https://api.githubcopilot.com/mcp/
            token:
              env_vars:
              - PERSONAL_ACCESS_TOKEN_GITHUB
              strict: true
              type: env_var
            timeout: 10
          tool_names: [get_issue, get_issue_comments]
        type: haystack_integrations.tools.mcp.MCPToolset
    type: haystack.components.agents.agent.Agent

connections: []
"""
        monkeypatch.setenv("PERSONAL_ACCESS_TOKEN_GITHUB", "SOME_OBVIOUSLY_INVALID_TOKEN")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-key-for-testing")

        # Deserialization should succeed because eager_connect defaults to False
        # With lazy connection, MCPToolset creates a placeholder tool and doesn't try to connect
        pipeline = Pipeline.loads(pipeline_yaml)

        # Verify the pipeline was created successfully
        assert pipeline is not None
        agent = pipeline.get_component("agent")
        assert agent is not None

        # The key point is that deserialization succeeded even with an invalid token
        # because the connection is deferred until warm_up() is called


class TestStateConfigHelpers:
    """Tests for the state configuration serialization helper functions."""

    def test_serialize_outputs_to_string_with_handler(self):
        """Test serializing outputs_to_string config with a handler function."""
        config = {
            "add": {"source": "content", "handler": format_result},
            "subtract": {"source": "diff"},
        }

        serialized = _serialize_state_config(config)

        assert serialized is not None
        assert "add" in serialized
        assert "subtract" in serialized
        assert isinstance(serialized["add"]["handler"], str)  # Handler serialized to string
        assert serialized["subtract"]["source"] == "diff"
        assert "handler" not in serialized["subtract"]  # No handler for subtract

    def test_serialize_outputs_to_state_with_handler(self):
        """Test serializing outputs_to_state config with a handler function."""
        config = {
            "add": {
                "sum_result": {"source": "content", "handler": format_result},
                "raw_result": {},
            },
        }

        serialized = _serialize_state_config(config)

        assert serialized is not None
        assert "add" in serialized
        assert isinstance(serialized["add"]["sum_result"]["handler"], str)
        assert serialized["add"]["raw_result"] == {}

    def test_serialize_empty_config(self):
        """Test that empty config returns None."""
        assert _serialize_state_config({}) is None
        assert _serialize_state_config(None) is None

    def test_deserialize_outputs_to_string_with_handler(self):
        """Test deserializing outputs_to_string config with a handler function."""
        # First serialize to get the correct handler path
        original = {"add": {"source": "content", "handler": format_result}}
        serialized = _serialize_state_config(original)

        # Now deserialize
        deserialized = _deserialize_state_config(serialized)

        assert "add" in deserialized
        assert callable(deserialized["add"]["handler"])
        assert deserialized["add"]["source"] == "content"

    def test_deserialize_outputs_to_state_with_handler(self):
        """Test deserializing outputs_to_state config with a handler function."""
        # First serialize to get the correct handler path
        original = {"add": {"sum_result": {"source": "content", "handler": format_result}}}
        serialized = _serialize_state_config(original)

        # Now deserialize
        deserialized = _deserialize_state_config(serialized)

        assert "add" in deserialized
        assert callable(deserialized["add"]["sum_result"]["handler"])

    def test_deserialize_empty_config(self):
        """Test that empty config returns empty dict."""
        assert _deserialize_state_config({}) == {}
        assert _deserialize_state_config(None) == {}

    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization are inverse operations."""
        original = {
            "add": {"source": "content", "handler": format_result},
            "subtract": {"source": "diff"},
        }

        serialized = _serialize_state_config(original)
        deserialized = _deserialize_state_config(serialized)

        assert "add" in deserialized
        assert "subtract" in deserialized
        assert deserialized["add"]["source"] == "content"
        assert callable(deserialized["add"]["handler"])
        assert deserialized["subtract"]["source"] == "diff"
