import json
import os

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.tools.errors import ToolInvocationError
from haystack.tools.from_function import tool

from haystack_integrations.tools.mcp import (
    MCPTool,
    StdioServerInfo,
)

from .mcp_memory_transport import InMemoryServerInfo
from .mcp_servers_fixtures import calculator_mcp, echo_mcp


@tool
def simple_haystack_tool(name: str) -> str:
    """A simple Haystack tool for comparison."""
    return f"Hello, {name}!"


class TestMCPTool:
    """Tests for the MCPTool class using in-memory servers."""

    @pytest.fixture
    def mcp_add_tool(self, mcp_tool_cleanup):
        """Provides an MCPTool instance for the 'add' tool using the in-memory calculator server."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        # The MCPTool constructor will fetch the tool's schema from the in-memory server
        tool = MCPTool(name="add", server_info=server_info, eager_connect=True)
        return mcp_tool_cleanup(tool)

    @pytest.fixture
    def mcp_echo_tool(self, mcp_tool_cleanup):
        """Provides an MCPTool instance for the 'echo' tool using the in-memory echo server."""
        server_info = InMemoryServerInfo(server=echo_mcp._mcp_server)
        tool = MCPTool(name="echo", server_info=server_info, eager_connect=True)
        return mcp_tool_cleanup(tool)

    @pytest.fixture
    def mcp_error_tool(self, mcp_tool_cleanup):
        """Provides an MCPTool instance for the 'divide_by_zero' tool for error testing."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(name="divide_by_zero", server_info=server_info, eager_connect=True)
        return mcp_tool_cleanup(tool)

    # New tests using in-memory approach will be added below
    def test_mcp_tool_initialization(self, mcp_add_tool, mcp_echo_tool):
        """Test MCPTool initialization with in-memory servers."""
        # Test add tool initialization
        assert mcp_add_tool.name == "add"
        assert mcp_add_tool.description == "Add two integers."  # Fetched from server
        assert "a" in mcp_add_tool.parameters["properties"]
        assert mcp_add_tool.parameters["properties"]["a"]["type"] == "integer"
        assert "b" in mcp_add_tool.parameters["properties"]
        assert mcp_add_tool.parameters["properties"]["b"]["type"] == "integer"
        assert mcp_add_tool.parameters["required"] == ["a", "b"]
        assert isinstance(mcp_add_tool._server_info, InMemoryServerInfo)

        # Test echo tool initialization
        assert mcp_echo_tool.name == "echo"
        assert mcp_echo_tool.description == "Echo the input text."  # Fetched from server
        assert "text" in mcp_echo_tool.parameters["properties"]
        assert mcp_echo_tool.parameters["properties"]["text"]["type"] == "string"
        assert mcp_echo_tool.parameters["required"] == ["text"]
        assert isinstance(mcp_echo_tool._server_info, InMemoryServerInfo)

    def test_mcp_tool_invoke(self, mcp_add_tool, mcp_echo_tool):
        """Test invoking MCPTools connected to in-memory servers."""
        # Test add tool invocation
        add_result = mcp_add_tool.invoke(a=25, b=17)
        add_result = json.loads(add_result)
        assert add_result["content"][0]["text"] == "42"

        # Test echo tool invocation
        echo_result = mcp_echo_tool.invoke(text="Hello MCP!")
        echo_result = json.loads(echo_result)
        assert echo_result["content"][0]["text"] == "Hello MCP!"

    def test_mcp_tool_error_handling(self, mcp_error_tool):
        """Test error handling with the in-memory server."""
        with pytest.raises(ToolInvocationError) as exc_info:
            mcp_error_tool.invoke(a=10)  # Invokes divide_by_zero

        # Check the actual error message content
        error_message = str(exc_info.value)
        # The first part of the message comes from ToolInvocationError's formatting
        assert "Failed to invoke Tool `divide_by_zero`" in error_message

    def test_mcp_tool_serde(self, mcp_tool_cleanup):
        """Test serialization and deserialization of MCPTool with in-memory server."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)

        tool = MCPTool(
            name="add", server_info=server_info, description="Addition tool for serde testing", eager_connect=True
        )
        # Register tool for cleanup
        mcp_tool_cleanup(tool)

        # Test serialization (to_dict)
        tool_dict = tool.to_dict()

        # Verify serialization format
        assert tool_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.MCPTool"
        assert tool_dict["data"]["name"] == "add"
        assert tool_dict["data"]["description"] == "Addition tool for serde testing"
        assert tool_dict["data"]["server_info"]["type"] == "tests.mcp_memory_transport.InMemoryServerInfo"

        # MCP Tool does not preserve the parameters field, but recreates it from mcp server on re-initialization
        # see below for more details
        assert "parameters" not in tool_dict["data"]

        # Test deserialization (from_dict)
        new_tool = MCPTool.from_dict(tool_dict)
        mcp_tool_cleanup(new_tool)

        assert new_tool.name == "add"
        assert new_tool.description == "Addition tool for serde testing"

        # Recreated parameters from mcp server on re-initialization
        assert new_tool.parameters is not None
        assert new_tool.parameters == {
            "properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}},
            "required": ["a", "b"],
            "title": "addArguments",
            "type": "object",
        }

        assert isinstance(new_tool._server_info, InMemoryServerInfo)

    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_pipeline_warmup_with_mcp_tool(self):
        """Test lazy connection with Pipeline.warm_up() - replicates time_pipeline.py."""

        # Replicate time_pipeline.py using MCPTool instead of MCPToolset
        server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])

        # Create tool with lazy connection (default behavior)
        tool = MCPTool(name="get_current_time", server_info=server_info)
        try:
            # Build pipeline with Agent, Pipeline will warm up the tool in the agent automatically
            agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"), tools=[tool])
            pipeline = Pipeline()
            pipeline.add_component("agent", agent)

            user_input_msg = ChatMessage.from_user(text="What is the time in New York?")
            result = pipeline.run({"agent": {"messages": [user_input_msg]}})
            assert "New York" in result["agent"]["messages"][3].text
        finally:
            if tool:
                tool.close()
