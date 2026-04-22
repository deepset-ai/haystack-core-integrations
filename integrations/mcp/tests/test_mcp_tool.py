import asyncio
import io
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from exceptiongroup import ExceptionGroup
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.tools.errors import ToolInvocationError
from haystack.tools.from_function import tool

from haystack_integrations.tools.mcp import (
    MCPTool,
    MCPToolNotFoundError,
    StdioServerInfo,
)
from haystack_integrations.tools.mcp.mcp_tool import (
    AsyncExecutor,
    MCPClient,
    MCPConnectionError,
    MCPInvocationError,
    StdioClient,
    _extract_first_text_element,
    _MCPClientSessionManager,
)

from .mcp_memory_transport import InMemoryServerInfo
from .mcp_servers_fixtures import calculator_mcp, echo_mcp, image_mcp, state_calculator_mcp


@tool
def simple_haystack_tool(name: str) -> str:
    """A simple Haystack tool for comparison."""
    return f"Hello, {name}!"


# from https://modelcontextprotocol.io/specification/draft/server/tools#output-schema
EXAMPLE_MCP_TOOL_CALL_RESULT = {
    "content": [{"type": "text", "text": '{"temperature": 22.5, "conditions": "Partly cloudy", "humidity": 65}'}],
    "structuredContent": {"temperature": 22.5, "conditions": "Partly cloudy", "humidity": 65},
}


def test_extract_first_text_element():
    """Test that extract_first_text skips non-text blocks and parses the first text block."""
    tool_call_result = EXAMPLE_MCP_TOOL_CALL_RESULT
    tool_call_result["content"].insert(0, {"type": "image", "data": "ignored"})
    tool_call_result["content"].insert(1, {"type": "text", "text": '{"answer": 42}'})  # target
    tool_call_result = json.dumps(tool_call_result)

    extracted = _extract_first_text_element(tool_call_result)

    assert extracted == {"answer": 42}


def test_async_executor_run_raises_timeout_error():
    async def never() -> None:
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError, match="timed out"):
        AsyncExecutor.get_instance().run(never(), timeout=0.01)


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

    def test_mcp_tool_outputs_to_state_falls_back_to_full_response_for_non_text_content(self, mcp_tool_cleanup):
        """Test that non-text MCP content returns the full parsed response when state output is enabled."""
        server_info = InMemoryServerInfo(server=image_mcp._mcp_server)
        tool = MCPTool(
            name="image_tool",
            server_info=server_info,
            eager_connect=True,
            outputs_to_state={"image_payload": {}},
        )
        mcp_tool_cleanup(tool)

        result = tool.invoke()

        assert isinstance(result, dict)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "image"
        assert result["content"][0]["data"] == "ZmFrZQ=="
        assert result["content"][0]["mimeType"] == "image/png"
        assert result["isError"] is False

    def test_mcp_tool_outputs_to_state_returns_raw_text_when_text_is_not_json(self, mcp_tool_cleanup):
        """Test that plain text content is returned as-is when state output parsing cannot decode JSON."""
        server_info = InMemoryServerInfo(server=echo_mcp._mcp_server)
        tool = MCPTool(
            name="echo",
            server_info=server_info,
            eager_connect=True,
            outputs_to_state={"echo_payload": {}},
        )
        mcp_tool_cleanup(tool)

        result = tool.invoke(text="Hello MCP!")

        assert result == "Hello MCP!"

    def test_mcp_tool_error_handling(self, mcp_error_tool):
        """Test error handling with the in-memory server."""
        with pytest.raises(ToolInvocationError) as exc_info:
            mcp_error_tool.invoke(a=10)  # Invokes divide_by_zero

        # Check the actual error message content
        error_message = str(exc_info.value)
        # The first part of the message comes from ToolInvocationError's formatting
        assert "Failed to invoke Tool `divide_by_zero`" in error_message

    def test_mcp_tool_lazy_missing_tool_raises_with_available_tools(self, mcp_tool_cleanup):
        """Test that lazy warm-up surfaces missing-tool errors with the available tool names."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(name="multiply", server_info=server_info, eager_connect=False)
        mcp_tool_cleanup(tool)

        mock_worker = MagicMock()
        mock_worker.tools.return_value = [
            SimpleNamespace(name="add"),
            SimpleNamespace(name="subtract"),
            SimpleNamespace(name="divide_by_zero"),
        ]

        with (
            patch("haystack_integrations.tools.mcp.mcp_tool._MCPClientSessionManager", return_value=mock_worker),
            pytest.raises(MCPToolNotFoundError) as exc_info,
        ):
            tool.warm_up()

        assert exc_info.value.tool_name == "multiply"
        assert set(exc_info.value.available_tools) == {"add", "subtract", "divide_by_zero"}

    def test_mcp_tool_lazy_no_tools_server_raises_tool_not_found(self, mcp_tool_cleanup):
        """Test that lazy warm-up fails cleanly when the server exposes no tools."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(name="anything", server_info=server_info, eager_connect=False)
        mcp_tool_cleanup(tool)

        mock_worker = MagicMock()
        mock_worker.tools.return_value = []

        with (
            patch("haystack_integrations.tools.mcp.mcp_tool._MCPClientSessionManager", return_value=mock_worker),
            pytest.raises(MCPToolNotFoundError) as exc_info,
        ):
            tool.warm_up()

        assert str(exc_info.value) == "No tools available on server"
        assert exc_info.value.tool_name == "anything"
        assert exc_info.value.available_tools == []

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

    def test_mcp_tool_state_mapping_parameters(self, mcp_tool_cleanup):
        """Test that MCPTool correctly initializes with state-mapping parameters."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)

        # Create tool with state-mapping parameters
        # Map state key "state_a" to tool parameter "a"
        tool = MCPTool(
            name="add",
            server_info=server_info,
            eager_connect=False,
            outputs_to_string={"source": "result", "handler": str},
            inputs_from_state={"state_a": "a"},
            outputs_to_state={"result": {"source": "output", "handler": str}},
        )
        mcp_tool_cleanup(tool)

        # Verify the parameters are stored correctly
        assert tool._outputs_to_string == {"source": "result", "handler": str}
        assert tool._inputs_from_state == {"state_a": "a"}
        assert tool._outputs_to_state == {"result": {"source": "output", "handler": str}}

        # Warm up the tool to trigger schema adjustment
        tool.warm_up()

        # Verify that "a" was removed from parameters since it's in inputs_from_state
        assert "a" not in tool.parameters["properties"]
        assert "a" not in tool.parameters.get("required", [])
        # Verify that "b" is still present (not removed)
        assert "b" in tool.parameters["properties"]
        assert "b" in tool.parameters["required"]

    def test_mcp_tool_eager_state_mapping_removes_inputs_from_schema(self, mcp_tool_cleanup):
        """Test that eager MCPTool initialization removes state-injected params from its public schema."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(
            name="add",
            server_info=server_info,
            eager_connect=True,
            inputs_from_state={"state_a": "a"},
        )
        mcp_tool_cleanup(tool)

        assert "a" not in tool.parameters["properties"]
        assert "a" not in tool.parameters.get("required", [])
        assert "b" in tool.parameters["properties"]
        assert "b" in tool.parameters["required"]

    def test_mcp_tool_serde_with_state_mapping(self, mcp_tool_cleanup):
        """Test serialization and deserialization of MCPTool with state-mapping parameters."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)

        # Create tool with state-mapping parameters
        # The 'add' tool has parameters 'a' and 'b', so we map to 'a'
        tool = MCPTool(
            name="add",
            server_info=server_info,
            eager_connect=False,
            outputs_to_string={"source": "result"},
            inputs_from_state={"state_a": "a"},
            outputs_to_state={"result": {"source": "output"}},
        )
        mcp_tool_cleanup(tool)

        # Test serialization (to_dict)
        tool_dict = tool.to_dict()

        # Verify state-mapping parameters are serialized
        assert tool_dict["data"]["outputs_to_string"] == {"source": "result"}
        assert tool_dict["data"]["inputs_from_state"] == {"state_a": "a"}
        assert tool_dict["data"]["outputs_to_state"] == {"result": {"source": "output"}}

        # Test deserialization (from_dict)
        new_tool = MCPTool.from_dict(tool_dict)
        mcp_tool_cleanup(new_tool)

        # Verify state-mapping parameters are restored
        assert new_tool._outputs_to_string == {"source": "result"}
        assert new_tool._inputs_from_state == {"state_a": "a"}
        assert new_tool._outputs_to_state == {"result": {"source": "output"}}

    @pytest.mark.skipif(
        not hasattr(__import__("haystack.tools", fromlist=["Tool"]).Tool, "_get_valid_inputs"),
        reason="Requires Haystack >= 2.22.0 for inputs_from_state validation",
    )
    def test_mcp_tool_lazy_invalid_parameter_raises_on_warm_up(self, mcp_tool_cleanup):
        """Test that lazy MCPTool defers invalid inputs_from_state validation until warm_up()."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(
            name="add",
            server_info=server_info,
            eager_connect=False,
            inputs_from_state={"state_key": "non_existent_param"},
        )
        mcp_tool_cleanup(tool)

        assert tool.parameters == {"type": "object", "properties": {}, "additionalProperties": True}

        with pytest.raises(ValueError, match="unknown parameter"):
            tool.warm_up()

    def test_mcp_tool_invoke_auto_warms_up_once(self, mcp_tool_cleanup):
        """Test that lazy MCPTool initializes on first invoke and reuses that connection."""
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(name="add", server_info=server_info, eager_connect=False)
        mcp_tool_cleanup(tool)

        assert tool.parameters == {"type": "object", "properties": {}, "additionalProperties": True}

        with patch.object(tool, "_connect_and_initialize", wraps=tool._connect_and_initialize) as mock_connect:
            first_result = json.loads(tool.invoke(a=20, b=22))
            second_result = json.loads(tool.invoke(a=1, b=2))

        assert first_result["content"][0]["text"] == "42"
        assert second_result["content"][0]["text"] == "3"
        assert "a" in tool.parameters["properties"]
        assert "b" in tool.parameters["properties"]
        assert mock_connect.call_count == 1

    @pytest.mark.asyncio
    async def test_mcp_tool_ainvoke_matches_invoke_with_outputs_to_state(self, mcp_tool_cleanup):
        """Test that sync and async invocation paths return the same parsed state output."""
        server_info = InMemoryServerInfo(server=state_calculator_mcp._mcp_server)
        tool = MCPTool(
            name="state_add",
            server_info=server_info,
            eager_connect=True,
            outputs_to_state={"result": {"source": "result"}},
        )
        mcp_tool_cleanup(tool)

        sync_result = tool.invoke(a=20, b=22)
        async_result = await tool.ainvoke(a=20, b=22)

        assert sync_result == {"result": 42}
        assert async_result == sync_result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fileno_side_effect,fileno_return_value,notebook_environment",
        [
            (io.UnsupportedOperation("fileno"), None, True),
            (None, 2, False),
        ],
    )
    async def test_stdio_client_stderr_handling(self, fileno_side_effect, fileno_return_value, notebook_environment):
        """Test that StdioClient uses sys.stderr in terminals and falls back to a file in notebooks."""
        client = StdioClient(command="echo", args=["hello"])

        mock_stderr = MagicMock()
        mock_stderr.fileno.side_effect = fileno_side_effect
        mock_stderr.fileno.return_value = fileno_return_value

        with (
            patch.object(client, "exit_stack") as mock_stack,
            patch("haystack_integrations.tools.mcp.mcp_tool.stdio_client") as mock_stdio_client,
            patch("haystack_integrations.tools.mcp.mcp_tool.sys") as mock_sys,
            patch.object(client, "_initialize_session_with_transport", new_callable=AsyncMock) as mock_init,
        ):
            mock_sys.stderr = mock_stderr
            mock_stack.enter_async_context = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_init.return_value = []

            await client.connect()

            _, kwargs = mock_stdio_client.call_args
            errlog = kwargs["errlog"]
            if notebook_environment:
                assert errlog is not mock_stderr
                assert hasattr(errlog, "write")
            else:
                assert errlog is mock_stderr

    @pytest.mark.asyncio
    async def test_mcp_client_aclose_clears_references_even_when_cleanup_fails(self, caplog):
        """Test that client cleanup always clears connection state, even if exit_stack cleanup raises."""
        client = StdioClient(command="echo")
        client.session = MagicMock()
        client.stdio = MagicMock()
        client.write = MagicMock()
        client.exit_stack = MagicMock()
        client.exit_stack.aclose = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        with caplog.at_level("WARNING"):
            await client.aclose()

        assert any("Error during MCP client cleanup: cleanup failed" in record.message for record in caplog.records)
        assert client.session is None
        assert client.stdio is None
        assert client.write is None

    @pytest.mark.asyncio
    async def test_initialize_session_wraps_errors_as_connection_error(self):
        client = StdioClient(command="echo")

        with patch.object(client.exit_stack, "enter_async_context", new_callable=AsyncMock) as mock_enter:
            mock_enter.side_effect = RuntimeError("boom")
            with pytest.raises(MCPConnectionError, match="Failed to connect to ctx"):
                await client._initialize_session_with_transport((MagicMock(), MagicMock()), "ctx")

    def test_mcp_tool_eager_connect_surfaces_exception_group_inner_message(self):
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        err = ExceptionGroup("outer", [RuntimeError("inner details")])

        with patch(
            "haystack_integrations.tools.mcp.mcp_tool._MCPClientSessionManager",
            side_effect=err,
        ):
            with pytest.raises(MCPConnectionError, match="inner details"):
                MCPTool(name="add", server_info=server_info, eager_connect=True)

    @pytest.mark.asyncio
    async def test_mcp_tool_ainvoke_raises_timeout_error(self, mcp_tool_cleanup):
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(name="add", server_info=server_info, eager_connect=True)
        mcp_tool_cleanup(tool)

        async def hanging(*_args, **_kwargs):
            await asyncio.Event().wait()

        tool._client.call_tool = hanging
        tool._invocation_timeout = 0.01

        with pytest.raises(TimeoutError, match="timed out"):
            await tool.ainvoke(a=1, b=2)

    @pytest.mark.asyncio
    async def test_mcp_tool_ainvoke_wraps_unexpected_errors(self, mcp_tool_cleanup):
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(name="add", server_info=server_info, eager_connect=True)
        mcp_tool_cleanup(tool)

        async def crash(*_args, **_kwargs):
            message = "kaboom"
            raise RuntimeError(message)

        tool._client.call_tool = crash

        with pytest.raises(MCPInvocationError, match="kaboom"):
            await tool.ainvoke(a=1, b=2)

    def test_mcp_tool_close_swallows_worker_stop_exceptions(self, mcp_tool_cleanup):
        server_info = InMemoryServerInfo(server=calculator_mcp._mcp_server)
        tool = MCPTool(name="add", server_info=server_info, eager_connect=True)
        mcp_tool_cleanup(tool)

        with patch.object(tool._worker, "stop", side_effect=RuntimeError("stop failed")):
            tool.close()

    def test_session_manager_propagates_connect_failures(self):
        class BrokenClient(MCPClient):
            async def connect(self):
                message = "broken"
                raise RuntimeError(message)

        client = BrokenClient()

        with pytest.raises(RuntimeError, match="broken"):
            _MCPClientSessionManager(client, timeout=5.0)

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
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

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_agent_with_state_mapping(self):
        """Test Agent with MCPTool using state-mapping to inject location from state."""

        # Create MCPTool with state-mapping that injects home_city from state as timezone parameter
        server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])
        tool = MCPTool(
            name="get_current_time",
            server_info=server_info,
            inputs_from_state={"home_city": "timezone"},  # Inject home_city from state as timezone
        )

        try:
            # Build Agent with state schema that includes home_city
            agent = Agent(
                chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
                tools=[tool],
                state_schema={"home_city": {"type": str}},
            )
            pipeline = Pipeline()
            pipeline.add_component("agent", agent)

            # Ask for time without mentioning the location - it should use home_city from state
            user_input_msg = ChatMessage.from_user(text="What time is it at home?")
            result = pipeline.run(
                {
                    "agent": {
                        "messages": [user_input_msg],
                        "home_city": "America/New_York",  # Inject New York as home city
                    }
                }
            )

            # Verify the agent got the time for New York
            final_message = result["agent"]["messages"][-1].text

            # The response should mention time
            assert any(keyword in final_message.lower() for keyword in ["time", "o'clock", "am", "pm"]), (
                f"Expected time in response: {final_message}"
            )

            # Verify the response mentions New York or Eastern timezone (proving state-mapping injected it)
            # The user never mentioned location, but timezone info should appear in the response
            assert any(keyword in final_message for keyword in ["New York", "New_York"]), (
                f"Expected timezone reference (New York) to confirm state-mapping: {final_message}"
            )
        finally:
            if tool:
                tool.close()
