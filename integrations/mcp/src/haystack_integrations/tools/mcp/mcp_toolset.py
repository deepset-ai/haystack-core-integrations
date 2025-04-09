import asyncio
import logging
from typing import Any

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Toolset

# Import MCP-related classes
from .mcp_tool import (
    MCPConnectionError,
    MCPServerInfo,
    MCPTool,
)

logger = logging.getLogger(__name__)


class MCPToolset(Toolset):
    """
    A Toolset that connects to an MCP (Model Context Protocol) server and provides
    access to its tools.

    MCPToolset dynamically discovers and loads all tools from any MCP-compliant server,
    supporting both network-based SSE connections and local process-based stdio connections.
    This dual connectivity allows for integrating with both remote and local MCP servers.

    Example using MCPToolset in a Haystack Pipeline:
    ```python
    import os
    from haystack import Pipeline
    from haystack.components.converters import OutputAdapter
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.tools import ToolInvoker
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo

    # Create server info for the time service (can also use SSEServerInfo for remote servers)
    server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])

    # Create the toolset - this will automatically discover all available tools
    # You can optionally specify which tools to include
    mcp_toolset = MCPToolset(
        server_info=server_info,
        tool_names=["get_current_time"]  # Only include the get_current_time tool
    )

    # Create a pipeline with the toolset
    pipeline = Pipeline()
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=mcp_toolset))
    pipeline.add_component("tool_invoker", ToolInvoker(tools=mcp_toolset))
    pipeline.add_component(
        "adapter",
        OutputAdapter(
            template="{{ initial_msg + initial_tool_messages + tool_messages }}",
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )
    pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))
    pipeline.connect("llm.replies", "tool_invoker.messages")
    pipeline.connect("llm.replies", "adapter.initial_tool_messages")
    pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
    pipeline.connect("adapter.output", "response_llm.messages")

    # Run the pipeline with a user question
    user_input = "What is the time in New York? Be brief."
    user_input_msg = ChatMessage.from_user(text=user_input)

    result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})
    print(result["response_llm"]["replies"][0].text)
    ```

    You can also use the toolset via MCP SSE to talk to remote servers:
    ```python
    from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo
    from haystack.components.tools import ToolInvoker

    # Create the toolset with an SSE connection
    sse_toolset = MCPToolset(
        server_info=SSEServerInfo(base_url="http://some-remote-server.com:8000"),
        tool_names=["add", "subtract"]  # Only include specific tools
    )

    # Use the toolset as shown in the pipeline example above
    ```
    """

    def __init__(
        self,
        server_info: MCPServerInfo,
        tool_names: list[str] | None = None,
        connection_timeout: int = 30,
        invocation_timeout: int = 30,
    ):
        """
        Initialize the MCP toolset.

        :param server_info: Connection information for the MCP server
        :param tool_names: Optional list of tool names to include. If provided, only tools with
                          matching names will be added to the toolset.
        :param connection_timeout: Timeout in seconds for server connection
        :param invocation_timeout: Default timeout in seconds for tool invocations
        """
        # Initialize with empty tools list first
        super().__init__(tools=[])

        # Store configuration
        self.server_info = server_info
        self.tool_names = tool_names
        self.connection_timeout = connection_timeout
        self.invocation_timeout = invocation_timeout

        # Connect and load tools
        try:
            # Create the appropriate client using the factory method
            client = self.server_info.create_client()

            # Connect and get available tools
            tools = self._run_sync(client.connect(), timeout=self.connection_timeout)

            # Create MCPTool instances for each available tool and add them
            for tool_info in tools:
                # Skip tools not in the tool_names list if tool_names is provided
                if self.tool_names is not None and tool_info.name not in self.tool_names:
                    logger.debug(f"Skipping tool '{tool_info.name}' as it's not in the requested tool_names list")
                    continue

                tool = MCPTool(
                    name=tool_info.name,
                    server_info=self.server_info,
                    description=tool_info.description,
                    connection_timeout=self.connection_timeout,
                    invocation_timeout=self.invocation_timeout,
                )
                # Handles duplicates and other validation
                self.add(tool)

        except Exception as e:
            message = f"Failed to initialize MCPToolset: {e}"
            raise MCPConnectionError(message=message, server_info=self.server_info, operation="initialize") from e

    def _run_sync(self, coro, timeout: float | None = None):
        """
        Run an async coroutine synchronously.

        :param coro: The coroutine to run
        :param timeout: Optional timeout in seconds
        :returns: The result of the coroutine
        :raises TimeoutError: If the operation times out
        """
        try:
            # Apply timeout if specified
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout=timeout)

            # Try to get a running loop first
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # If we can't get a running loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)

            # If the loop is already running, run our coroutine inline
            if loop.is_running():
                # Create a new loop for our coroutine to avoid conflicts
                # This should prevent "This event loop is already running" errors
                temp_loop = asyncio.new_event_loop()
                try:
                    return temp_loop.run_until_complete(coro)
                finally:
                    temp_loop.close()
            else:
                # Use the existing loop
                return loop.run_until_complete(coro)

        except asyncio.TimeoutError as e:
            message = f"Operation timed out after {timeout} seconds"
            raise TimeoutError(message) from e
        except Exception as e:
            # Re-raise any other exceptions
            raise e

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the MCPToolset to a dictionary.

        :returns: A dictionary representation of the MCPToolset
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "server_info": self.server_info.to_dict(),
                "tool_names": self.tool_names,
                "connection_timeout": self.connection_timeout,
                "invocation_timeout": self.invocation_timeout,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPToolset":
        """
        Deserialize an MCPToolset from a dictionary.

        :param data: Dictionary representation of the MCPToolset
        :returns: A new MCPToolset instance
        """
        inner_data = data["data"]

        # Reconstruct the server_info object
        server_info_dict = inner_data.get("server_info", {})
        server_info_class = import_class_by_name(server_info_dict["type"])
        server_info = server_info_class.from_dict(server_info_dict)

        # Create a new MCPToolset instance
        return cls(
            server_info=server_info,
            tool_names=inner_data.get("tool_names"),
            connection_timeout=inner_data.get("connection_timeout", 30),
            invocation_timeout=inner_data.get("invocation_timeout", 30),
        )
