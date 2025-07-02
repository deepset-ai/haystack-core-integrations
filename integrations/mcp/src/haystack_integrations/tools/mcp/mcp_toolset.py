# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx
from exceptiongroup import ExceptionGroup
from haystack import logging
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Tool, Toolset

from .mcp_tool import (
    AsyncExecutor,
    MCPClient,
    MCPConnectionError,
    MCPServerInfo,
    MCPToolNotFoundError,
    SSEServerInfo,
    StdioServerInfo,
    StreamableHttpServerInfo,
    _MCPClientSessionManager,
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
    # Prerequisites:
    # 1. pip install uvx mcp-server-time  # Install required MCP server and tools
    # 2. export OPENAI_API_KEY="your-api-key"  # Set up your OpenAI API key

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
        server_info=SSEServerInfo(url="http://some-remote-server.com:8000/sse"),
        tool_names=["add", "subtract"]  # Only include specific tools
    )

    # Use the toolset as shown in the pipeline example above
    ```
    """

    def __init__(
        self,
        server_info: MCPServerInfo,
        tool_names: list[str] | None = None,
        connection_timeout: float = 30.0,
        invocation_timeout: float = 30.0,
    ):
        """
        Initialize the MCP toolset.

        :param server_info: Connection information for the MCP server
        :param tool_names: Optional list of tool names to include. If provided, only tools with
                          matching names will be added to the toolset.
        :param connection_timeout: Timeout in seconds for server connection
        :param invocation_timeout: Default timeout in seconds for tool invocations
        :raises MCPToolNotFoundError: If any of the specified tool names are not found on the server
        """
        # Store configuration
        self.server_info = server_info
        self.tool_names = tool_names
        self.connection_timeout = connection_timeout
        self.invocation_timeout = invocation_timeout

        # Connect and load tools
        try:
            # Create the client and spin up a worker so open/close happen in the
            # same coroutine, avoiding AnyIO cancel-scope issues.
            client = self.server_info.create_client()
            self._worker = _MCPClientSessionManager(client, timeout=self.connection_timeout)

            tools = self._worker.tools()

            # If tool_names is provided, validate that all requested tools exist
            if self.tool_names:
                available_tools = {tool.name for tool in tools}
                missing_tools = set(self.tool_names) - available_tools
                if missing_tools:
                    message = (
                        f"The following tools were not found: {', '.join(missing_tools)}. "
                        f"Available tools: {', '.join(available_tools)}"
                    )
                    raise MCPToolNotFoundError(
                        message=message, tool_name=next(iter(missing_tools)), available_tools=list(available_tools)
                    )

            # This is a factory that creates the invocation function for the Tool
            def create_invoke_tool(
                owner_toolset: "MCPToolset",
                mcp_client: MCPClient,
                tool_name: str,
                tool_timeout: float,
            ) -> Callable[..., Any]:
                """Return a closure that keeps a strong reference to *owner_toolset* alive."""

                def invoke_tool(**kwargs) -> Any:
                    _ = owner_toolset  # strong reference so GC can't collect the toolset too early
                    return AsyncExecutor.get_instance().run(
                        mcp_client.call_tool(tool_name, kwargs), timeout=tool_timeout
                    )

                return invoke_tool

            # Create Tool instances not MCPTool for each available tool
            haystack_tools = []
            for tool_info in tools:
                # Skip tools not in the tool_names list if tool_names is provided
                if self.tool_names is not None and tool_info.name not in self.tool_names:
                    logger.debug(
                        "Skipping tool '{name}' as it's not in the requested tool_names list", name=tool_info.name
                    )
                    continue

                # Use the helper function to create the invoke_tool function
                tool = Tool(
                    name=tool_info.name,
                    description=tool_info.description,
                    parameters=tool_info.inputSchema,
                    function=create_invoke_tool(self, client, tool_info.name, self.invocation_timeout),
                )
                haystack_tools.append(tool)

            # Initialize parent class with complete tools list
            super().__init__(tools=haystack_tools)

        except Exception as e:
            # We need to close because we could connect properly, retrieve tools yet
            # fail because of an MCPToolNotFoundError
            self.close()

            # Create informative error message for SSE connection errors
            # Common error handling for HTTP-based transports
            if isinstance(self.server_info, (SSEServerInfo | StreamableHttpServerInfo)):
                # Determine transport type for messages
                transport_name = "SSE" if isinstance(self.server_info, SSEServerInfo) else "streamable HTTP"
                server_url = self.server_info.url

                base_message = f"Failed to connect to MCP server via {transport_name}"
                checks = [
                    f"1. The server URL is correct (attempted: {server_url})",
                    "2. The server is running and accessible",
                    "3. Authentication token is correct (if required)",
                ]

                # Add specific connection error details for network issues
                has_connect_error = isinstance(e, httpx.ConnectError) or (
                    isinstance(e, ExceptionGroup) and any(isinstance(exc, httpx.ConnectError) for exc in e.exceptions)
                )

                if has_connect_error:
                    # Use urlparse to reliably get scheme, hostname, and port
                    parsed_url = urlparse(server_url)
                    port_str = ""
                    if parsed_url.port:
                        port_str = str(parsed_url.port)
                    elif parsed_url.scheme == "http":
                        port_str = "80 (default)"
                    elif parsed_url.scheme == "https":
                        port_str = "443 (default)"
                    else:
                        port_str = "unknown (scheme not http/https or missing)"

                    # Ensure hostname is handled correctly (it might be None)
                    hostname_str = str(parsed_url.hostname) if parsed_url.hostname else "<unknown>"

                    # Replace generic accessible message with specific network details
                    checks[1] = f"2. The address '{hostname_str}' and port '{port_str}' are correct"
                    checks.append("4. There are no firewall or network connectivity issues")

                message = f"{base_message}. Please check if:\n" + "\n".join(checks)

            # and for stdio connection errors
            elif isinstance(self.server_info, StdioServerInfo):  # stdio connection
                base_message = "Failed to start MCP server process"
                stdio_info = self.server_info
                args_str = " ".join(stdio_info.args) if stdio_info.args else ""
                cmd = f"{stdio_info.command}{' ' + args_str if args_str else ''}"
                checks = [f"1. The command and arguments are correct (attempted: {cmd})"]
                message = f"{base_message}. Please check if:\n" + "\n".join(checks)
            else:
                message = f"Unsupported server info type: {type(self.server_info)}"

            raise MCPConnectionError(message=message, server_info=self.server_info, operation="initialize") from e

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
            connection_timeout=inner_data.get("connection_timeout", 30.0),
            invocation_timeout=inner_data.get("invocation_timeout", 30.0),
        )

    def close(self):
        """Close the underlying MCP client safely."""
        if hasattr(self, "_worker") and self._worker:
            try:
                self._worker.stop()
            except Exception as e:
                logger.debug(f"TOOLSET: error during worker stop: {e!s}")

    def __del__(self):
        self.close()
