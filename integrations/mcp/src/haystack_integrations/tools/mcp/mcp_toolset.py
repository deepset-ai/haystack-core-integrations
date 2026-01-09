# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from exceptiongroup import ExceptionGroup
from haystack import logging
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Tool, Toolset
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

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


def _serialize_state_config(config: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
    """
    Serialize a state configuration dictionary, converting any callable handlers to their string representation.

    Works for both outputs_to_state (tool_name -> {state_key -> {source, handler}})
    and outputs_to_string (tool_name -> {source, handler}).

    :param config: The state configuration dictionary to serialize
    :returns: The serialized configuration dictionary, or None if empty
    """
    if not config:
        return None

    serialized = {}
    for tool_name, tool_config in config.items():
        if not tool_config:
            continue

        # Check if this is outputs_to_string format (flat with optional source/handler)
        # or outputs_to_state format (nested with state keys)
        if "source" in tool_config or "handler" in tool_config:
            # outputs_to_string format: {source?, handler?}
            serialized_tool_config = tool_config.copy()
            if "handler" in tool_config and callable(tool_config["handler"]):
                serialized_tool_config["handler"] = serialize_callable(tool_config["handler"])
            serialized[tool_name] = serialized_tool_config
        else:
            # outputs_to_state format: {state_key -> {source?, handler?}}
            serialized_tool_config = {}
            for state_key, state_config in tool_config.items():
                serialized_state_config = state_config.copy()
                if "handler" in state_config and callable(state_config["handler"]):
                    serialized_state_config["handler"] = serialize_callable(state_config["handler"])
                serialized_tool_config[state_key] = serialized_state_config
            serialized[tool_name] = serialized_tool_config

    return serialized if serialized else None


def _deserialize_state_config(config: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """
    Deserialize a state configuration dictionary, converting any serialized handlers back to callables.

    Works for both outputs_to_state (tool_name -> {state_key -> {source, handler}})
    and outputs_to_string (tool_name -> {source, handler}).

    :param config: The state configuration dictionary to deserialize
    :returns: The deserialized configuration dictionary
    """
    if not config:
        return {}

    deserialized = {}
    for tool_name, tool_config in config.items():
        if not tool_config:
            continue

        # Check if this is outputs_to_string format (flat with optional source/handler)
        # or outputs_to_state format (nested with state keys)
        if "source" in tool_config or "handler" in tool_config:
            # outputs_to_string format: {source?, handler?}
            deserialized_tool_config = tool_config.copy()
            if "handler" in tool_config and isinstance(tool_config["handler"], str):
                deserialized_tool_config["handler"] = deserialize_callable(tool_config["handler"])
            deserialized[tool_name] = deserialized_tool_config
        else:
            # outputs_to_state format: {state_key -> {source?, handler?}}
            deserialized_tool_config = {}
            for state_key, state_config in tool_config.items():
                deserialized_state_config = state_config.copy()
                if "handler" in state_config and isinstance(state_config["handler"], str):
                    deserialized_state_config["handler"] = deserialize_callable(state_config["handler"])
                deserialized_tool_config[state_key] = deserialized_state_config
            deserialized[tool_name] = deserialized_tool_config

    return deserialized


class MCPToolset(Toolset):
    """
    A Toolset that connects to an MCP (Model Context Protocol) server and provides
    access to its tools.

    MCPToolset dynamically discovers and loads all tools from any MCP-compliant server,
    supporting both network-based streaming connections (Streamable HTTP, SSE) and local
    process-based stdio connections.
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

    You can also use the toolset via Streamable HTTP to talk to remote servers:
    ```python
    from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

    # Create the toolset with streamable HTTP connection
    toolset = MCPToolset(
        server_info=StreamableHttpServerInfo(url="http://localhost:8000/mcp"),
        tool_names=["multiply"]  # Optional: only include specific tools
    )
    # Use the toolset as shown in the pipeline example above
    ```

    Example with state configuration for Agent integration:
    ```python
    from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo

    # Create the toolset with per-tool state configuration
    # This enables tools to read from and write to the Agent's State
    toolset = MCPToolset(
        server_info=StdioServerInfo(command="uvx", args=["mcp-server-git"]),
        tool_names=["git_status", "git_diff", "git_log"],

        # Map state keys to tool parameters for each tool
        inputs_from_state={
            "git_status": {"repository": "repo_path"},
            "git_diff": {"repository": "repo_path"},
            "git_log": {"repository": "repo_path"},
        },
        # Map tool outputs to state keys
        outputs_to_state={
            "git_status": {"status_result": {"source": "status"}},
            "git_diff": {"diff_result": {}},
        },
    )
    ```

    Example using SSE (deprecated):
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
        eager_connect: bool = False,
        inputs_from_state: dict[str, dict[str, str]] | None = None,
        outputs_to_state: dict[str, dict[str, dict[str, Any]]] | None = None,
        outputs_to_string: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Initialize the MCP toolset.

        :param server_info: Connection information for the MCP server
        :param tool_names: Optional list of tool names to include. If provided, only tools with
                          matching names will be added to the toolset.
        :param connection_timeout: Timeout in seconds for server connection
        :param invocation_timeout: Default timeout in seconds for tool invocations
        :param eager_connect: If True, connect to server and load tools during initialization.
                             If False (default), defer connection to warm_up.
        :param inputs_from_state: Optional dictionary mapping tool names to their inputs_from_state config.
                                 Each config maps state keys to tool parameter names.
                                 Example: `{"git_status": {"repository": "repo_path"}}`
        :param outputs_to_state: Optional dictionary mapping tool names to their outputs_to_state config.
                                Each config defines how tool outputs map to state keys with optional handlers.
                                Example: `{"git_status": {"status_result": {"source": "status"}}}`
        :param outputs_to_string: Optional dictionary mapping tool names to their outputs_to_string config.
                                 Each config defines how tool outputs are converted to strings.
                                 Example: `{"git_diff": {"source": "diff", "handler": format_diff}}`
        :raises MCPToolNotFoundError: If any of the specified tool names are not found on the server
        """
        # Store configuration
        self.server_info = server_info
        self.tool_names = tool_names
        self.connection_timeout = connection_timeout
        self.invocation_timeout = invocation_timeout
        self.eager_connect = eager_connect
        self.inputs_from_state = inputs_from_state or {}
        self.outputs_to_state = outputs_to_state or {}
        self.outputs_to_string = outputs_to_string or {}
        self._warmup_called = False

        if not eager_connect:
            # Do not connect during validation; expose a toolset with one fake tool to pass validation
            placeholder_tool = Tool(
                name=f"mcp_not_connected_placeholder_{id(self)}",
                description="Placeholder tool initialised when eager_connect is turned off",
                parameters={"type": "object", "properties": {}, "additionalProperties": True},
                function=lambda: None,
            )
            super().__init__(tools=[placeholder_tool])
        else:
            tools = self._connect_and_load_tools()
            super().__init__(tools=tools)
            self._warmup_called = True

    def warm_up(self) -> None:
        """Connect and load tools when eager_connect is turned off.

        This method is automatically called by ``ToolInvoker.warm_up()`` and ``Pipeline.warm_up()``.
        You can also call it directly before using the toolset to ensure all tool schemas
        are available without performing a real invocation.
        """
        if self._warmup_called:
            return

        # connect and load tools never adds duplicate tools, set the tools attribute directly
        self.tools = self._connect_and_load_tools()
        self._warmup_called = True

    def _connect_and_load_tools(self) -> list[Tool]:
        """Connect and load tools."""
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

                def invoke_tool(**kwargs: Any) -> Any:
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
                    description=tool_info.description or "",
                    parameters=tool_info.inputSchema,
                    function=create_invoke_tool(self, client, tool_info.name, self.invocation_timeout),
                    inputs_from_state=self.inputs_from_state.get(tool_info.name),
                    outputs_to_state=self.outputs_to_state.get(tool_info.name),
                    outputs_to_string=self.outputs_to_string.get(tool_info.name),
                )
                haystack_tools.append(tool)

            # Validate state configs reference known tools
            self._validate_state_configs({tool.name for tool in haystack_tools})

            return haystack_tools
        except Exception as e:
            # We need to close because we could connect properly, retrieve tools yet
            # fail because of an MCPToolNotFoundError
            self.close()

            if isinstance(e, MCPToolNotFoundError):
                raise  # re-raise MCPToolNotFoundError as is to show original message

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

    def _validate_state_configs(self, available_tool_names: set[str]) -> None:
        """
        Validate that state configuration tool names exist in the toolset.

        Logs a warning for any tool names in the state configs that don't match
        available tools in the toolset.

        :param available_tool_names: Set of tool names that are available in the toolset
        """
        configs: list[tuple[str, dict[str, Any]]] = [
            ("inputs_from_state", self.inputs_from_state),
            ("outputs_to_state", self.outputs_to_state),
            ("outputs_to_string", self.outputs_to_string),
        ]
        for config_name, config in configs:
            if config:
                unknown_tools = set(config.keys()) - available_tool_names
                if unknown_tools:
                    logger.warning(
                        "{config_name} references unknown tools: {unknown_tools}. Available tools: {available_tools}",
                        config_name=config_name,
                        unknown_tools=unknown_tools,
                        available_tools=available_tool_names,
                    )

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
                "eager_connect": self.eager_connect,
                "inputs_from_state": self.inputs_from_state if self.inputs_from_state else None,
                "outputs_to_state": _serialize_state_config(self.outputs_to_state),
                "outputs_to_string": _serialize_state_config(self.outputs_to_string),
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
        server_info = cast(MCPServerInfo, server_info_class).from_dict(server_info_dict)

        # Deserialize state configuration parameters
        inputs_from_state = inner_data.get("inputs_from_state") or {}
        outputs_to_state = _deserialize_state_config(inner_data.get("outputs_to_state"))
        outputs_to_string = _deserialize_state_config(inner_data.get("outputs_to_string"))

        # Create a new MCPToolset instance
        return cls(
            server_info=server_info,
            tool_names=inner_data.get("tool_names"),
            connection_timeout=inner_data.get("connection_timeout", 30.0),
            invocation_timeout=inner_data.get("invocation_timeout", 30.0),
            eager_connect=inner_data.get("eager_connect", False),
            inputs_from_state=inputs_from_state if inputs_from_state else None,
            outputs_to_state=outputs_to_state if outputs_to_state else None,
            outputs_to_string=outputs_to_string if outputs_to_string else None,
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
