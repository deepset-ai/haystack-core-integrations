# MCP Haystack Integration

This integration adds support for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) to Haystack. MCP is an open protocol that standardizes how applications provide context to LLMs, similar to how USB-C provides a standardized way to connect devices.

## Installation

```bash
pip install mcp-haystack
```

## Usage

### Using Individual Tools (MCPTool)

```python
from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo

# Create an MCP tool that connects to an HTTP server
server_info = SSEServerInfo(url="http://localhost:8000/sse")
tool = MCPTool(name="my_tool", server_info=server_info)

# Use the tool
result = tool.invoke(param1="value1", param2="value2")
```

### Using Tool Collections (MCPToolset)

```python
from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo

# Create a toolset that automatically discovers all available tools
server_info = StdioServerInfo(command="uvx", args=["mcp-server-time"])
toolset = MCPToolset(server_info)

# Use tools from the toolset
for tool in toolset:
    print(f"Available tool: {tool.name} - {tool.description}")

# Or filter to specific tools
filtered_toolset = MCPToolset(server_info, tool_names=["get_current_time"])
```

# Examples

Check out the examples directory to see practical demonstrations of how to integrate the MCPTool into Haystack's tooling architecture. These examples will help you get started quickly with your own agentic applications.

## What is uvx?

In some examples below, we use the `StdioServerInfo` class which relies on `uvx` behind the scenes. `uvx` is a convenient command from the uv package that runs Python tools in temporary, isolated environments. You only need to install `uvx` once, and it will automatically fetch any required packages on first use without needing manual installation.

## Example 1: MCP Server with SSE Transport

This example demonstrates how to create a simple calculator server using MCP and connect to it using MCPTool with SSE transport.

### Step 1: Run the MCP Server

First, run the server that exposes calculator functionality (addition and subtraction) via MCP:

```bash
python examples/mcp_server.py --transport sse
```

This creates a FastMCP server with two tools:
- `add(a, b)`: Adds two numbers
- `subtract(a, b)`: Subtracts two numbers

The server runs on http://localhost:8000 by default.

### Step 2: Connect with Individual Tools (MCPTool)

In a separate terminal, run the client that connects to the calculator server using individual MCPTool instances:

```bash
python examples/mcp_client.py --transport sse
```

The client creates MCPTool instances that connect to the server, inspect the tool specifications, and invoke the calculator functions remotely.

### Step 3: Connect with Tool Collections (MCPToolset)

Alternatively, use MCPToolset to automatically discover and use all available tools:

```bash
python examples/mcp_sse_toolset.py
```

This demonstrates how MCPToolset can automatically discover all tools from the server and create a collection of tools for easy access.

## Example 2: MCP with StdIO Transport

This example shows how to use MCP tools with stdio transport to execute a local program directly.

### Using Individual Tools (MCPTool)

```bash
python examples/mcp_stdio_client.py
```

The example creates an MCPTool that uses stdio transport with `StdioServerInfo`, which automatically uses `uvx` behind the scenes to run the `mcp-server-time` tool without requiring manual installation. It queries the current time in different timezones (New York and Los Angeles) by invoking the tool with different parameters.

### Using Tool Collections (MCPToolset)

```bash
python examples/mcp_stdio_toolset.py
```

This example demonstrates how MCPToolset can automatically discover all available tools from a stdio-based MCP server and create a collection for easy access.

Both approaches demonstrate how MCP tools can work with local programs without running a separate server process, using standard input/output for communication.

## Example 3: MCP Tools in Haystack Pipelines

These examples showcase how to integrate MCP tools into Haystack pipelines along with LLMs.

### Using Individual Tools (MCPTool)

```bash
python examples/time_pipeline.py
```

This example creates a pipeline that:
1. Takes a user query about the current time in a city
2. Uses an LLM (GPT-4o-mini) to interpret the query and decide which tool to use
3. Invokes the time tool with the appropriate parameters (using `uvx` behind the scenes)
4. Sends the tool's response back to the LLM to generate a final answer

### Using Tool Collections (MCPToolset)

```bash
python examples/time_pipeline_toolset.py
```

This example demonstrates the same functionality but uses MCPToolset instead of individual MCPTool instances. The toolset automatically discovers all available tools from the MCP server and makes them available to the LLM.

Both examples demonstrate how MCP tools can be seamlessly integrated into Haystack's agentic architecture, allowing LLMs to use external tools via the Model Context Protocol.

## Example 4: Advanced MCPToolset Features

### Tool Filtering

```bash
python examples/mcp_filtered_tools.py
```

This example demonstrates how to use MCPToolset with tool filtering, allowing you to selectively include only specific tools from an MCP server. This is useful when you want to limit which tools are available to your application or LLM.

## License

Apache 2.0 
