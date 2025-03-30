# MCP Haystack Integration

This integration adds support for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) to Haystack. MCP is an open protocol that standardizes how applications provide context to LLMs, similar to how USB-C provides a standardized way to connect devices.

## Installation

```bash
pip install mcp-haystack
```

## Usage

```python
from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo

# Create an MCP tool that connects to an HTTP server
server_info = SSEServerInfo(base_url="http://localhost:8000")
tool = MCPTool(name="my_tool", server_info=server_info)

# Use the tool
result = tool.invoke(param1="value1", param2="value2")
```

# Examples

Check out the examples directory to see practical demonstrations of how to integrate the MCPTool into Haystack's tooling architecture. These examples will help you get started quickly with your own agentic applications.

## What is uvx?

In some examples below, we use the `StdioServerInfo` class which relies on `uvx` behind the scenes. `uvx` is a convenient command from the uv package that runs Python tools in temporary, isolated environments. You only need to install `uvx` once, and it will automatically fetch any required packages on first use without needing manual installation.

## Example 1: MCP Server with SSE Transport

This example demonstrates how to create a simple calculator server using MCP and connect to it using the MCPTool with SSE transport.

### Step 1: Run the MCP Server

First, run the server that exposes calculator functionality (addition and subtraction) via MCP:

```bash
python examples/mcp_sse_server.py
```

This creates a FastMCP server with two tools:
- `add(a, b)`: Adds two numbers
- `subtract(a, b)`: Subtracts two numbers

The server runs on http://localhost:8000 by default.

### Step 2: Connect with the MCP Client

In a separate terminal, run the client that connects to the calculator server:

```bash
python examples/mcp_sse_client.py
```

The client creates MCPTool instances that connect to the server, inspect the tool specifications, and invoke the calculator functions remotely.

## Example 2: MCP with StdIO Transport

This example shows how to use MCPTool with stdio transport to execute a local program directly:

```bash
python examples/mcp_stdio_client.py
```

The example creates an MCPTool that uses stdio transport with `StdioServerInfo`, which automatically uses `uvx` behind the scenes to run the `mcp-server-time` tool without requiring manual installation. It queries the current time in different timezones (New York and Los Angeles) by invoking the tool with different parameters.

This demonstrates how MCPTool can work with local programs without running a separate server process, using standard input/output for communication.

## Example 3: MCPTool in a Haystack Pipeline

This example showcases how to integrate MCPTool into a Haystack pipeline along with an LLM:

```bash
python examples/time_pipeline.py
```

This example creates a pipeline that:
1. Takes a user query about the current time in a city
2. Uses an LLM (GPT-4o-mini) to interpret the query and decide which tool to use
3. Invokes the time tool with the appropriate parameters (using `uvx` behind the scenes)
4. Sends the tool's response back to the LLM to generate a final answer

This demonstrates how MCPTool can be seamlessly integrated into Haystack's agentic architecture, allowing LLMs to use external tools via the Model Context Protocol.

## License

Apache 2.0 
