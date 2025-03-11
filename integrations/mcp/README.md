# MCP Haystack Integration

This integration adds support for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) to Haystack. MCP is an open protocol that standardizes how applications provide context to LLMs, similar to how USB-C provides a standardized way to connect devices.

## Installation

```bash
pip install mcp-haystack
```

## Usage

```python
from haystack_integrations.components.tools.mcp import MCPTool, SSEServerInfo

# Create an MCP tool that connects to an HTTP server
server_info = SSEServerInfo(base_url="http://localhost:8000")
tool = MCPTool(name="my_tool", server_info=server_info)

# Use the tool
result = tool.invoke(param1="value1", param2="value2")
```

# Examples

Check out the examples directory to see practical demonstrations of how to integrate the MCPTool into Haystack's tooling architecture. These examples will help you get started quickly with your own agentic applications.

## License

Apache 2.0 