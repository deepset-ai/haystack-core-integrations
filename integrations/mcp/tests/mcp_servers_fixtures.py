from mcp import types

from haystack_integrations.tools.mcp.compatibility_layer import MCP_V2

if MCP_V2:
    from mcp.server import MCPServer

    class FixtureServer(MCPServer):
        """v2 renamed the `_mcp_server` attribute to `_lowlevel_server`; keep the v1 name working too."""

        @property
        def _mcp_server(self):
            return self._lowlevel_server

else:
    from mcp.server.fastmcp import FastMCP as FixtureServer

################################################
# Calculator MCP Server
################################################

calculator_mcp = FixtureServer("Calculator")


@calculator_mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@calculator_mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract integer b from integer a."""
    return a - b


@calculator_mcp.tool()
def divide_by_zero(a: int) -> float:
    """Intentionally cause a division by zero error."""
    return a / 0


################################################
# State IO Calculator MCP Server (returns dicts for state propagation)
################################################

state_calculator_mcp = FixtureServer("StateCalculator")


@state_calculator_mcp.tool()
def state_add(a: int, b: int) -> dict:
    """Add two integers."""
    return {"result": a + b}


@state_calculator_mcp.tool()
def state_subtract(a: int, b: int) -> dict:
    """Subtract integer b from integer a."""
    return {"result": a - b}


################################################
# Echo MCP Server
################################################

echo_mcp = FixtureServer("Echo")


@echo_mcp.tool()
def echo(text: str) -> str:
    """Echo the input text."""
    return text


################################################
# Image MCP Server
################################################

image_mcp = FixtureServer("Image")


@image_mcp.tool()
def image_tool() -> list[types.ImageContent]:
    """Return image content without any text blocks."""
    return [types.ImageContent(type="image", data="ZmFrZQ==", mimeType="image/png")]
