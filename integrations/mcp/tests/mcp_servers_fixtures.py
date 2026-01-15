from mcp.server.fastmcp import FastMCP

################################################
# Calculator MCP Server
################################################

calculator_mcp = FastMCP("Calculator")


@calculator_mcp.tool()
def add(a: int, b: int) -> dict:
    """Add two integers."""
    return {"result": a + b}


@calculator_mcp.tool()
def subtract(a: int, b: int) -> dict:
    """Subtract integer b from integer a."""
    return {"result": a - b}


@calculator_mcp.tool()
def divide_by_zero(a: int) -> float:
    """Intentionally cause a division by zero error."""
    return a / 0


################################################
# Echo MCP Server
################################################

echo_mcp = FastMCP("Echo")


@echo_mcp.tool()
def echo(text: str) -> str:
    """Echo the input text."""
    return text
