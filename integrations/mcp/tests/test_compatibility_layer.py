from unittest.mock import MagicMock

import anyio
import pytest
from mcp import types

from haystack_integrations.tools.mcp.compatibility_layer import (
    MCP_V2,
    http_lib,
    is_reconnectable,
    mcp_field_value,
)

if MCP_V2:
    from mcp.shared.exceptions import MCPError

    def sdk_error(code):
        return MCPError(code=code, message="boom")

else:
    from mcp.shared.exceptions import McpError

    def sdk_error(code):
        return McpError(types.ErrorData(code=code, message="boom"))


class TestMCPFieldValue:
    @pytest.mark.parametrize(
        "model, name, expected",
        [
            (types.Tool(name="add", inputSchema={"type": "object"}), "input_schema", {"type": "object"}),
            (types.CallToolResult(content=[], isError=True), "is_error", True),
            (types.CallToolResult(content=[], isError=False), "is_error", False),
            (types.Tool(name="add", inputSchema={}), "name", "add"),
        ],
    )
    def test_reads_fields_by_whichever_spelling_the_sdk_uses(self, model, name, expected):
        assert mcp_field_value(model=model, name=name) == expected

    def test_returns_none_for_a_field_the_sdk_does_not_define(self):
        assert mcp_field_value(model=types.Tool(name="add", inputSchema={}), name="not_a_field") is None

    @pytest.mark.parametrize("not_a_model", [MagicMock(), {"is_error": True}, "is_error"])
    def test_rejects_objects_that_are_not_protocol_models(self, not_a_model):
        with pytest.raises(TypeError):
            mcp_field_value(model=not_a_model, name="is_error")


class TestIsReconnectable:
    @pytest.mark.parametrize(
        "error",
        [
            anyio.ClosedResourceError(),
            ConnectionError("connection lost"),
            OSError("connection lost"),
            http_lib.RemoteProtocolError("incomplete chunked read"),
            http_lib.ConnectError("refused"),
            sdk_error(types.CONNECTION_CLOSED),
        ],
    )
    def test_true_for_dropped_connections(self, error):
        assert is_reconnectable(error)

    @pytest.mark.parametrize(
        "error",
        [
            sdk_error(types.INVALID_PARAMS),
            sdk_error(types.METHOD_NOT_FOUND),
            ValueError("unrelated failure"),
            RuntimeError("unrelated failure"),
        ],
    )
    def test_false_for_server_side_and_unrelated_failures(self, error):
        assert not is_reconnectable(error)
