# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for the mcp SDK v1 and v2"""

from importlib.metadata import version
from typing import Any

import anyio
from pydantic import BaseModel

from mcp import types

MCP_V2 = int(version("mcp").split(".")[0]) >= 2  # noqa: PLR2004

if MCP_V2:
    import httpx2 as http_lib
else:
    import httpx as http_lib  # type: ignore[no-redef]

_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    anyio.ClosedResourceError,
    ConnectionError,
    OSError,
    http_lib.TransportError,
)

if MCP_V2:
    from mcp.shared.exceptions import MCPError as _SDKError  # type: ignore[attr-defined,no-redef]
else:
    from mcp.shared.exceptions import McpError as _SDKError  # type: ignore[attr-defined,no-redef]


def is_reconnectable(error: BaseException) -> bool:
    """
    Whether a failed tool call looks like a dropped connection rather than a server-side error.

    The v2 SDK reports a dropped stream as an `MCPError` with the `CONNECTION_CLOSED` code, where v1 surfaced the
    underlying transport exception.

    :param error: The exception raised by the tool call.
    :returns: True if reconnecting and retrying is worth attempting.
    """
    if isinstance(error, _TRANSPORT_ERRORS):
        return True
    if not isinstance(error, _SDKError):
        return False
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(getattr(error, "error", None), "code", None)
    return code == types.CONNECTION_CLOSED


def mcp_field_value(model: BaseModel, name: str) -> Any:
    """
    Read an MCP protocol model field irrespective of the installed SDK.

    :param model: The MCP protocol model to read from.
    :param name: The field name in the v2 (snake_case) spelling.
    :returns: The field value, or None if the installed SDK does not define the field.
    """
    model_fields = getattr(type(model), "model_fields", None)
    if model_fields is None:
        message = f"Expected an MCP protocol model, got {type(model).__name__}"
        raise TypeError(message)

    if name in model_fields:
        field_name = name
    else:
        first, *rest = name.split("_")
        field_name = first + "".join(word.title() for word in rest)

    return getattr(model, field_name, None)
