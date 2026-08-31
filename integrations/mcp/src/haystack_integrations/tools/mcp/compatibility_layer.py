# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility helpers for the mcp SDK v1 and v2.

The v2 SDK renamed the protocol model fields from camelCase to snake_case, replaced ``httpx``
with ``httpx2``, and reports dropped connections differently. This module isolates those
differences so the rest of the integration reads the same on both.
"""

from contextlib import AbstractAsyncContextManager
from importlib.metadata import version
from typing import Any

import anyio
from pydantic import BaseModel

from mcp import types
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

_V2_MAJOR = 2

# The v1 and v2 SDKs ship as the same distribution, so the generation is read off its version.
MCP_V2 = int(version("mcp").split(".")[0]) >= _V2_MAJOR

if MCP_V2:
    import httpx2 as http_lib
else:
    import httpx as http_lib  # type: ignore[no-redef]

# Transport failures that a reconnect can plausibly recover from. The HTTP errors are not OSError
# subclasses on either httpx generation, so they have to be named explicitly.
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

# Everything the retry loop catches; `is_reconnectable` decides which of them warrants a reconnect.
RECONNECTABLE_ERRORS: tuple[type[BaseException], ...] = (*_TRANSPORT_ERRORS, _SDKError)


def is_reconnectable(error: BaseException) -> bool:
    """
    Whether a failed tool call looks like a dropped connection rather than a server-side error.

    The v2 SDK reports a dropped stream as an ``MCPError`` with the ``CONNECTION_CLOSED`` code,
    where v1 surfaced the underlying transport exception.

    :param error: The exception raised by the tool call.
    :returns: True if reconnecting and retrying is worth attempting.
    """
    if isinstance(error, _TRANSPORT_ERRORS):
        return True
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(getattr(error, "error", None), "code", None)
    return code == types.CONNECTION_CLOSED


def _wire_name(name: str) -> str:
    """The camelCase (wire and SDK v1) spelling of a snake_case field name."""
    first, *rest = name.split("_")
    return first + "".join(word.title() for word in rest)


def mcp_field_value(model: BaseModel, name: str) -> Any:
    """
    Read an MCP protocol model field by whichever spelling the installed SDK uses.

    :param model: The MCP protocol model to read from.
    :param name: The field name in the v2 (snake_case) spelling.
    :returns: The field value, or None if the installed SDK does not define the field.
    """
    model_fields = getattr(type(model), "model_fields", None)
    if model_fields is None:
        message = f"Expected an MCP protocol model, got {type(model).__name__}"
        raise TypeError(message)
    return getattr(model, name if name in model_fields else _wire_name(name), None)


def open_streamable_http(
    url: str, headers: dict[str, str] | None, timeout: int
) -> AbstractAsyncContextManager[tuple[Any, ...]]:
    """
    Open a streamable HTTP transport.

    Both SDK generations take the HTTP settings through a pre-built client, so this only has to
    build that client from the right httpx generation.

    :param url: The MCP server endpoint URL.
    :param headers: Optional headers to send with every request.
    :param timeout: HTTP timeout in seconds.
    :returns: Async context manager yielding the transport streams.
    """
    http_client = create_mcp_http_client(headers=headers, timeout=http_lib.Timeout(timeout))  # type: ignore[arg-type]
    return streamable_http_client(url, http_client=http_client)
