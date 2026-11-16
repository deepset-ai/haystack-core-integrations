# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for the mcp SDK v1 and v2"""

import inspect
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from typing import Any, Literal

import anyio
from pydantic import BaseModel

from mcp import types
from mcp.client.auth import OAuthClientProvider
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider

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
    from mcp.shared.auth import AuthorizationCodeResult  # type: ignore[attr-defined,unused-ignore]
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


def create_client_credentials_provider(
    *,
    server_url: str,
    storage: Any,
    client_id: str,
    client_secret: str,
    token_endpoint_auth_method: Literal["client_secret_basic", "client_secret_post"],
    scope: str | None,
    issuer: str | None,
) -> Any:
    """
    Create the SDK's OAuth Client Credentials provider irrespective of the installed SDK.

    The scope argument is named `scopes` in v1 and `scope` in v2, so it is passed positionally. `issuer` is only
    accepted from mcp 1.30.0 (v1) and 2.2.0 (v2).

    :param server_url: URL of the MCP server.
    :param storage: Implementation of the `mcp.client.auth.TokenStorage` protocol.
    :param client_id: OAuth client identifier.
    :param client_secret: OAuth client secret.
    :param token_endpoint_auth_method: `"client_secret_basic"` or `"client_secret_post"`.
    :param scope: Space-separated scopes to request.
    :param issuer: Expected issuer of the authorization server.
    :returns: A `ClientCredentialsOAuthProvider`, usable as `auth` of the SDK's HTTP client.
    :raises ValueError: If `issuer` is set but the installed SDK doesn't support it.
    """
    kwargs: dict[str, Any] = {}
    if issuer is not None:
        if "issuer" not in inspect.signature(ClientCredentialsOAuthProvider.__init__).parameters:
            message = (
                f"OAuthConfig.issuer needs mcp>=1.30.0 (v1) or mcp>=2.2.0 (v2), but mcp {version('mcp')} is installed."
            )
            raise ValueError(message)
        kwargs["issuer"] = issuer

    return ClientCredentialsOAuthProvider(
        server_url, storage, client_id, client_secret, token_endpoint_auth_method, scope, **kwargs
    )


def create_authorization_code_provider(
    *,
    server_url: str,
    client_metadata: Any,
    storage: Any,
    redirect_handler: Callable[[str], Awaitable[None]],
    callback_handler: Callable[[], Awaitable[Any]],
) -> Any:
    """
    Create the SDK's OAuth provider for the Authorization Code grant.

    The provider discovers the authorization server, registers the client dynamically when no client information is
    stored, and runs the Authorization Code flow with PKCE when the MCP server asks for a token.

    :param server_url: URL of the MCP server.
    :param client_metadata: `OAuthClientMetadata` used for dynamic client registration.
    :param storage: Implementation of the `mcp.client.auth.TokenStorage` protocol.
    :param redirect_handler: Coroutine function called with the authorization URL the user has to visit.
    :param callback_handler: Coroutine function returning the authorization redirect parameters, as built by
        `authorization_code_result`.
    :returns: An `OAuthClientProvider`, usable as `auth` of the SDK's HTTP client.
    """
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


def authorization_code_result(*, code: str, state: str | None, iss: str | None) -> Any:
    """
    Build the value an `OAuthClientProvider` callback handler returns, irrespective of the installed SDK.

    v1 expects a `(code, state)` tuple. v2 expects an `AuthorizationCodeResult`, which also carries the RFC 9207
    `iss` parameter so the SDK can check which authorization server sent the redirect.

    :param code: The authorization code.
    :param state: The `state` parameter of the redirect.
    :param iss: The `iss` parameter of the redirect, if the authorization server sent one.
    :returns: The callback handler result for the installed SDK.
    """
    if MCP_V2:
        return AuthorizationCodeResult(code=code, state=state, iss=iss)
    return (code, state)
