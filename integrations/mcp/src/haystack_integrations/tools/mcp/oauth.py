# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
OAuth 2.0 support for MCP server connections.

Authentication is delegated to the OAuth providers shipped with the `mcp` SDK, which handle the MCP authorization
spec: authorization server discovery (RFC 9728 and RFC 8414), the `resource` parameter (RFC 8707), dynamic client
registration (RFC 7591), PKCE, and fetching a new token when the MCP server rejects the current one.

Two grants are supported:
- Client Credentials (RFC 6749 §4.4), for machine-to-machine connections.
- Authorization Code (RFC 6749 §4.1), for acting on behalf of a user who signs in through a browser. The redirect is
  captured by a temporary HTTP server on a loopback address (RFC 8252 §7.3).
"""

import html
import socket
import sys
import webbrowser
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit, urlunsplit

import anyio
from anyio.abc import SocketStream
from haystack import logging
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from pydantic import AnyUrl

from mcp.client.auth import OAuthFlowError, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from .compatibility_layer import (
    authorization_code_result,
    create_authorization_code_provider,
    create_client_credentials_provider,
    http_lib,
)

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
# Limits for reading the redirect request, so a stray connection can't hold the callback server
_MAX_REQUEST_HEAD_BYTES = 16 * 1024
_REQUEST_READ_TIMEOUT = 10.0


class InMemoryTokenStorage:
    """
    Token storage that keeps OAuth tokens in memory.

    Implements the `mcp.client.auth.TokenStorage` protocol. Tokens are lost when the process exits.
    """

    def __init__(self) -> None:
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        """Return the stored tokens, if any."""
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store the given tokens."""
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return the stored client information, if any."""
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store the given client information."""
        self._client_info = client_info


@dataclass
class OAuthConfig:
    """
    OAuth 2.0 configuration for authenticating to an MCP server.

    The authorization server and its endpoints are discovered from the MCP server, as described in the MCP
    authorization spec.

    With the Client Credentials grant (RFC 6749 §4.4), the default, the client authenticates with its own credentials.
    It is non-interactive and suited for machine-to-machine connections.

    With the Authorization Code grant (RFC 6749 §4.1), a user signs in through a browser and the client acts on their
    behalf. When the MCP server first asks for a token, the authorization URL is opened in the system browser and a
    temporary HTTP server on `redirect_uri` captures the redirect. When the redirect can't be captured (the port is
    taken, or no redirect arrives within `callback_timeout`) and the process runs in an interactive terminal, the user
    is asked to paste the URL they were redirected to. Without `client_id`, the client registers itself with the
    authorization server (dynamic client registration, RFC 7591). Access tokens are refreshed with the refresh token
    when possible, so the browser is only needed again when the refresh token is rejected.

    Signing in usually takes longer than the default `connection_timeout` of `MCPTool` and `MCPToolset`, so raise it
    (for example to `callback_timeout`) when using the Authorization Code grant.

    ### Usage example
    ```python
    from haystack.utils import Secret
    from haystack_integrations.tools.mcp import MCPToolset, OAuthConfig, StreamableHttpServerInfo

    # Client Credentials grant
    server_info = StreamableHttpServerInfo(
        url="https://my-mcp-server.com/mcp",
        oauth_config=OAuthConfig(
            client_id="my-client-id",
            client_secret=Secret.from_env_var("MCP_CLIENT_SECRET"),
            issuer="https://auth.example.com",
            scopes=["mcp:tools"],
        ),
    )
    toolset = MCPToolset(server_info=server_info)

    # Authorization Code grant: opens a browser to sign in on first use
    server_info = StreamableHttpServerInfo(
        url="https://my-mcp-server.com/mcp",
        oauth_config=OAuthConfig(grant_type="authorization_code"),
    )
    toolset = MCPToolset(server_info=server_info, connection_timeout=300)
    ```

    :param client_id: OAuth client identifier. Required for the Client Credentials grant. For the Authorization Code
        grant, the identifier of a client registered in advance with `redirect_uri` as redirect URI; when not set, the
        client is registered dynamically.
    :param client_secret: OAuth client secret. Required for the Client Credentials grant. For the Authorization Code
        grant, only set it for a confidential client registered in advance; leave it unset for a public client.
    :param issuer: Issuer identifier of the authorization server that issued the client credentials.
        When set, tokens are only requested from an authorization server whose metadata names exactly this issuer.
        Only used by the Client Credentials grant, where it is recommended and required by future versions of the
        `mcp` SDK. Needs `mcp>=1.30.0` (v1) or `mcp>=2.2.0` (v2).
    :param scopes: OAuth scopes to request. Only used by the Client Credentials grant. With the Authorization Code
        grant, the scopes follow the MCP authorization spec: those the MCP server asks for in its `WWW-Authenticate`
        challenge, or else the ones its metadata lists.
    :param token_endpoint_auth_method: How the client authenticates to the token endpoint:
        `"client_secret_basic"` (HTTP Basic, the default) or `"client_secret_post"` (credentials in the request body).
        Ignored for a public client of the Authorization Code grant, which sends no secret.
    :param grant_type: The OAuth grant type: `"client_credentials"` or `"authorization_code"`.
    :param token_storage: Where tokens, and client information obtained by dynamic client registration, are kept
        between connections. Any implementation of the `mcp.client.auth.TokenStorage` protocol works. Defaults to an
        `InMemoryTokenStorage`. Not serialized.
    :param redirect_uri: Authorization Code grant only. The `http` loopback URL (host `localhost`, `127.0.0.1` or
        `[::1]`) the authorization server redirects to, and the callback server listens on. Without a port, or with
        port 0, a free port is picked once per `OAuthConfig` instance. Defaults to `http://127.0.0.1/callback` with a
        free port. A client registered in advance must have this exact URL registered, so give it a fixed port.
    :param callback_timeout: Authorization Code grant only. Seconds to wait for the redirect after the authorization
        URL is opened, before falling back to asking for the redirected URL in the terminal.
    :param on_authorization_url: Authorization Code grant only. Coroutine function called with the authorization URL
        instead of opening it in the system browser, for example to show the URL in your own UI. The callback server
        still captures the redirect. Serialized by its import path, so it must be a module-level function.
    """

    client_id: str | None = None
    client_secret: Secret | None = None
    issuer: str | None = None
    scopes: list[str] | None = None
    token_endpoint_auth_method: Literal["client_secret_basic", "client_secret_post"] = "client_secret_basic"
    grant_type: Literal["client_credentials", "authorization_code"] = "client_credentials"
    token_storage: TokenStorage = field(default_factory=InMemoryTokenStorage, compare=False, repr=False)
    redirect_uri: str | None = None
    callback_timeout: float = 300.0
    on_authorization_url: Callable[[str], Awaitable[None]] | None = None
    # The redirect URI with its free port filled in, picked once so every authorization uses the registered URI
    _resolved_redirect_uri: str | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.grant_type not in ("client_credentials", "authorization_code"):
            message = f"Unsupported grant_type '{self.grant_type}'. Use 'client_credentials' or 'authorization_code'."
            raise ValueError(message)
        if self.token_endpoint_auth_method not in ("client_secret_basic", "client_secret_post"):
            message = (
                f"Unsupported token_endpoint_auth_method '{self.token_endpoint_auth_method}'. "
                "Use 'client_secret_basic' or 'client_secret_post'."
            )
            raise ValueError(message)

        if self.grant_type == "client_credentials":
            if not self.client_id or self.client_secret is None:
                message = "The 'client_credentials' grant needs both `client_id` and `client_secret`."
                raise ValueError(message)
            authorization_code_only = {
                "redirect_uri": self.redirect_uri is not None,
                "on_authorization_url": self.on_authorization_url is not None,
            }
            for name, is_set in authorization_code_only.items():
                if is_set:
                    message = f"`{name}` is only used by the 'authorization_code' grant."
                    raise ValueError(message)
            return

        if self.client_secret is not None and not self.client_id:
            message = "`client_secret` needs the `client_id` of a client registered in advance."
            raise ValueError(message)
        # The SDK's Authorization Code provider checks discovered metadata against the issuer the MCP server names,
        # and replaces the requested scope with the one the MCP server asks for
        for name in ("issuer", "scopes"):
            if getattr(self, name) is not None:
                message = f"`{name}` is only supported by the 'client_credentials' grant."
                raise ValueError(message)
        if self.callback_timeout <= 0:
            message = f"`callback_timeout` must be positive, got {self.callback_timeout}."
            raise ValueError(message)
        if self.redirect_uri is not None:
            _validate_loopback_redirect_uri(self.redirect_uri)

    def create_auth(self, server_url: str) -> Any:
        """
        Create the `httpx.Auth` that authenticates requests to the given MCP server.

        :param server_url: URL of the MCP server.
        :returns: An `mcp` SDK OAuth provider, to be passed as `auth` to the HTTP client.
        :raises ValueError: If the client secret can't be resolved.
        """
        client_secret = None
        if self.client_secret is not None:
            client_secret = self.client_secret.resolve_value()
            if not client_secret:
                message = "OAuthConfig.client_secret resolved to an empty value."
                raise ValueError(message)

        if self.grant_type == "authorization_code":
            return self._create_authorization_code_auth(server_url=server_url, client_secret=client_secret)

        if not self.client_id or client_secret is None:
            message = "The 'client_credentials' grant needs both `client_id` and `client_secret`."
            raise ValueError(message)
        return create_client_credentials_provider(
            server_url=server_url,
            storage=self.token_storage,
            client_id=self.client_id,
            client_secret=client_secret,
            token_endpoint_auth_method=self.token_endpoint_auth_method,
            scope=" ".join(self.scopes) if self.scopes else None,
            issuer=self.issuer,
        )

    def _create_authorization_code_auth(self, server_url: str, client_secret: str | None) -> Any:
        if self._resolved_redirect_uri is None:
            self._resolved_redirect_uri = _resolve_redirect_uri(self.redirect_uri)
        redirect_uri = self._resolved_redirect_uri

        auth_method = self.token_endpoint_auth_method if client_secret else "none"
        client_metadata = OAuthClientMetadata(
            client_name="Haystack",
            redirect_uris=[AnyUrl(redirect_uri)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method=auth_method,
        )

        storage: TokenStorage = self.token_storage
        if self.client_id:
            pre_registered = OAuthClientInformationFull(
                client_id=self.client_id,
                client_secret=client_secret,
                redirect_uris=[AnyUrl(redirect_uri)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method=auth_method,
            )
            storage = _PreRegisteredClientStorage(storage=storage, client_info=pre_registered)

        flow = _LoopbackAuthorizationFlow(
            redirect_uri=redirect_uri,
            callback_timeout=self.callback_timeout,
            on_authorization_url=self.on_authorization_url,
        )
        return create_authorization_code_provider(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=flow.redirect,
            callback_handler=flow.callback,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this configuration to a dictionary.

        `token_storage` is not serialized.

        :returns: Dictionary representation of this configuration.
        """
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret.to_dict() if self.client_secret else None,
            "issuer": self.issuer,
            "scopes": self.scopes,
            "token_endpoint_auth_method": self.token_endpoint_auth_method,
            "grant_type": self.grant_type,
            "redirect_uri": self.redirect_uri,
            "callback_timeout": self.callback_timeout,
            "on_authorization_url": (
                serialize_callable(self.on_authorization_url) if self.on_authorization_url else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthConfig":
        """
        Deserialize a configuration from a dictionary.

        :param data: Dictionary created by `to_dict`.
        :returns: The deserialized configuration.
        """
        data = data.copy()
        deserialize_secrets_inplace(data, keys=["client_secret"])
        if data.get("on_authorization_url"):
            data["on_authorization_url"] = deserialize_callable(data["on_authorization_url"])
        return cls(**data)


def _validate_loopback_redirect_uri(redirect_uri: str) -> None:
    """
    Check that the callback server can listen on the redirect URI.

    :param redirect_uri: The redirect URI to check.
    :raises ValueError: If the redirect URI isn't an `http` URL on a loopback host.
    """
    parts = urlsplit(redirect_uri)
    try:
        parts.port  # noqa: B018 - raises ValueError for an invalid port
    except ValueError as e:
        message = f"Invalid port in redirect_uri '{redirect_uri}'."
        raise ValueError(message) from e
    if parts.scheme != "http" or parts.hostname not in _LOOPBACK_HOSTS:
        message = (
            f"redirect_uri '{redirect_uri}' must be an http URL on a loopback host (localhost, 127.0.0.1 or [::1]), "
            "where the redirect is captured."
        )
        raise ValueError(message)


def _resolve_redirect_uri(redirect_uri: str | None) -> str:
    """
    Fill in a free port when the redirect URI has none, or port 0.

    The port is only reserved while it is looked up, so another process could take it before the authorization
    redirect arrives. The callback then falls back to asking for the redirected URL.

    :param redirect_uri: The configured redirect URI, or None for the default.
    :returns: The redirect URI with a port.
    """
    parts = urlsplit(redirect_uri or "http://127.0.0.1/callback")
    if parts.port:
        return urlunsplit(parts)

    host = parts.hostname or "127.0.0.1"
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]
    netloc = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    return urlunsplit(parts._replace(netloc=netloc, path=parts.path or "/"))


class _PreRegisteredClientStorage:
    """
    Token storage that supplies the information of a client registered in advance.

    The SDK registers a client dynamically when the storage holds no client information; this wrapper prevents that
    by answering with the configured client until other client information is stored.
    """

    def __init__(self, storage: TokenStorage, client_info: OAuthClientInformationFull) -> None:
        self._storage = storage
        self._client_info = client_info

    async def get_tokens(self) -> OAuthToken | None:
        """Return the stored tokens, if any."""
        return await self._storage.get_tokens()

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store the given tokens."""
        await self._storage.set_tokens(tokens)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return the stored client information, or else the client registered in advance."""
        return await self._storage.get_client_info() or self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store the given client information."""
        await self._storage.set_client_info(client_info)


class _LoopbackAuthorizationFlow:
    """
    Sends the user to the authorization URL and captures the redirect on a loopback address.

    `redirect` and `callback` are the `redirect_handler` and `callback_handler` of the SDK's `OAuthClientProvider`.
    The listening socket is opened in `redirect`, before the user can be redirected, and served in `callback`.
    Only a redirect to the expected path carrying the `state` of the authorization request is accepted, so other
    requests to the port can't inject an authorization code (CSRF protection); the SDK checks `state` again.
    """

    def __init__(
        self,
        redirect_uri: str,
        callback_timeout: float,
        on_authorization_url: Callable[[str], Awaitable[None]] | None,
    ) -> None:
        parts = urlsplit(redirect_uri)
        self._redirect_uri = redirect_uri
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port or 0
        self._path = parts.path or "/"
        self._callback_timeout = callback_timeout
        self._on_authorization_url = on_authorization_url
        self._authorization_url = ""
        self._expected_state: str | None = None
        self._listener: Any = None

    async def redirect(self, authorization_url: str) -> None:
        """
        Start listening for the redirect, then send the user to the authorization URL.

        :param authorization_url: The URL where the user authorizes the client.
        """
        self._authorization_url = authorization_url
        self._expected_state = _first_param(parse_qs(urlsplit(authorization_url).query), "state")

        await self._close_listener()
        try:
            self._listener = await anyio.create_tcp_listener(local_host=self._host, local_port=self._port)
        except OSError as e:
            logger.warning(
                "Can't listen for the OAuth redirect on {redirect_uri}: {error}. "
                "After authorizing, you will be asked for the URL you were redirected to.",
                redirect_uri=self._redirect_uri,
                error=e,
            )

        if self._on_authorization_url is not None:
            try:
                await self._on_authorization_url(authorization_url)
            except BaseException:
                # The SDK won't call `callback` after a failed redirect, so the listener would stay open
                await self._close_listener()
                raise
            return

        if not await anyio.to_thread.run_sync(_open_in_browser, authorization_url):
            logger.warning(
                "Couldn't open a browser. Open this URL to authorize access to the MCP server: {authorization_url}",
                authorization_url=authorization_url,
            )

    async def callback(self) -> Any:
        """
        Wait for the authorization redirect and return its parameters.

        :returns: The redirect parameters, in the shape the installed SDK expects.
        :raises OAuthFlowError: If the authorization server returned an error, or no redirect was received.
        """
        params: dict[str, str] | None = None
        if self._listener is not None:
            try:
                params = await self._wait_for_redirect()
            finally:
                await self._close_listener()
            if params is None:
                logger.warning(
                    "No OAuth redirect received on {redirect_uri} within {timeout} seconds.",
                    redirect_uri=self._redirect_uri,
                    timeout=self._callback_timeout,
                )
        if params is None:
            params = await self._ask_for_redirected_url()

        if "error" in params:
            description = params.get("error_description")
            message = f"Authorization failed: {params['error']}" + (f" ({description})" if description else "")
            raise OAuthFlowError(message)
        if not params.get("code"):
            message = "The authorization redirect carries no authorization code."
            raise OAuthFlowError(message)
        return authorization_code_result(code=params["code"], state=params.get("state"), iss=params.get("iss"))

    async def _wait_for_redirect(self) -> dict[str, str] | None:
        """
        Serve the listening socket until the expected redirect arrives or `callback_timeout` passes.

        :returns: The redirect parameters, or None on timeout.
        """
        listener = self._listener
        result: dict[str, str] | None = None
        received = anyio.Event()

        async def handle(stream: SocketStream) -> None:
            nonlocal result
            async with stream:
                params = await self._handle_request(stream)
            if params is not None and result is None:
                result = params
                received.set()

        with anyio.move_on_after(self._callback_timeout):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(listener.serve, handle)
                await received.wait()
                task_group.cancel_scope.cancel()
        return result

    async def _handle_request(self, stream: SocketStream) -> dict[str, str] | None:
        """
        Answer one HTTP request on the callback server.

        :param stream: The accepted connection.
        :returns: The redirect parameters if this is the expected redirect, otherwise None.
        """
        head = b""
        with anyio.move_on_after(_REQUEST_READ_TIMEOUT):
            try:
                while b"\r\n\r\n" not in head and len(head) < _MAX_REQUEST_HEAD_BYTES:
                    head += await stream.receive()
            except (anyio.EndOfStream, anyio.BrokenResourceError, OSError):
                return None
        request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
        method, _, rest = request_line.partition(" ")
        target = rest.rpartition(" ")[0] or rest
        parts = urlsplit(target)

        if method != "GET" or parts.path != self._path:
            await _send_response(stream, 404, "Not found.")
            return None
        params = {key: values[0] for key, values in parse_qs(parts.query).items()}
        if self._expected_state is not None and params.get("state") != self._expected_state:
            await _send_response(stream, 400, "This authorization response doesn't match the pending request.")
            return None

        if "error" in params:
            await _send_response(stream, 200, f"Authorization failed: {params['error']}. You can close this window.")
        else:
            await _send_response(stream, 200, "Authorization complete. You can close this window.")
        return params

    async def _ask_for_redirected_url(self) -> dict[str, str]:
        """
        Ask the user in the terminal for the URL they were redirected to.

        :returns: The parameters of the pasted URL.
        :raises OAuthFlowError: If the process has no interactive terminal.
        """
        if not _is_interactive():
            message = (
                f"No authorization redirect was received on {self._redirect_uri}, and there is no interactive terminal "
                "to paste the redirected URL into."
            )
            raise OAuthFlowError(message)
        prompt = (
            f"Open this URL to authorize access to the MCP server:\n{self._authorization_url}\n"
            "Then paste the URL you were redirected to: "
        )
        # Abandoned if cancelled, since input() can't be interrupted
        answer = await anyio.to_thread.run_sync(input, prompt, abandon_on_cancel=True)
        answer = answer.strip()
        query = urlsplit(answer).query if "?" in answer else answer
        return {key: values[0] for key, values in parse_qs(query).items()}

    async def _close_listener(self) -> None:
        if self._listener is not None:
            await self._listener.aclose()
            self._listener = None


def _first_param(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    return values[0] if values else None


def _open_in_browser(url: str) -> bool:
    try:
        return webbrowser.open(url)
    except webbrowser.Error:
        return False


def _is_interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


async def _send_response(stream: SocketStream, status: int, text: str) -> None:
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}[status]
    body = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Haystack MCP authorization</title></head>"
        f"<body><p>{html.escape(text)}</p></body></html>"
    ).encode()
    head = (
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode()
    try:
        await stream.send(head + body)
    except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
        pass


class _FailFastAuth(http_lib.Auth):
    """
    Wraps an `mcp` SDK OAuth provider so that a failed OAuth flow aborts the pending MCP request.

    The SDK raises OAuth errors (e.g. rejected client credentials) inside the transport's background task, where
    they are only logged: the session keeps waiting for a response until it times out. This wrapper records the
    error and cancels the scope returned by `failure_scope()`, so the caller can fail immediately with the error.
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.requires_request_body = provider.requires_request_body
        self.requires_response_body = provider.requires_response_body
        self.error: OAuthFlowError | None = None
        self._scope: anyio.CancelScope | None = None

    def failure_scope(self) -> anyio.CancelScope:
        """
        Return a cancel scope that is cancelled when the OAuth flow fails while it is active.

        :returns: The cancel scope to wrap the MCP requests with.
        """
        self.error = None
        self._scope = anyio.CancelScope()
        return self._scope

    async def async_auth_flow(self, request: Any) -> AsyncGenerator[Any, Any]:
        """Run the wrapped provider's auth flow, recording OAuth errors."""
        flow = self._provider.async_auth_flow(request)
        try:
            outgoing = await flow.__anext__()
            while True:
                response = yield outgoing
                outgoing = await flow.asend(response)
        except StopAsyncIteration:
            return
        except OAuthFlowError as e:
            self.error = e
            if self._scope is not None:
                self._scope.cancel()
            raise
        finally:
            await flow.aclose()
