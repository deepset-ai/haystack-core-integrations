# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
OAuth 2.0 support for MCP server connections.

Authentication is delegated to the OAuth providers shipped with the `mcp` SDK, which handle the MCP authorization
spec: authorization server discovery (RFC 9728 and RFC 8414), the `resource` parameter (RFC 8707), and fetching a
new token when the MCP server rejects the current one.

Only the Client Credentials grant (RFC 6749 §4.4) is supported for now.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
from haystack.utils import Secret, deserialize_secrets_inplace

from mcp.client.auth import OAuthFlowError, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from .compatibility_layer import create_client_credentials_provider, http_lib


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

    Supports the Client Credentials grant (RFC 6749 §4.4), which is non-interactive and suited for
    machine-to-machine connections. The authorization server and its token endpoint are discovered from the
    MCP server, as described in the MCP authorization spec.

    Usage example:
    ```python
    from haystack.utils import Secret
    from haystack_integrations.tools.mcp import MCPToolset, OAuthConfig, StreamableHttpServerInfo

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
    ```

    :param client_id: OAuth client identifier.
    :param client_secret: OAuth client secret.
    :param issuer: Issuer identifier of the authorization server that issued the client credentials.
        When set, tokens are only requested from an authorization server whose metadata names exactly this issuer.
        Recommended, and required by future versions of the `mcp` SDK. Needs `mcp>=1.30.0` (v1) or `mcp>=2.2.0`
        (v2).
    :param scopes: OAuth scopes to request.
    :param token_endpoint_auth_method: How the client authenticates to the token endpoint:
        `"client_secret_basic"` (HTTP Basic, the default) or `"client_secret_post"` (credentials in the request body).
    :param grant_type: The OAuth grant type. Only `"client_credentials"` is supported for now.
    :param token_storage: Where tokens are kept between connections. Any implementation of the
        `mcp.client.auth.TokenStorage` protocol works. Defaults to an `InMemoryTokenStorage`. Not serialized.
    """

    client_id: str
    client_secret: Secret
    issuer: str | None = None
    scopes: list[str] | None = None
    token_endpoint_auth_method: Literal["client_secret_basic", "client_secret_post"] = "client_secret_basic"
    grant_type: Literal["client_credentials"] = "client_credentials"
    token_storage: TokenStorage = field(default_factory=InMemoryTokenStorage, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.grant_type != "client_credentials":
            message = f"Unsupported grant_type '{self.grant_type}'. Only 'client_credentials' is supported."
            raise ValueError(message)
        if self.token_endpoint_auth_method not in ("client_secret_basic", "client_secret_post"):
            message = (
                f"Unsupported token_endpoint_auth_method '{self.token_endpoint_auth_method}'. "
                "Use 'client_secret_basic' or 'client_secret_post'."
            )
            raise ValueError(message)

    def create_auth(self, server_url: str) -> Any:
        """
        Create the `httpx.Auth` that authenticates requests to the given MCP server.

        :param server_url: URL of the MCP server.
        :returns: An `mcp` SDK OAuth provider, to be passed as `auth` to the HTTP client.
        :raises ValueError: If the client secret can't be resolved.
        """
        client_secret = self.client_secret.resolve_value()
        if not client_secret:
            message = "OAuthConfig.client_secret resolved to an empty value."
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

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this configuration to a dictionary.

        `token_storage` is not serialized.

        :returns: Dictionary representation of this configuration.
        """
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret.to_dict(),
            "issuer": self.issuer,
            "scopes": self.scopes,
            "token_endpoint_auth_method": self.token_endpoint_auth_method,
            "grant_type": self.grant_type,
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
        return cls(**data)


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
