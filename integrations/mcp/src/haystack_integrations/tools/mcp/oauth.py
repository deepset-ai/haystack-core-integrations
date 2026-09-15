# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
OAuth 2.0 Client Credentials support for MCP server connections.

Only the Client Credentials grant (RFC 6749 §4.4) is implemented here.
Authorization Code + PKCE requires a local callback server (see issue #2547)
and will be added in a follow-up.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from haystack import logging
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.auth import SecretType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TokenData:
    """Cached OAuth token with expiry information."""

    access_token: str
    token_type: str = "Bearer"
    expires_at: float = 0.0
    scope: str | None = None

    def is_valid(self, leeway: float = 30.0) -> bool:
        """Return True if the token has not yet expired (with a leeway buffer in seconds)."""
        if self.expires_at == 0.0:
            return True
        return time.time() < (self.expires_at - leeway)


class TokenStorage(ABC):
    """Abstract token storage interface."""

    @abstractmethod
    def get_token(self, key: str) -> TokenData | None:
        """Return a cached token for *key*, or None if absent / expired."""

    @abstractmethod
    def store_token(self, key: str, token: TokenData) -> None:
        """Persist *token* under *key*."""

    @abstractmethod
    def clear_token(self, key: str) -> None:
        """Remove the cached token for *key* if present."""


class InMemoryTokenStorage(TokenStorage):
    """In-process token cache. Tokens are lost when the process exits."""

    def __init__(self) -> None:
        self._store: dict[str, TokenData] = {}

    def get_token(self, key: str) -> TokenData | None:
        """Return the cached token for *key*, or None if absent or expired."""
        token = self._store.get(key)
        if token is None:
            return None
        if not token.is_valid():
            del self._store[key]
            return None
        return token

    def store_token(self, key: str, token: TokenData) -> None:
        """Persist *token* under *key*."""
        self._store[key] = token

    def clear_token(self, key: str) -> None:
        """Remove the cached token for *key* if present."""
        self._store.pop(key, None)


def _default_storage() -> InMemoryTokenStorage:
    return InMemoryTokenStorage()


@dataclass
class OAuthConfig:
    """
    OAuth 2.0 configuration for MCP server authentication.

    Currently supports the **Client Credentials** grant (RFC 6749 §4.4),
    which is non-interactive and suitable for machine-to-machine connections.

    Usage example:
    ```python
    from haystack.utils import Secret
    from haystack_integrations.tools.mcp import OAuthConfig, StreamableHttpServerInfo

    server_info = StreamableHttpServerInfo(
        url="https://my-mcp-server.com/mcp",
        oauth_config=OAuthConfig(
            client_id="my-client-id",
            client_secret=Secret.from_env_var("MCP_CLIENT_SECRET"),
            token_endpoint="https://auth.example.com/oauth/token",
            scopes=["mcp:tools"],
        ),
    )
    ```

    :param client_id: OAuth client identifier.
    :param client_secret: OAuth client secret (use a `Secret` for secure handling).
    :param token_endpoint: Full URL of the token endpoint.
        Required for Client Credentials; optional when `authorization_server_url`
        is set (the endpoint is then discovered via ``<authorization_server_url>/.well-known/…``).
    :param authorization_server_url: Base URL of the authorisation server.
        Used for metadata discovery when `token_endpoint` is not given.
    :param scopes: OAuth scopes to request.
    :param token_storage: Where to cache tokens.  Defaults to `InMemoryTokenStorage`.
    """

    client_id: str
    client_secret: str | Secret | None = None
    token_endpoint: str | None = None
    authorization_server_url: str | None = None
    scopes: list[str] | None = None
    token_storage: TokenStorage = field(default_factory=_default_storage)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary (token_storage is not serialized)."""
        secret = self.client_secret
        return {
            "type": "haystack_integrations.tools.mcp.oauth.OAuthConfig",
            "client_id": self.client_id,
            "client_secret": secret.to_dict() if isinstance(secret, Secret) else secret,
            "token_endpoint": self.token_endpoint,
            "authorization_server_url": self.authorization_server_url,
            "scopes": self.scopes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthConfig:
        """Deserialize from a dictionary produced by `to_dict`."""
        d = {k: v for k, v in data.items() if k != "type"}
        secret_types = {e.value for e in SecretType}
        cs = d.get("client_secret")
        if isinstance(cs, dict) and cs.get("type") in secret_types:
            deserialize_secrets_inplace(d, keys=["client_secret"])
        return cls(**d)


def _discover_token_endpoint(authorization_server_url: str) -> str:
    """
    Discover the token endpoint via OAuth 2.0 Server Metadata (RFC 8414).

    :param authorization_server_url: Base URL of the authorisation server.
    :returns: Token endpoint URL.
    :raises ValueError: If discovery fails or the endpoint is not found.
    """
    base = authorization_server_url.rstrip("/")
    # Try RFC 8414 well-known first, then OIDC discovery
    for path in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
        try:
            resp = httpx.get(f"{base}{path}", timeout=10)
            if resp.status_code == 200:  # noqa: PLR2004
                metadata = resp.json()
                endpoint = metadata.get("token_endpoint")
                if endpoint:
                    return str(endpoint)
        except httpx.HTTPError:
            continue
    msg = f"Could not discover token_endpoint from {authorization_server_url}"
    raise ValueError(msg)


def _fetch_client_credentials_token(oauth_config: OAuthConfig) -> TokenData:
    """
    Obtain an access token via the OAuth 2.0 Client Credentials grant.

    :param oauth_config: OAuth configuration containing client credentials and endpoint.
    :returns: A `TokenData` instance with the obtained token.
    :raises ValueError: If neither `token_endpoint` nor `authorization_server_url` is set.
    :raises httpx.HTTPStatusError: If the token endpoint returns an error.
    """
    token_endpoint = oauth_config.token_endpoint
    if not token_endpoint:
        if not oauth_config.authorization_server_url:
            msg = "OAuthConfig requires either token_endpoint or authorization_server_url"
            raise ValueError(msg)
        token_endpoint = _discover_token_endpoint(oauth_config.authorization_server_url)

    payload: dict[str, str] = {
        "grant_type": "client_credentials",
        "client_id": oauth_config.client_id,
    }

    secret = oauth_config.client_secret
    client_secret_value = secret.resolve_value() if isinstance(secret, Secret) else secret
    if client_secret_value:
        payload["client_secret"] = client_secret_value

    if oauth_config.scopes:
        payload["scope"] = " ".join(oauth_config.scopes)

    resp = httpx.post(token_endpoint, data=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()

    expires_in = body.get("expires_in")
    expires_at = (time.time() + float(expires_in)) if expires_in else 0.0

    return TokenData(
        access_token=body["access_token"],
        token_type=body.get("token_type", "Bearer"),
        expires_at=expires_at,
        scope=body.get("scope"),
    )


def ensure_oauth(server_info: Any) -> None:
    """
    Ensure that a valid OAuth access token is injected into *server_info.headers*.

    If the server info has no `oauth_config` this is a no-op.  Otherwise the
    token cache is checked first; a fresh token is fetched only when necessary.
    The token is written to ``server_info.headers["Authorization"]`` and takes
    precedence over the ``token`` field.

    :param server_info: An `SSEServerInfo` or `StreamableHttpServerInfo` instance.
    """
    oauth_config: OAuthConfig | None = getattr(server_info, "oauth_config", None)
    if oauth_config is None:
        return

    cache_key = f"{oauth_config.client_id}@{oauth_config.token_endpoint or oauth_config.authorization_server_url}"
    token = oauth_config.token_storage.get_token(cache_key)

    if token is None:
        logger.info("Fetching OAuth token for client_id={cid}", cid=oauth_config.client_id)
        token = _fetch_client_credentials_token(oauth_config)
        oauth_config.token_storage.store_token(cache_key, token)
        logger.info("OAuth token obtained (expires_at={exp})", exp=token.expires_at or "never")

    # Inject into headers — takes precedence over the token field
    if server_info.headers is None:
        server_info.headers = {}
    server_info.headers["Authorization"] = f"{token.token_type} {token.access_token}"
