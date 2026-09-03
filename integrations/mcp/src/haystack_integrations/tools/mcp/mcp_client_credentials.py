# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""OAuth2 client-credentials tokens for MCP servers."""

import base64
import threading
import time
from typing import Any, Literal

import httpx
from haystack import logging
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.tools.mcp.mcp_token_provider import MCPTokenProvider

logger = logging.getLogger(__name__)

ClientAuthMethod = Literal["client_secret_basic", "client_secret_post"]

DEFAULT_EXPIRY_BUFFER_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 30
HTTP_OK = 200

# Long enough that a token with no advertised lifetime is not refetched constantly, short enough
# that a silently rotated one is picked up without waiting for a rejection.
NO_EXPIRY_LIFETIME_SECONDS = 3600


class MCPTokenRequestError(Exception):
    """Raised when an access token could not be obtained."""


class ClientCredentialsTokenProvider(MCPTokenProvider):
    """
    Obtains access tokens from an OAuth2 token endpoint using the client-credentials grant.

    This is the machine-to-machine case: the pipeline itself is the client, so there is no user to
    send through a consent screen. Tokens are fetched on demand, cached until shortly before they
    expire, and re-fetched when the server rejects one.

    ```python
    from haystack.utils import Secret
    from haystack_integrations.tools.mcp import ClientCredentialsTokenProvider, StreamableHttpServerInfo

    server_info = StreamableHttpServerInfo(
        url="https://mcp.example.com/mcp",
        token_provider=ClientCredentialsTokenProvider(
            token_url="https://auth.example.com/oauth/token",
            client_id="my-client",
            client_secret=Secret.from_env_var("MCP_CLIENT_SECRET"),
            scope="mcp:read",
        ),
    )
    ```

    Unlike a ``token`` set from a ``Secret``, which is read once when the client is built, this is
    consulted per request — so a pipeline that runs longer than a token's lifetime keeps working.

    :param token_url: The OAuth2 token endpoint
    :param client_id: Client identifier issued by the authorization server
    :param client_secret: Client secret, as a ``Secret`` so it is never serialized in the clear
    :param scope: Space-separated scopes to request, when the server needs them
    :param client_auth_method: How to present the client credentials. ``client_secret_basic`` sends
        an HTTP Basic header, which RFC 6749 requires every authorization server to accept;
        ``client_secret_post`` puts them in the form body, which some servers require instead.
    :param expiry_buffer_seconds: Refresh this many seconds before a token actually expires, so a
        token does not lapse in flight
    :param timeout_seconds: Timeout for the token request
    """

    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        client_secret: Secret,
        scope: str | None = None,
        client_auth_method: ClientAuthMethod = "client_secret_basic",
        expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.client_auth_method: ClientAuthMethod = client_auth_method
        self.expiry_buffer_seconds = expiry_buffer_seconds
        self.timeout_seconds = timeout_seconds

        # A tool call can be made from more than one thread, and every HTTP request asks for a
        # token, so the cache is guarded rather than merely assigned.
        self._lock = threading.Lock()
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def token(self) -> str:
        """
        Returns a cached token, fetching a new one when none is held or it is about to expire.

        :returns: A bearer token
        :raises MCPTokenRequestError: If the token endpoint cannot be reached or refuses
        """
        with self._lock:
            if self._access_token is not None and time.monotonic() < self._expires_at:
                return self._access_token

            token, expires_in = self._fetch()
            self._access_token = token
            self._expires_at = time.monotonic() + max(expires_in - self.expiry_buffer_seconds, 0)
            return token

    def invalidate(self) -> None:
        """
        Drops the cached token, so the next call fetches a fresh one.

        Called when a server rejected the token — which can happen well before it was due to
        expire, for instance after it was revoked.
        """
        with self._lock:
            self._access_token = None
            self._expires_at = 0.0

    def _fetch(self) -> tuple[str, int]:
        """
        Performs the client-credentials request.

        :returns: The access token and its lifetime in seconds
        :raises MCPTokenRequestError: If the request fails or the response carries no token
        """
        data: dict[str, str] = {"grant_type": "client_credentials"}
        if self.scope:
            data["scope"] = self.scope

        headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        secret = self.client_secret.resolve_value() or ""

        if self.client_auth_method == "client_secret_basic":
            encoded = base64.b64encode(f"{self.client_id}:{secret}".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        else:
            data["client_id"] = self.client_id
            data["client_secret"] = secret

        try:
            response = httpx.post(self.token_url, data=data, headers=headers, timeout=self.timeout_seconds)
        except httpx.HTTPError as error:
            message = f"Could not reach the token endpoint {self.token_url}: {error}"
            raise MCPTokenRequestError(message) from error

        if response.status_code != HTTP_OK:
            message = f"Token endpoint {self.token_url} refused the request with status {response.status_code}."
            raise MCPTokenRequestError(message)

        try:
            document = response.json()
        except ValueError as error:
            message = f"Token endpoint {self.token_url} did not return JSON."
            raise MCPTokenRequestError(message) from error

        access_token = document.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            message = f"Token endpoint {self.token_url} returned no access token."
            raise MCPTokenRequestError(message)

        # `expires_in` is optional. Treating an absent lifetime as zero would refetch on every
        # request; instead the token is held until a server rejects it.
        try:
            lifetime = int(float(document["expires_in"]))
        except (KeyError, TypeError, ValueError):
            logger.debug("Token endpoint returned no usable expires_in; holding the token for an hour")
            lifetime = NO_EXPIRY_LIFETIME_SECONDS

        return access_token, max(lifetime, 0)

    def _init_parameters(self) -> dict[str, Any]:
        """The keyword arguments needed to rebuild this provider, with the secret by reference."""
        return {
            "token_url": self.token_url,
            "client_id": self.client_id,
            "client_secret": self.client_secret.to_dict(),
            "scope": self.scope,
            "client_auth_method": self.client_auth_method,
            "expiry_buffer_seconds": self.expiry_buffer_seconds,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClientCredentialsTokenProvider":
        """
        Rebuilds the provider, resolving the client secret back into a ``Secret``.

        :param data: A descriptor produced by ``to_dict``
        :returns: The reconstructed provider
        """
        init_parameters = dict(data.get("init_parameters") or {})
        deserialize_secrets_inplace(init_parameters, keys=["client_secret"])
        return cls(**init_parameters)
