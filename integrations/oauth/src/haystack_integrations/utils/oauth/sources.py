# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock
from typing import Any

import httpx
from haystack import default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_callable, serialize_callable

from .errors import OAuthConfigError, TokenRefreshError

logger = logging.getLogger(__name__)

# RFC 8693 OAuth 2.0 Token Exchange grant
DEFAULT_TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
DEFAULT_EXPIRY_BUFFER_SECONDS = 300
DEFAULT_ACCESS_TOKEN_TTL_SECONDS = 3600
DEFAULT_CACHE_MAX_SIZE = 1000
DEFAULT_TIMEOUT_SECONDS = 30.0


def _parse_token_response(response: httpx.Response) -> tuple[str, float, dict[str, Any]]:
    """Validate an OAuth token endpoint response and extract the access token, its TTL, and the raw payload."""
    if response.is_error:
        msg = f"Token request failed: {response.status_code} {response.text}"
        raise TokenRefreshError(msg, status_code=response.status_code)
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        msg = "Token response did not contain an access_token."
        raise TokenRefreshError(msg)
    raw_expires_in = payload.get("expires_in", DEFAULT_ACCESS_TOKEN_TTL_SECONDS)
    try:
        expires_in = float(raw_expires_in)
    except (TypeError, ValueError):
        logger.warning(
            "Token response had a non-numeric 'expires_in' value ({value}); falling back to {default}s.",
            value=raw_expires_in,
            default=DEFAULT_ACCESS_TOKEN_TTL_SECONDS,
        )
        expires_in = float(DEFAULT_ACCESS_TOKEN_TTL_SECONDS)
    return access_token, expires_in, payload


class RefreshTokenSource:
    """
    Resolves access tokens by running the RFC 6749 refresh-token grant against an OAuth token endpoint.

    Given a stored refresh token plus client credentials, it exchanges them for an access token and caches it in
    process until shortly before expiry. If the identity provider rotates the refresh token on exchange, the new value
    is kept for the lifetime of the process and surfaced through the optional `on_rotate` callback so it can be
    persisted.

    This source is **single-identity**: one refresh token per instance. For per-user, multi-tenant resolution that
    needs no persistent storage, use `TokenExchangeSource` instead. It takes no per-request input.
    """

    requires_subject_token = False

    def __init__(
        self,
        token_url: str,
        client_id: str,
        *,
        refresh_token: Secret = Secret.from_env_var("OAUTH_REFRESH_TOKEN"),
        client_secret: Secret | None = None,
        scopes: list[str] | None = None,
        scope_delimiter: str = " ",
        expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        on_rotate: Callable[[str], None] | None = None,
    ) -> None:
        """
        Initialize the source.

        :param token_url: The OAuth 2.0 token endpoint.
        :param client_id: The OAuth client identifier.
        :param refresh_token: The refresh token to exchange. Defaults to the value of the `OAUTH_REFRESH_TOKEN`
            environment variable.
        :param client_secret: The client secret for confidential clients. Omit it for public clients.
        :param scopes: The scopes to request, joined with `scope_delimiter`.
        :param scope_delimiter: The delimiter used to join scopes. Defaults to a space (some providers use a comma).
        :param expiry_buffer_seconds: Refresh the cached access token this many seconds before its declared expiry.
        :param timeout: The timeout, in seconds, for the request to the token endpoint.
        :param on_rotate: An optional callback invoked with the new refresh token whenever the provider rotates it.
            Use it to persist the rotated token durably (the source itself only keeps it in process).
        :raises OAuthConfigError: If the configuration is invalid.
        """
        if not token_url:
            msg = "token_url must be a non-empty string."
            raise OAuthConfigError(msg)
        if not client_id:
            msg = "client_id must be a non-empty string."
            raise OAuthConfigError(msg)
        if expiry_buffer_seconds < 0:
            msg = "expiry_buffer_seconds must be non-negative."
            raise OAuthConfigError(msg)

        self.token_url = token_url
        self.client_id = client_id
        self.refresh_token = refresh_token
        self.client_secret = client_secret
        self.scopes = list(scopes) if scopes is not None else None
        self.scope_delimiter = scope_delimiter
        self.expiry_buffer_seconds = expiry_buffer_seconds
        self.timeout = timeout
        self.on_rotate = on_rotate

        # Runtime state — never serialized.
        self._sync_lock = Lock()
        self._async_lock: asyncio.Lock | None = None
        self._cached_token: str | None = None
        self._expires_at: float = 0.0
        self._current_refresh_token: str | None = None

    def _cached_if_valid(self) -> str | None:
        if self._cached_token and time.monotonic() < self._expires_at - self.expiry_buffer_seconds:
            return self._cached_token
        return None

    def _build_request_data(self) -> dict[str, str]:
        refresh_token = self._current_refresh_token or self.refresh_token.resolve_value()
        request_data: dict[str, str] = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": refresh_token or "",
        }
        if self.client_secret is not None:
            client_secret = self.client_secret.resolve_value()
            if client_secret:
                request_data["client_secret"] = client_secret
        if self.scopes:
            request_data["scope"] = self.scope_delimiter.join(self.scopes)
        return request_data

    def _handle_response(self, response: httpx.Response) -> str:
        access_token, expires_in, payload = _parse_token_response(response)
        self._cached_token = access_token
        self._expires_at = time.monotonic() + expires_in
        new_refresh_token = payload.get("refresh_token")
        if new_refresh_token:
            self._current_refresh_token = new_refresh_token
            if self.on_rotate is not None:
                self.on_rotate(new_refresh_token)
        logger.debug("Refreshed access token via {token_url}.", token_url=self.token_url)
        return access_token

    def resolve(self) -> str:
        """Return a cached access token, or run the refresh-token grant to obtain a fresh one."""
        # Refreshes for this single-identity source are serialized, so a burst of concurrent callers
        # collapses into one network efresh and the shared cache + on_rotate aren't updated concurrently
        with self._sync_lock:
            cached = self._cached_if_valid()
            if cached is not None:
                return cached
            response = httpx.post(
                self.token_url,
                data=self._build_request_data(),
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            return self._handle_response(response)

    async def resolve_async(self) -> str:
        """Asynchronous counterpart of `resolve`. Use a single instance in either sync or async mode, not both."""
        # Create the asyncio.Lock lazily (not in __init__): it must bind to the running event loop, but __init__
        # is sync and may run without one.
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            cached = self._cached_if_valid()
            if cached is not None:
                return cached
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data=self._build_request_data(),
                    headers={"Accept": "application/json"},
                    timeout=self.timeout,
                )
            return self._handle_response(response)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source to a dictionary."""
        return default_to_dict(
            self,
            token_url=self.token_url,
            client_id=self.client_id,
            refresh_token=self.refresh_token.to_dict(),
            client_secret=self.client_secret.to_dict() if self.client_secret else None,
            scopes=self.scopes,
            scope_delimiter=self.scope_delimiter,
            expiry_buffer_seconds=self.expiry_buffer_seconds,
            timeout=self.timeout,
            on_rotate=serialize_callable(self.on_rotate) if self.on_rotate else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefreshTokenSource":
        """Deserialize the source from a dictionary."""
        init_params = data["init_parameters"]
        if init_params.get("on_rotate"):
            init_params["on_rotate"] = deserialize_callable(init_params["on_rotate"])
        return default_from_dict(cls, data)


class TokenExchangeSource:
    """
    Resolves access tokens by exchanging a per-request subject token at an OAuth token endpoint.

    This implements RFC 8693 token exchange (and, via configuration, Microsoft's on-behalf-of flow). Unlike
    `RefreshTokenSource`, it is **multi-user without any persistent storage**: the per-request `subject_token` (the
    incoming user assertion) *is* the user identity and is exchanged fresh for a downstream token. Resolved tokens
    are cached in memory per subject token (bounded, LRU) until shortly before expiry.

    Provider differences are expressed as configuration: `grant_type`, `subject_token_param` (for example
    `assertion` for Microsoft), `scopes`, and `extra_token_params` (for example
    `{"requested_token_use": "on_behalf_of"}`).
    """

    requires_subject_token = True

    def __init__(
        self,
        token_url: str,
        client_id: str,
        *,
        client_secret: Secret | None = None,
        grant_type: str = DEFAULT_TOKEN_EXCHANGE_GRANT,
        subject_token_param: str = "subject_token",
        subject_token_type: str | None = None,
        requested_token_type: str | None = None,
        scopes: list[str] | None = None,
        scope_delimiter: str = " ",
        extra_token_params: dict[str, str] | None = None,
        expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
        cache_max_size: int = DEFAULT_CACHE_MAX_SIZE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize the source.

        :param token_url: The OAuth 2.0 token endpoint.
        :param client_id: The OAuth client identifier.
        :param client_secret: The client secret for confidential clients. Omit it for public clients.
        :param grant_type: The grant type sent as the `grant_type` form parameter. Defaults to the RFC 8693
            token-exchange grant. Set it to the value your provider expects (for example the
            `urn:ietf:params:oauth:grant-type:jwt-bearer` grant for Microsoft on-behalf-of).
        :param subject_token_param: The name of the form parameter carrying the per-request subject token. Defaults
            to `subject_token` (RFC 8693). Some providers expect a different name, such as `assertion`.
        :param subject_token_type: The RFC 8693 identifier for the type of the supplied subject token, sent as the
            `subject_token_type` form parameter (omitted when not set). Required by RFC 8693 token exchange
            (e.g. `urn:ietf:params:oauth:token-type:access_token`); not used by Microsoft's on-behalf-of flow.
        :param requested_token_type: The RFC 8693 identifier for the token to return, sent as the
            `requested_token_type` form parameter (omitted when not set). Optional.
        :param scopes: The scopes to request, joined with `scope_delimiter`.
        :param scope_delimiter: The delimiter used to join scopes. Defaults to a space.
        :param extra_token_params: Additional form parameters included verbatim in every request (for example
            `{"requested_token_use": "on_behalf_of"}`).
        :param expiry_buffer_seconds: Refresh a cached access token this many seconds before its declared expiry.
        :param cache_max_size: The maximum number of per-user tokens to keep in the in-memory cache. The
            least-recently-used entry is evicted when the cache is full.
        :param timeout: The timeout, in seconds, for the request to the token endpoint.
        :raises OAuthConfigError: If the configuration is invalid.
        """
        if not token_url:
            msg = "token_url must be a non-empty string."
            raise OAuthConfigError(msg)
        if not client_id:
            msg = "client_id must be a non-empty string."
            raise OAuthConfigError(msg)
        if expiry_buffer_seconds < 0:
            msg = "expiry_buffer_seconds must be non-negative."
            raise OAuthConfigError(msg)
        if cache_max_size < 1:
            msg = "cache_max_size must be a positive integer."
            raise OAuthConfigError(msg)

        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.grant_type = grant_type
        self.subject_token_param = subject_token_param
        self.subject_token_type = subject_token_type
        self.requested_token_type = requested_token_type
        self.scopes = list(scopes) if scopes is not None else None
        self.scope_delimiter = scope_delimiter
        self.extra_token_params = dict(extra_token_params) if extra_token_params else None
        self.expiry_buffer_seconds = expiry_buffer_seconds
        self.cache_max_size = cache_max_size
        self.timeout = timeout

        # Runtime state — never serialized.
        # `_lock` guards the shared LRU cache. It's held only around cache access, never across the network call, so
        # distinct subjects (users) exchange tokens concurrently.
        self._lock = Lock()
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()

    @staticmethod
    def _cache_key(subject_token: str) -> str:
        return hashlib.sha256(subject_token.encode("utf-8")).hexdigest()

    def _cached_if_valid(self, cache_key: str) -> str | None:
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        token, expires_at = entry
        if time.monotonic() >= expires_at - self.expiry_buffer_seconds:
            del self._cache[cache_key]
            return None
        self._cache.move_to_end(cache_key)
        return token

    def _store(self, cache_key: str, token: str, expires_in: float) -> None:
        self._cache[cache_key] = (token, time.monotonic() + expires_in)
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_max_size:
            self._cache.popitem(last=False)

    def _build_request_data(self, subject_token: str) -> dict[str, str]:
        request_data: dict[str, str] = {
            "grant_type": self.grant_type,
            self.subject_token_param: subject_token,
            "client_id": self.client_id,
        }
        if self.subject_token_type:
            request_data["subject_token_type"] = self.subject_token_type
        if self.requested_token_type:
            request_data["requested_token_type"] = self.requested_token_type
        if self.client_secret is not None:
            client_secret = self.client_secret.resolve_value()
            if client_secret:
                request_data["client_secret"] = client_secret
        if self.scopes:
            request_data["scope"] = self.scope_delimiter.join(self.scopes)
        if self.extra_token_params:
            request_data.update(self.extra_token_params)
        return request_data

    def resolve(self, subject_token: str) -> str:
        """Exchange the per-request `subject_token` for an access token (cached per subject token)."""
        if not subject_token:
            msg = "TokenExchangeSource requires a non-empty subject_token."
            raise OAuthConfigError(msg)
        cache_key = self._cache_key(subject_token)
        with self._lock:
            cached = self._cached_if_valid(cache_key)
        if cached is not None:
            return cached
        response = httpx.post(
            self.token_url,
            data=self._build_request_data(subject_token),
            headers={"Accept": "application/json"},
            timeout=self.timeout,
        )
        access_token, expires_in, _ = _parse_token_response(response)
        with self._lock:
            self._store(cache_key, access_token, expires_in)
        return access_token

    async def resolve_async(self, subject_token: str) -> str:
        """Asynchronous counterpart of `resolve`."""
        if not subject_token:
            msg = "TokenExchangeSource requires a non-empty subject_token."
            raise OAuthConfigError(msg)
        cache_key = self._cache_key(subject_token)
        with self._lock:
            cached = self._cached_if_valid(cache_key)
        if cached is not None:
            return cached
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=self._build_request_data(subject_token),
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
        access_token, expires_in, _ = _parse_token_response(response)
        with self._lock:
            self._store(cache_key, access_token, expires_in)
        return access_token

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source to a dictionary."""
        return default_to_dict(
            self,
            token_url=self.token_url,
            client_id=self.client_id,
            client_secret=self.client_secret.to_dict() if self.client_secret else None,
            grant_type=self.grant_type,
            subject_token_param=self.subject_token_param,
            subject_token_type=self.subject_token_type,
            requested_token_type=self.requested_token_type,
            scopes=self.scopes,
            scope_delimiter=self.scope_delimiter,
            extra_token_params=self.extra_token_params,
            expiry_buffer_seconds=self.expiry_buffer_seconds,
            cache_max_size=self.cache_max_size,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenExchangeSource":
        """Deserialize the source from a dictionary."""
        return default_from_dict(cls, data)


class StaticTokenSource:
    """
    Returns a configured long-lived access token as-is.

    Suitable for providers that issue non-expiring tokens (for example Slack or Notion), where no refresh flow is
    needed and the token is managed out of band. It takes no per-request input.
    """

    requires_subject_token = False

    def __init__(self, token: Secret) -> None:
        """
        Initialize the source.

        :param token: The long-lived access token to return.
        """
        self.token = token

    def resolve(self) -> str:
        """Return the configured token."""
        try:
            value = self.token.resolve_value()
        except ValueError as error:
            # A strict `EnvVarSecret` raises ValueError when its env var is unset; surface it as our error type.
            msg = "StaticTokenSource could not resolve its token."
            raise TokenRefreshError(msg) from error
        if not value:
            msg = "StaticTokenSource has no token value."
            raise TokenRefreshError(msg)
        return value

    async def resolve_async(self) -> str:
        """Asynchronous counterpart of `resolve`."""
        return self.resolve()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source to a dictionary."""
        return default_to_dict(self, token=self.token.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticTokenSource":
        """Deserialize the source from a dictionary."""
        return default_from_dict(cls, data)
