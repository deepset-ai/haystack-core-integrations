# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import binascii
import hashlib
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock
from typing import Any, Literal

import httpx
from haystack import default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_callable, serialize_callable

from .errors import OAuthConfigError, TokenRefreshError
from .protocols import TokenSource

with LazyImport("Run 'pip install \"oauth-haystack[jwt]\"' to sign client assertions") as jwt_import:
    import jwt

logger = logging.getLogger(__name__)

# RFC 8693 OAuth 2.0 Token Exchange grant
DEFAULT_TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
# RFC 7523 §2.2 JWT client assertion
DEFAULT_JWT_BEARER_CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
DEFAULT_EXPIRY_BUFFER_SECONDS = 300
DEFAULT_ACCESS_TOKEN_TTL_SECONDS = 3600
DEFAULT_CACHE_MAX_SIZE = 1000
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CLIENT_ASSERTION_LIFETIME_SECONDS = 300

# The first certificate of a PEM bundle; in a chain that is the leaf certificate the private key belongs to.
_PEM_CERTIFICATE = re.compile(r"-----BEGIN CERTIFICATE-----(.+?)-----END CERTIFICATE-----", re.DOTALL)


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


def _validate_client_auth(client_secret: Secret | None, client_assertion_source: TokenSource | None) -> None:
    """Reject client-authentication settings that cannot work."""
    if client_assertion_source is None:
        return
    if client_secret is not None:
        msg = "client_secret and client_assertion_source are mutually exclusive; set only one of them."
        raise OAuthConfigError(msg)
    # `runtime_checkable` only checks that the members exist, so a `SubjectTokenSource` passes `isinstance` too;
    # its `resolve` needs a per-request subject token that is not available when authenticating the client.
    if not isinstance(client_assertion_source, TokenSource) or client_assertion_source.requires_subject_token:
        msg = "client_assertion_source must be a config-only TokenSource (requires_subject_token = False)."
        raise OAuthConfigError(msg)


def _resolve_client_assertion(source: TokenSource | None) -> str | None:
    """Resolve a fresh client assertion, or return None when client-assertion authentication is not configured."""
    if source is None:
        return None
    assertion = source.resolve()
    if not assertion:
        msg = "The client_assertion_source returned an empty client assertion."
        raise TokenRefreshError(msg)
    return assertion


async def _resolve_client_assertion_async(source: TokenSource | None) -> str | None:
    """Asynchronous counterpart of `_resolve_client_assertion`."""
    if source is None:
        return None
    assertion = await source.resolve_async()
    if not assertion:
        msg = "The client_assertion_source returned an empty client assertion."
        raise TokenRefreshError(msg)
    return assertion


class _PerLoopAsyncLock:
    """
    Hands out the `asyncio.Lock` of the running event loop, building it if there isn't one yet.

    The lock cannot be created in a source's `__init__`, which is synchronous and may run with no loop. It also
    cannot be created once and kept for the lifetime of the source: an `asyncio.Lock` binds itself to the loop that
    first awaits it under contention, and raises `RuntimeError` when awaited from any other one. A source routinely
    outlives the loop that first used it — one `asyncio.run` per request is a common deployment — so the lock is
    rebuilt whenever the running loop changes.

    `_guard` is held for the check and the assignment only, never across an await, so that two threads driving
    separate event loops cannot each install a lock and lose mutual exclusion between them.
    """

    def __init__(self) -> None:
        self._guard = Lock()
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> asyncio.Lock:
        """Return the lock for the running event loop."""
        loop = asyncio.get_running_loop()
        with self._guard:
            if self._lock is None or self._loop is not loop:
                self._lock = asyncio.Lock()
                self._loop = loop
            return self._lock


class OAuthRefreshTokenSource:
    """
    Resolves access tokens by running the RFC 6749 refresh-token grant against an OAuth token endpoint.

    Given a stored refresh token plus client credentials, it exchanges them for an access token and caches it in
    process until shortly before expiry. If the identity provider rotates the refresh token on exchange, the new value
    is kept for the lifetime of the process and surfaced through the optional `on_rotate` callback so it can be
    persisted.

    This source is **single-identity**: one refresh token per instance, and its in-process cache is not shared across
    processes. In a multi-replica deployment each replica keeps its own cache, so for providers that rotate (issue
    single-use) refresh tokens the replicas can invalidate one another's token unless rotations are persisted to a
    shared store via `on_rotate` and a single owner drives the refresh.

    Choose this source for a single fixed identity backed by a refresh grant. For a long-lived, non-expiring token
    use `OAuthStaticTokenSource`; for multi-replica or multi-user backends use `OAuthTokenExchangeSource`.
    """

    requires_subject_token: Literal[False] = False

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
        :param scopes: The OAuth scopes to request, joined with `scope_delimiter`. Scope *values* are
            provider-specific (consult your identity provider's documentation).
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
        self._async_lock = _PerLoopAsyncLock()
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
        """
        Return a cached access token, or run the refresh-token grant to obtain a fresh one.

        :returns: A valid bearer access token.
        """
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
        async with self._async_lock.get():
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
    def from_dict(cls, data: dict[str, Any]) -> "OAuthRefreshTokenSource":
        """Deserialize the source from a dictionary."""
        init_params = data["init_parameters"]
        if init_params.get("on_rotate"):
            init_params["on_rotate"] = deserialize_callable(init_params["on_rotate"])
        return default_from_dict(cls, data)


class OAuthClientCredentialsSource:
    """
    Resolves access tokens by running the RFC 6749 client-credentials grant against an OAuth token endpoint.

    The token belongs to the client application itself, not to a user, so this source takes no per-request input.
    Like `OAuthRefreshTokenSource` it is **single-identity**: one token per instance, cached in process until shortly
    before expiry.

    The client authenticates with a `client_secret` or with a client assertion (RFC 7523). The assertion is resolved
    from another token source whenever a new token is requested: a JWT signed by `OAuthPrivateKeyJWTSource`, a fixed
    value from `OAuthStaticTokenSource`, or a custom source.

    Besides feeding an `OAuthTokenResolver`, an instance can be the `client_assertion_source` of
    `OAuthTokenExchangeSource`, for providers where one client's token authenticates another client's exchange.

    ### Usage example
    ```python
    from haystack.utils import Secret
    from haystack_integrations.components.connectors.oauth import OAuthTokenResolver
    from haystack_integrations.utils.oauth import OAuthClientCredentialsSource

    resolver = OAuthTokenResolver(
        token_source=OAuthClientCredentialsSource(
            token_url="https://idp.example.com/oauth2/token",
            client_id="my-service",
            client_secret=Secret.from_env_var("OAUTH_CLIENT_SECRET"),
            scopes=["api.read"],
        ),
    )
    access_token = resolver.run()["access_token"]
    ```
    """

    requires_subject_token: Literal[False] = False

    def __init__(
        self,
        token_url: str,
        client_id: str,
        *,
        client_secret: Secret | None = None,
        client_assertion_source: TokenSource | None = None,
        client_assertion_type: str = DEFAULT_JWT_BEARER_CLIENT_ASSERTION_TYPE,
        scopes: list[str] | None = None,
        scope_delimiter: str = " ",
        extra_token_params: dict[str, str] | None = None,
        expiry_buffer_seconds: int = DEFAULT_EXPIRY_BUFFER_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize the source.

        :param token_url: The OAuth 2.0 token endpoint.
        :param client_id: The OAuth client identifier.
        :param client_secret: The client secret. Mutually exclusive with `client_assertion_source`.
        :param client_assertion_source: A config-only token source whose token is sent as the `client_assertion`
            form parameter to authenticate the client, instead of a client secret. It is resolved each time a new
            access token is requested, so an expiring assertion stays fresh. Mutually exclusive with `client_secret`.
        :param client_assertion_type: The value of the `client_assertion_type` form parameter sent along with the
            client assertion. Defaults to the RFC 7523 JWT bearer type.
        :param scopes: The OAuth scopes to request, joined with `scope_delimiter`. Scope *values* are
            provider-specific (consult your identity provider's documentation).
        :param scope_delimiter: The delimiter used to join scopes. Defaults to a space.
        :param extra_token_params: Additional form parameters included verbatim in every request. Applied last, so
            any key here overrides the corresponding form parameter derived from the other arguments (for example
            `scope`, `client_secret`, or `client_assertion`).
        :param expiry_buffer_seconds: Request a new access token this many seconds before the cached one expires.
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
        _validate_client_auth(client_secret, client_assertion_source)

        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_assertion_source = client_assertion_source
        self.client_assertion_type = client_assertion_type
        self.scopes = list(scopes) if scopes is not None else None
        self.scope_delimiter = scope_delimiter
        self.extra_token_params = dict(extra_token_params) if extra_token_params else None
        self.expiry_buffer_seconds = expiry_buffer_seconds
        self.timeout = timeout

        # Runtime state — never serialized.
        self._sync_lock = Lock()
        self._async_lock = _PerLoopAsyncLock()
        self._cached_token: str | None = None
        self._expires_at: float = 0.0

    def _cached_if_valid(self) -> str | None:
        if self._cached_token and time.monotonic() < self._expires_at - self.expiry_buffer_seconds:
            return self._cached_token
        return None

    def _build_request_data(self, client_assertion: str | None) -> dict[str, str]:
        request_data: dict[str, str] = {"grant_type": "client_credentials", "client_id": self.client_id}
        if self.client_secret is not None:
            client_secret = self.client_secret.resolve_value()
            if client_secret:
                request_data["client_secret"] = client_secret
        if client_assertion:
            request_data["client_assertion_type"] = self.client_assertion_type
            request_data["client_assertion"] = client_assertion
        if self.scopes:
            request_data["scope"] = self.scope_delimiter.join(self.scopes)
        if self.extra_token_params:
            request_data.update(self.extra_token_params)
        return request_data

    def _handle_response(self, response: httpx.Response) -> str:
        access_token, expires_in, _ = _parse_token_response(response)
        self._cached_token = access_token
        self._expires_at = time.monotonic() + expires_in
        logger.debug("Obtained a client-credentials access token via {token_url}.", token_url=self.token_url)
        return access_token

    def resolve(self) -> str:
        """
        Return a cached access token, or run the client-credentials grant to obtain a fresh one.

        :returns: A valid bearer access token.
        :raises TokenRefreshError: If the token endpoint rejects the request or returns no access token, or if the
            `client_assertion_source` fails or returns an empty assertion.
        """
        # Requests for this single-identity source are serialized, so a burst of concurrent callers collapses into
        # one network request.
        with self._sync_lock:
            cached = self._cached_if_valid()
            if cached is not None:
                return cached
            client_assertion = _resolve_client_assertion(self.client_assertion_source)
            response = httpx.post(
                self.token_url,
                data=self._build_request_data(client_assertion),
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            return self._handle_response(response)

    async def resolve_async(self) -> str:
        """Asynchronous counterpart of `resolve`. Use a single instance in either sync or async mode, not both."""
        async with self._async_lock.get():
            cached = self._cached_if_valid()
            if cached is not None:
                return cached
            client_assertion = await _resolve_client_assertion_async(self.client_assertion_source)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data=self._build_request_data(client_assertion),
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
            client_secret=self.client_secret,
            client_assertion_source=self.client_assertion_source.to_dict() if self.client_assertion_source else None,
            client_assertion_type=self.client_assertion_type,
            scopes=self.scopes,
            scope_delimiter=self.scope_delimiter,
            extra_token_params=self.extra_token_params,
            expiry_buffer_seconds=self.expiry_buffer_seconds,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthClientCredentialsSource":
        """Deserialize the source from a dictionary."""
        return default_from_dict(cls, data)


class OAuthTokenExchangeSource:
    """
    Resolves access tokens by exchanging a per-request subject token at an OAuth token endpoint.

    This implements RFC 8693 token exchange (and, via configuration, Microsoft's on-behalf-of flow). Unlike
    `OAuthRefreshTokenSource`, it is **multi-user without any persistent storage**: the per-request `subject_token` (the
    incoming user assertion) *is* the user identity and is exchanged fresh for a downstream token. Resolved tokens
    are cached in memory per subject token (bounded, LRU) until shortly before expiry. Because no per-instance state
    is persisted, it is also the right choice for multi-replica deployments.

    Provider differences are expressed as configuration: `grant_type`, `subject_token_param` (for example
    `assertion` for Microsoft), `scopes`, and `extra_token_params` (for example
    `{"requested_token_use": "on_behalf_of"}`).

    The client authenticates with a `client_secret` or with a client assertion (RFC 7523) taken from
    `client_assertion_source`: a JWT signed by `OAuthPrivateKeyJWTSource` for certificate authentication, or another
    client's token from `OAuthClientCredentialsSource`. For the Microsoft Entra Agent ID on-behalf-of flow, which
    needs the latter, use the preconfigured `EntraAgentOBOTokenSource`.
    """

    requires_subject_token: Literal[True] = True

    def __init__(
        self,
        token_url: str,
        client_id: str,
        *,
        client_secret: Secret | None = None,
        client_assertion_source: TokenSource | None = None,
        client_assertion_type: str = DEFAULT_JWT_BEARER_CLIENT_ASSERTION_TYPE,
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
        :param client_secret: The client secret for confidential clients. Omit it for public clients. Mutually
            exclusive with `client_assertion_source`.
        :param client_assertion_source: A config-only token source whose token is sent as the `client_assertion`
            form parameter to authenticate the client, instead of a client secret. It is resolved for every exchange
            that is not served from the cache, so an expiring assertion stays fresh. Mutually exclusive with
            `client_secret`. A custom source must live in a module that Haystack is allowed to deserialize.
        :param client_assertion_type: The value of the `client_assertion_type` form parameter sent along with the
            client assertion. Defaults to the RFC 7523 JWT bearer type.
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
        :param scopes: The OAuth scopes to request, joined with `scope_delimiter`. Scope *values* are
            provider-specific (consult your identity provider's documentation); only the wire format is standardized
            (RFC 6749 §3.3).
        :param scope_delimiter: The delimiter used to join scopes. Defaults to a space.
        :param extra_token_params: Additional form parameters included verbatim in every request (for example
            `{"requested_token_use": "on_behalf_of"}`). Applied last, so any key here overrides the corresponding
            form parameter derived from the other arguments (for example `grant_type`, `subject_token_type`,
            `requested_token_type`, `scope`, `client_secret`, or `client_assertion`).
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
        _validate_client_auth(client_secret, client_assertion_source)

        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_assertion_source = client_assertion_source
        self.client_assertion_type = client_assertion_type
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
        # `_lock` guards the shared LRU cache. It's held only around cache access, never across the network call or
        # the resolution of the client assertion, so distinct subjects (users) exchange tokens concurrently.
        self._lock = Lock()
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()

    @staticmethod
    def _cache_key(subject_token: str) -> str:
        # The client assertion is deliberately not part of the key: a cached token stays valid when the assertion
        # that authenticated its exchange is replaced by a newer one.
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

    def _build_request_data(self, subject_token: str, client_assertion: str | None = None) -> dict[str, str]:
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
        if client_assertion:
            request_data["client_assertion_type"] = self.client_assertion_type
            request_data["client_assertion"] = client_assertion
        if self.scopes:
            request_data["scope"] = self.scope_delimiter.join(self.scopes)
        if self.extra_token_params:
            request_data.update(self.extra_token_params)
        return request_data

    def resolve(self, subject_token: str) -> str:
        """
        Exchange the per-request `subject_token` for an access token (cached per subject token).

        :param subject_token: The controller-injected per-request subject token (for example an incoming user
            assertion) to exchange for a downstream access token.
        :returns: A valid bearer access token for the given `subject_token`.
        :raises OAuthConfigError: If `subject_token` is empty.
        :raises TokenRefreshError: If the token endpoint rejects the exchange or returns no access token, or if the
            `client_assertion_source` fails or returns an empty assertion.
        """
        if not subject_token:
            msg = "OAuthTokenExchangeSource requires a non-empty subject_token."
            raise OAuthConfigError(msg)
        cache_key = self._cache_key(subject_token)
        with self._lock:
            cached = self._cached_if_valid(cache_key)
        if cached is not None:
            return cached
        client_assertion = _resolve_client_assertion(self.client_assertion_source)
        response = httpx.post(
            self.token_url,
            data=self._build_request_data(subject_token, client_assertion),
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
            msg = "OAuthTokenExchangeSource requires a non-empty subject_token."
            raise OAuthConfigError(msg)
        cache_key = self._cache_key(subject_token)
        with self._lock:
            cached = self._cached_if_valid(cache_key)
        if cached is not None:
            return cached
        client_assertion = await _resolve_client_assertion_async(self.client_assertion_source)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=self._build_request_data(subject_token, client_assertion),
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
            client_assertion_source=self.client_assertion_source.to_dict() if self.client_assertion_source else None,
            client_assertion_type=self.client_assertion_type,
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
    def from_dict(cls, data: dict[str, Any]) -> "OAuthTokenExchangeSource":
        """Deserialize the source from a dictionary."""
        return default_from_dict(cls, data)


class OAuthStaticTokenSource:
    """
    Returns a configured long-lived access token as-is.

    Suitable for providers that issue non-expiring tokens (for example Slack or Notion), where no refresh flow is
    needed and the token is managed out of band. If the provider issues short-lived tokens that must be refreshed,
    use `OAuthRefreshTokenSource` instead. It takes no per-request input.
    """

    requires_subject_token: Literal[False] = False

    def __init__(self, token: Secret) -> None:
        """
        Initialize the source.

        :param token: The long-lived access token to return.
        """
        self.token = token

    def resolve(self) -> str:
        """
        Return the configured token.

        :returns: The configured long-lived access token.
        """
        try:
            value = self.token.resolve_value()
        except ValueError as error:
            # A strict `EnvVarSecret` raises ValueError when its env var is unset; surface it as our error type.
            msg = "OAuthStaticTokenSource could not resolve its token."
            raise TokenRefreshError(msg) from error
        if not value:
            msg = "OAuthStaticTokenSource has no token value."
            raise TokenRefreshError(msg)
        return value

    async def resolve_async(self) -> str:
        """Asynchronous counterpart of `resolve`."""
        return self.resolve()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source to a dictionary."""
        return default_to_dict(self, token=self.token.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthStaticTokenSource":
        """Deserialize the source from a dictionary."""
        return default_from_dict(cls, data)


class OAuthPrivateKeyJWTSource:
    """
    Signs a short-lived JWT client assertion (RFC 7523, also known as `private_key_jwt`) with a private key.

    The result is not an access token: it proves the client's identity to a token endpoint. Pass an instance as the
    `client_assertion_source` of `OAuthClientCredentialsSource` or `OAuthTokenExchangeSource` to authenticate with a
    key pair or a certificate instead of a client secret.

    A new assertion is signed on every call, so it is never stale. The private key must be an unencrypted PEM key;
    passphrase-protected keys and PFX files are not supported.

    Requires the `jwt` extra: `pip install "oauth-haystack[jwt]"`.

    ### Usage example
    ```python
    from haystack.utils import Secret
    from haystack_integrations.utils.oauth import OAuthClientCredentialsSource, OAuthPrivateKeyJWTSource

    token_url = "https://idp.example.com/oauth2/token"
    source = OAuthClientCredentialsSource(
        token_url=token_url,
        client_id="my-service",
        client_assertion_source=OAuthPrivateKeyJWTSource(
            client_id="my-service",
            audience=token_url,
            private_key=Secret.from_env_var("OAUTH_PRIVATE_KEY"),
        ),
        scopes=["api.read"],
    )
    access_token = source.resolve()
    ```
    """

    requires_subject_token: Literal[False] = False

    def __init__(
        self,
        client_id: str,
        audience: str,
        *,
        private_key: Secret = Secret.from_env_var("OAUTH_PRIVATE_KEY"),
        certificate: Secret | None = None,
        algorithm: str = "RS256",
        key_id: str | None = None,
        lifetime_seconds: int = DEFAULT_CLIENT_ASSERTION_LIFETIME_SECONDS,
    ) -> None:
        """
        Initialize the source.

        :param client_id: The OAuth client identifier, used as the `iss` and `sub` claims of the assertion.
        :param audience: The `aud` claim: the identifier the identity provider expects for itself, usually its token
            endpoint URL.
        :param private_key: The PEM-encoded, unencrypted private key that signs the assertion. Defaults to the value
            of the `OAUTH_PRIVATE_KEY` environment variable.
        :param certificate: The PEM-encoded X.509 certificate that belongs to the private key. When set, its SHA-256
            thumbprint is sent in the `x5t#S256` header (RFC 7515), which providers that register certificates
            rather than bare public keys use to find the key. If the PEM holds a chain, the first certificate is
            used.
        :param algorithm: The JWT signing algorithm, for example `RS256`, `PS256`, or `ES256`. It must match the key
            type and be accepted by the identity provider.
        :param key_id: The `kid` header value, for providers that select the verification key by key ID.
        :param lifetime_seconds: How long each assertion is valid. Keep it short; identity providers commonly reject
            assertions that are valid for more than a few minutes.
        :raises ImportError: If the `jwt` extra is not installed.
        :raises OAuthConfigError: If the configuration is invalid.
        """
        jwt_import.check()
        if not client_id:
            msg = "client_id must be a non-empty string."
            raise OAuthConfigError(msg)
        if not audience:
            msg = "audience must be a non-empty string."
            raise OAuthConfigError(msg)
        if algorithm == "none" or algorithm not in jwt.algorithms.get_default_algorithms():
            msg = f"Unsupported signing algorithm '{algorithm}'."
            raise OAuthConfigError(msg)
        if lifetime_seconds <= 0:
            msg = "lifetime_seconds must be positive."
            raise OAuthConfigError(msg)

        self.client_id = client_id
        self.audience = audience
        self.private_key = private_key
        self.certificate = certificate
        self.algorithm = algorithm
        self.key_id = key_id
        self.lifetime_seconds = lifetime_seconds

    @staticmethod
    def _certificate_thumbprint(certificate_pem: str) -> str:
        """Return the `x5t#S256` value: the base64url-encoded SHA-256 digest of the certificate's DER encoding."""
        match = _PEM_CERTIFICATE.search(certificate_pem)
        if match is None:
            msg = "The certificate is not a PEM-encoded X.509 certificate."
            raise TokenRefreshError(msg)
        try:
            der = base64.b64decode(match.group(1))
        except binascii.Error as error:
            msg = "The certificate is not a valid PEM-encoded X.509 certificate."
            raise TokenRefreshError(msg) from error
        return base64.urlsafe_b64encode(hashlib.sha256(der).digest()).rstrip(b"=").decode("ascii")

    def resolve(self) -> str:
        """
        Sign and return a new client assertion.

        :returns: A signed JWT client assertion.
        :raises TokenRefreshError: If the private key or certificate cannot be resolved or parsed, or if the key
            cannot sign with the configured algorithm.
        """
        try:
            private_key = self.private_key.resolve_value()
            certificate = self.certificate.resolve_value() if self.certificate is not None else None
        except ValueError as error:
            # A strict `EnvVarSecret` raises ValueError when its env var is unset; surface it as our error type.
            msg = "OAuthPrivateKeyJWTSource could not resolve its private key or certificate."
            raise TokenRefreshError(msg) from error
        if not private_key:
            msg = "OAuthPrivateKeyJWTSource has no private key value."
            raise TokenRefreshError(msg)

        headers: dict[str, str] = {}
        if self.key_id:
            headers["kid"] = self.key_id
        if certificate:
            headers["x5t#S256"] = self._certificate_thumbprint(certificate)
        now = int(time.time())
        claims: dict[str, str | int] = {
            "iss": self.client_id,
            "sub": self.client_id,
            "aud": self.audience,
            "jti": str(uuid.uuid4()),
            "iat": now,
            "nbf": now,
            "exp": now + self.lifetime_seconds,
        }
        try:
            return jwt.encode(claims, private_key, algorithm=self.algorithm, headers=headers)
        except (jwt.PyJWTError, ValueError, TypeError) as error:
            msg = f"OAuthPrivateKeyJWTSource could not sign the client assertion with algorithm '{self.algorithm}'."
            raise TokenRefreshError(msg) from error

    async def resolve_async(self) -> str:
        """Asynchronous counterpart of `resolve`."""
        return self.resolve()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source to a dictionary."""
        return default_to_dict(
            self,
            client_id=self.client_id,
            audience=self.audience,
            private_key=self.private_key,
            certificate=self.certificate,
            algorithm=self.algorithm,
            key_id=self.key_id,
            lifetime_seconds=self.lifetime_seconds,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthPrivateKeyJWTSource":
        """Deserialize the source from a dictionary."""
        return default_from_dict(cls, data)
