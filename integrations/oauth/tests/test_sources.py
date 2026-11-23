# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any, Literal
from unittest.mock import Mock, patch

import httpx
import jwt
import pytest
from haystack.utils import Secret

from haystack_integrations.utils.oauth import (
    OAuthClientCredentialsSource,
    OAuthConfigError,
    OAuthPrivateKeyJWTSource,
    OAuthRefreshTokenSource,
    OAuthStaticTokenSource,
    OAuthTokenExchangeSource,
    TokenRefreshError,
    sources,
)

SOURCES_MODULE = "haystack_integrations.utils.oauth.sources"
TOKEN_URL = "https://idp.example.com/oauth2/token"
JWT_BEARER_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"


def _json(status=200, **payload):
    return httpx.Response(status, json=payload)


def _slow_post(access_token: str, expires_in: int = 3600):
    """A patched `post` that yields to the loop, so a second caller reaches the lock while the first holds it."""

    async def _post(*_args, **_kwargs):
        await asyncio.sleep(0)
        return _json(access_token=access_token, expires_in=expires_in)

    return _post


class _FakeAssertionSource:
    """Config-only TokenSource that returns a fixed client assertion and counts how often it is asked."""

    requires_subject_token: Literal[False] = False

    def __init__(self, assertion: str = "assertion-1") -> None:
        self.assertion = assertion
        self.calls = 0

    def resolve(self) -> str:
        self.calls += 1
        return self.assertion

    async def resolve_async(self) -> str:
        return self.resolve()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "tests.test_sources._FakeAssertionSource", "init_parameters": {}}

    @classmethod
    def from_dict(cls, _data: dict[str, Any]) -> "_FakeAssertionSource":
        return cls()


# Client-authentication settings that both sources with a `client_assertion_source` must reject.
INVALID_CLIENT_AUTH = [
    pytest.param(
        {"client_secret": Secret.from_token("secret-1"), "client_assertion_source": _FakeAssertionSource()},
        id="secret-and-assertion-source",
    ),
    pytest.param(
        {"client_assertion_source": OAuthTokenExchangeSource(token_url=TOKEN_URL, client_id="other")},
        id="source-that-needs-a-subject-token",
    ),
    pytest.param({"client_assertion_source": object()}, id="not-a-token-source"),
]


def _source_with_unset_secret(source_cls: type, secret_param: str) -> Any:
    """Build a source whose `secret_param` is a strict env var secret pointing at the unset `MISSING_SECRET`."""
    params: dict[str, Any] = {"token_url": TOKEN_URL, "client_id": "client-1"}
    if source_cls is OAuthRefreshTokenSource:
        params["refresh_token"] = Secret.from_token("rt-1")
    params[secret_param] = Secret.from_env_var("MISSING_SECRET")
    return source_cls(**params)


# A strict `EnvVarSecret` raises ValueError when its env var is unset; every source must surface it as
# TokenRefreshError, before any request is sent.
UNSET_STRICT_SECRETS = [
    pytest.param(OAuthRefreshTokenSource, "refresh_token", None, id="refresh-refresh_token"),
    pytest.param(OAuthRefreshTokenSource, "client_secret", None, id="refresh-client_secret"),
    pytest.param(OAuthClientCredentialsSource, "client_secret", None, id="client-credentials-client_secret"),
    pytest.param(OAuthTokenExchangeSource, "client_secret", "user-jwt", id="exchange-client_secret"),
]


@pytest.mark.parametrize(("source_cls", "secret_param", "subject_token"), UNSET_STRICT_SECRETS)
def test_unset_strict_secret_raises_token_refresh_error(monkeypatch, source_cls, secret_param, subject_token):
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    source = _source_with_unset_secret(source_cls, secret_param)
    args = [subject_token] if subject_token else []
    with patch("httpx.post") as mock_post, pytest.raises(TokenRefreshError, match=secret_param):
        source.resolve(*args)
    mock_post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_cls", "secret_param", "subject_token"), UNSET_STRICT_SECRETS)
async def test_unset_strict_secret_raises_token_refresh_error_async(
    monkeypatch, source_cls, secret_param, subject_token
):
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    source = _source_with_unset_secret(source_cls, secret_param)
    args = [subject_token] if subject_token else []
    with patch("httpx.AsyncClient.post") as mock_post, pytest.raises(TokenRefreshError, match=secret_param):
        await source.resolve_async(*args)
    mock_post.assert_not_called()


class TestOAuthRefreshTokenSource:
    def _source(self, **kwargs):
        params = {
            "token_url": TOKEN_URL,
            "client_id": "client-1",
            "refresh_token": Secret.from_token("rt-1"),
            "client_secret": Secret.from_token("secret-1"),
        }
        params.update(kwargs)
        return OAuthRefreshTokenSource(**params)

    def test_resolve_success_and_request_shape(self):
        source = self._source(scopes=["a", "b"])
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            token = source.resolve()
        assert token == "acc-1"
        sent = mock_post.call_args.kwargs["data"]
        assert sent["grant_type"] == "refresh_token"
        assert sent["client_id"] == "client-1"
        assert sent["refresh_token"] == "rt-1"
        assert sent["client_secret"] == "secret-1"
        assert sent["scope"] == "a b"

    def test_scope_delimiter(self):
        source = self._source(scopes=["a", "b"], scope_delimiter=",")
        with patch("httpx.post", return_value=_json(access_token="x", expires_in=3600)) as mock_post:
            source.resolve()
        assert mock_post.call_args.kwargs["data"]["scope"] == "a,b"

    def test_public_client_omits_secret(self):
        source = self._source(client_secret=None)
        with patch("httpx.post", return_value=_json(access_token="x", expires_in=3600)) as mock_post:
            source.resolve()
        assert "client_secret" not in mock_post.call_args.kwargs["data"]

    def test_unset_secret_env_var_omits_secret(self, monkeypatch):
        monkeypatch.delenv("MISSING_SECRET", raising=False)
        source = self._source(client_secret=Secret.from_env_var("MISSING_SECRET", strict=False))
        with patch("httpx.post", return_value=_json(access_token="x", expires_in=3600)) as mock_post:
            source.resolve()
        assert "client_secret" not in mock_post.call_args.kwargs["data"]

    def test_caches_token(self):
        source = self._source()
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve()
            source.resolve()
        assert mock_post.call_count == 1

    def test_refreshes_within_expiry_buffer(self):
        source = self._source(expiry_buffer_seconds=3600)
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=10)) as mock_post:
            source.resolve()
            source.resolve()
        assert mock_post.call_count == 2

    def test_rotation_updates_and_calls_on_rotate(self):
        rotated = []
        source = self._source(expiry_buffer_seconds=3600, on_rotate=rotated.append)
        responses = [
            _json(access_token="acc-1", expires_in=10, refresh_token="rt-2"),
            _json(access_token="acc-2", expires_in=10, refresh_token="rt-3"),
        ]
        with patch("httpx.post", side_effect=responses) as mock_post:
            source.resolve()
            source.resolve()
            second_sent = mock_post.call_args.kwargs["data"]
        assert rotated == ["rt-2", "rt-3"]
        # the second call must use the rotated refresh token from the first response
        assert second_sent["refresh_token"] == "rt-2"

    def test_error_raises(self):
        source = self._source()
        with patch("httpx.post", return_value=httpx.Response(400, text="invalid_grant")):
            with pytest.raises(TokenRefreshError):
                source.resolve()

    def test_response_without_access_token_raises(self):
        source = self._source()
        with patch("httpx.post", return_value=_json(token_type="Bearer", expires_in=3600)):
            with pytest.raises(TokenRefreshError):
                source.resolve()

    def test_empty_token_url_raises(self):
        with pytest.raises(OAuthConfigError):
            OAuthRefreshTokenSource(token_url="", client_id="c", refresh_token=Secret.from_token("rt"))

    def test_refresh_token_defaults_to_env_var(self, monkeypatch):
        monkeypatch.setenv("OAUTH_REFRESH_TOKEN", "env-rt")
        source = OAuthRefreshTokenSource(token_url=TOKEN_URL, client_id="client-1")
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve()
        assert mock_post.call_args.kwargs["data"]["refresh_token"] == "env-rt"

    def test_non_numeric_expires_in_falls_back_to_default_ttl(self):
        source = self._source()
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in="bogus")) as mock_post:
            assert source.resolve() == "acc-1"
            # Served from cache on the second call => a sane fallback TTL was applied, not 0.
            assert source.resolve() == "acc-1"
        assert mock_post.call_count == 1

    def test_round_trip(self, monkeypatch):
        monkeypatch.setenv("MS_REFRESH_TOKEN", "rt")
        source = OAuthRefreshTokenSource(
            token_url=TOKEN_URL,
            client_id="client-1",
            refresh_token=Secret.from_env_var("MS_REFRESH_TOKEN"),
            scopes=["a"],
        )
        data = source.to_dict()
        assert data["type"] == f"{SOURCES_MODULE}.OAuthRefreshTokenSource"
        restored = OAuthRefreshTokenSource.from_dict(source.to_dict())
        assert restored.to_dict() == data

    @pytest.mark.asyncio
    async def test_resolve_async_success_and_caches(self):
        source = self._source()
        with patch("httpx.AsyncClient.post", return_value=_json(access_token="acc-async", expires_in=3600)) as mp:
            token = await source.resolve_async()
            await source.resolve_async()
        assert token == "acc-async"
        assert mp.call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_resolve_async_refreshes_once(self):
        """The refresh lock exists to collapse a burst of concurrent callers into a single network refresh."""
        source = self._source()
        with patch("httpx.AsyncClient.post", side_effect=_slow_post("acc-async")) as mp:
            tokens = await asyncio.gather(*(source.resolve_async() for _ in range(4)))
        assert tokens == ["acc-async"] * 4
        assert mp.call_count == 1

    def test_resolve_async_works_in_a_second_event_loop(self):
        """
        A source outlives the loop that first used it: one `asyncio.run` per request is a common deployment.

        An `asyncio.Lock` binds to the loop that first awaits it *under contention* and rejects every other one, so
        a lock built once and cached on the instance made the second loop's first contended refresh raise
        `RuntimeError: ... is bound to a different event loop`. Both halves of this test must contend for the lock,
        otherwise the lock is never bound and the bug does not reproduce.
        """
        source = self._source()

        async def two_concurrent_callers() -> list[str]:
            # expires_in=0 keeps the cache from serving the second round, so both rounds really refresh.
            with patch("httpx.AsyncClient.post", side_effect=_slow_post("acc-async", expires_in=0)):
                return list(await asyncio.gather(source.resolve_async(), source.resolve_async()))

        assert asyncio.run(two_concurrent_callers()) == ["acc-async"] * 2
        # Same source, brand new event loop.
        assert asyncio.run(two_concurrent_callers()) == ["acc-async"] * 2


class TestOAuthClientCredentialsSource:
    def _source(self, **kwargs):
        params = {
            "token_url": TOKEN_URL,
            "client_id": "client-1",
            "client_secret": Secret.from_token("secret-1"),
        }
        params.update(kwargs)
        return OAuthClientCredentialsSource(**params)

    def test_resolve_success_and_request_shape(self):
        source = self._source(scopes=["a", "b"], extra_token_params={"fmi_path": "agent-1"})
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            token = source.resolve()
        assert token == "acc-1"
        assert mock_post.call_args.kwargs["data"] == {
            "grant_type": "client_credentials",
            "client_id": "client-1",
            "client_secret": "secret-1",
            "scope": "a b",
            "fmi_path": "agent-1",
        }

    def test_client_assertion_replaces_secret(self):
        source = self._source(client_secret=None, client_assertion_source=_FakeAssertionSource("assertion-1"))
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve()
        assert mock_post.call_args.kwargs["data"] == {
            "grant_type": "client_credentials",
            "client_id": "client-1",
            "client_assertion_type": JWT_BEARER_ASSERTION_TYPE,
            "client_assertion": "assertion-1",
        }

    def test_custom_client_assertion_type(self):
        source = self._source(
            client_secret=None, client_assertion_source=_FakeAssertionSource(), client_assertion_type="urn:custom"
        )
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve()
        assert mock_post.call_args.kwargs["data"]["client_assertion_type"] == "urn:custom"

    def test_without_client_auth_sends_no_credentials(self):
        source = self._source(client_secret=None)
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve()
        assert mock_post.call_args.kwargs["data"] == {"grant_type": "client_credentials", "client_id": "client-1"}

    @pytest.mark.parametrize("client_auth", INVALID_CLIENT_AUTH)
    def test_invalid_client_auth_raises(self, client_auth):
        with pytest.raises(OAuthConfigError):
            OAuthClientCredentialsSource(token_url=TOKEN_URL, client_id="client-1", **client_auth)

    @pytest.mark.parametrize("kwargs", [{"token_url": ""}, {"client_id": ""}, {"expiry_buffer_seconds": -1}])
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(OAuthConfigError):
            self._source(**kwargs)

    def test_unset_secret_env_var_omits_secret(self, monkeypatch):
        monkeypatch.delenv("MISSING_SECRET", raising=False)
        source = self._source(client_secret=Secret.from_env_var("MISSING_SECRET", strict=False))
        with patch("httpx.post", return_value=_json(access_token="x", expires_in=3600)) as mock_post:
            source.resolve()
        assert "client_secret" not in mock_post.call_args.kwargs["data"]

    def test_caches_token_and_client_assertion_is_resolved_once(self):
        assertion_source = _FakeAssertionSource()
        source = self._source(client_secret=None, client_assertion_source=assertion_source)
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve()
            source.resolve()
        assert mock_post.call_count == 1
        assert assertion_source.calls == 1

    def test_requests_again_within_expiry_buffer(self):
        source = self._source(expiry_buffer_seconds=3600)
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=10)) as mock_post:
            source.resolve()
            source.resolve()
        assert mock_post.call_count == 2

    def test_error_raises(self):
        with patch("httpx.post", return_value=httpx.Response(401, text="invalid_client")):
            with pytest.raises(TokenRefreshError):
                self._source().resolve()

    def test_response_without_access_token_raises(self):
        with patch("httpx.post", return_value=_json(token_type="Bearer", expires_in=3600)):
            with pytest.raises(TokenRefreshError):
                self._source().resolve()

    def test_empty_client_assertion_raises_before_the_request(self):
        source = self._source(client_secret=None, client_assertion_source=_FakeAssertionSource(""))
        with patch("httpx.post") as mock_post:
            with pytest.raises(TokenRefreshError):
                source.resolve()
        mock_post.assert_not_called()

    def test_round_trip_with_client_secret(self, monkeypatch):
        monkeypatch.setenv("CLIENT_SECRET", "secret-1")
        source = self._source(
            client_secret=Secret.from_env_var("CLIENT_SECRET"),
            scopes=["a"],
            extra_token_params={"fmi_path": "agent-1"},
            expiry_buffer_seconds=60,
            timeout=5.0,
        )
        data = source.to_dict()
        assert data["type"] == f"{SOURCES_MODULE}.OAuthClientCredentialsSource"
        restored = OAuthClientCredentialsSource.from_dict(source.to_dict())
        assert restored.client_secret.resolve_value() == "secret-1"
        assert restored.to_dict() == data

    def test_round_trip_with_client_assertion_source(self, monkeypatch):
        monkeypatch.setenv("CLIENT_ASSERTION", "assertion-1")
        source = self._source(
            client_secret=None,
            client_assertion_source=OAuthStaticTokenSource(token=Secret.from_env_var("CLIENT_ASSERTION")),
            client_assertion_type="urn:custom",
        )
        data = source.to_dict()
        assert data["init_parameters"]["client_assertion_source"]["type"] == f"{SOURCES_MODULE}.OAuthStaticTokenSource"
        restored = OAuthClientCredentialsSource.from_dict(source.to_dict())
        assert isinstance(restored.client_assertion_source, OAuthStaticTokenSource)
        assert restored.client_assertion_source.resolve() == "assertion-1"
        assert restored.to_dict() == data

    @pytest.mark.asyncio
    async def test_resolve_async_success_and_caches(self):
        assertion_source = _FakeAssertionSource("assertion-1")
        source = self._source(client_secret=None, client_assertion_source=assertion_source)
        with patch("httpx.AsyncClient.post", return_value=_json(access_token="acc-async", expires_in=3600)) as mp:
            token = await source.resolve_async()
            await source.resolve_async()
        assert token == "acc-async"
        assert mp.call_count == 1
        assert mp.call_args.kwargs["data"]["client_assertion"] == "assertion-1"
        assert assertion_source.calls == 1

    @pytest.mark.asyncio
    async def test_empty_client_assertion_async_raises(self):
        source = self._source(client_secret=None, client_assertion_source=_FakeAssertionSource(""))
        with pytest.raises(TokenRefreshError):
            await source.resolve_async()

    @pytest.mark.asyncio
    async def test_concurrent_resolve_async_requests_once(self):
        source = self._source()
        with patch("httpx.AsyncClient.post", side_effect=_slow_post("acc-async")) as mp:
            tokens = await asyncio.gather(*(source.resolve_async() for _ in range(4)))
        assert tokens == ["acc-async"] * 4
        assert mp.call_count == 1

    def test_resolve_async_works_in_a_second_event_loop(self):
        # Same contention scenario as the refresh source: the per-loop lock must be rebuilt for a new event loop.
        source = self._source()

        async def two_concurrent_callers() -> list[str]:
            with patch("httpx.AsyncClient.post", side_effect=_slow_post("acc-async", expires_in=0)):
                return list(await asyncio.gather(source.resolve_async(), source.resolve_async()))

        assert asyncio.run(two_concurrent_callers()) == ["acc-async"] * 2
        assert asyncio.run(two_concurrent_callers()) == ["acc-async"] * 2


class TestOAuthTokenExchangeSource:
    def _source(self, **kwargs):
        params = {
            "token_url": TOKEN_URL,
            "client_id": "client-1",
            "client_secret": Secret.from_token("secret-1"),
        }
        params.update(kwargs)
        return OAuthTokenExchangeSource(**params)

    def test_resolve_success_and_request_shape(self):
        source = self._source(
            grant_type="urn:ietf:params:oauth:grant-type:jwt-bearer",
            subject_token_param="assertion",
            scopes=["Files.Read.All"],
            extra_token_params={"requested_token_use": "on_behalf_of"},
        )
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            token = source.resolve(subject_token="user-jwt")
        assert token == "acc-1"
        sent = mock_post.call_args.kwargs["data"]
        assert sent["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
        assert sent["assertion"] == "user-jwt"
        assert sent["client_id"] == "client-1"
        assert sent["client_secret"] == "secret-1"
        assert sent["scope"] == "Files.Read.All"
        assert sent["requested_token_use"] == "on_behalf_of"

    def test_token_type_params_omitted_by_default(self):
        source = self._source()
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve(subject_token="user-jwt")
        sent = mock_post.call_args.kwargs["data"]
        assert "subject_token_type" not in sent
        assert "requested_token_type" not in sent

    def test_token_type_params_emitted_when_set(self):
        source = self._source(
            subject_token_type="urn:ietf:params:oauth:token-type:access_token",
            requested_token_type="urn:ietf:params:oauth:token-type:access_token",
        )
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve(subject_token="user-jwt")
        sent = mock_post.call_args.kwargs["data"]
        assert sent["subject_token_type"] == "urn:ietf:params:oauth:token-type:access_token"
        assert sent["requested_token_type"] == "urn:ietf:params:oauth:token-type:access_token"

    def test_empty_subject_token_raises(self):
        with pytest.raises(OAuthConfigError):
            self._source().resolve(subject_token="")

    def test_caches_per_subject_token(self):
        source = self._source()
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve(subject_token="jwt")
            source.resolve(subject_token="jwt")
        assert mock_post.call_count == 1

    def test_distinct_subjects_not_shared(self):
        source = self._source()
        responses = [_json(access_token="a", expires_in=3600), _json(access_token="b", expires_in=3600)]
        with patch("httpx.post", side_effect=responses) as mock_post:
            assert source.resolve(subject_token="jwt-a") == "a"
            assert source.resolve(subject_token="jwt-b") == "b"
        assert mock_post.call_count == 2

    def test_cache_eviction_lru(self):
        source = self._source(cache_max_size=1)
        responses = [_json(access_token=t, expires_in=3600) for t in ("a", "b", "a2")]
        with patch("httpx.post", side_effect=responses) as mock_post:
            source.resolve(subject_token="u-a")  # cache: [a]
            source.resolve(subject_token="u-b")  # evicts a, cache: [b]
            again = source.resolve(subject_token="u-a")  # miss -> refetch
        assert again == "a2"
        assert mock_post.call_count == 3

    def test_error_raises(self):
        with patch("httpx.post", return_value=httpx.Response(400, text="invalid_grant")):
            with pytest.raises(TokenRefreshError):
                self._source().resolve(subject_token="jwt")

    def test_round_trip(self):
        source = OAuthTokenExchangeSource(
            token_url=TOKEN_URL,
            client_id="client-1",
            subject_token_param="assertion",
            subject_token_type="urn:ietf:params:oauth:token-type:jwt",
            requested_token_type="urn:ietf:params:oauth:token-type:access_token",
            scopes=["a"],
            extra_token_params={"requested_token_use": "on_behalf_of"},
        )
        data = source.to_dict()
        assert data["type"] == f"{SOURCES_MODULE}.OAuthTokenExchangeSource"
        assert data["init_parameters"]["subject_token_type"] == "urn:ietf:params:oauth:token-type:jwt"
        assert data["init_parameters"]["requested_token_type"] == "urn:ietf:params:oauth:token-type:access_token"
        restored = OAuthTokenExchangeSource.from_dict(source.to_dict())
        assert restored.to_dict() == data

    @pytest.mark.asyncio
    async def test_resolve_async_success_and_caches(self):
        source = self._source()
        with patch("httpx.AsyncClient.post", return_value=_json(access_token="acc-async", expires_in=3600)) as mp:
            token = await source.resolve_async(subject_token="user-jwt")
            await source.resolve_async(subject_token="user-jwt")
        assert token == "acc-async"
        assert mp.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_subject_token_async_raises(self):
        with pytest.raises(OAuthConfigError):
            await self._source().resolve_async(subject_token="")

    def test_requires_subject_token_flag(self):
        assert self._source().requires_subject_token is True

    def test_client_assertion_replaces_secret(self):
        source = self._source(client_secret=None, client_assertion_source=_FakeAssertionSource("assertion-1"))
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)) as mock_post:
            source.resolve(subject_token="user-jwt")
        assert mock_post.call_args.kwargs["data"] == {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": "user-jwt",
            "client_id": "client-1",
            "client_assertion_type": JWT_BEARER_ASSERTION_TYPE,
            "client_assertion": "assertion-1",
        }

    def test_client_assertion_is_resolved_only_for_uncached_exchanges(self):
        assertion_source = _FakeAssertionSource()
        source = self._source(client_secret=None, client_assertion_source=assertion_source)
        with patch("httpx.post", return_value=_json(access_token="acc-1", expires_in=3600)):
            source.resolve(subject_token="user-a")
            source.resolve(subject_token="user-a")  # cache hit
            source.resolve(subject_token="user-b")
        assert assertion_source.calls == 2

    def _source_authenticated_by_another_client(self, **nested_kwargs):
        nested = OAuthClientCredentialsSource(
            token_url=TOKEN_URL, client_id="parent", client_secret=Secret.from_token("parent-secret"), **nested_kwargs
        )
        return self._source(client_secret=None, client_assertion_source=nested)

    def test_nested_client_token_is_reused_across_users(self):
        source = self._source_authenticated_by_another_client()
        responses = [_json(access_token=t, expires_in=3600) for t in ("T1", "a", "b")]
        with patch("httpx.post", side_effect=responses) as mock_post:
            assert source.resolve(subject_token="user-a") == "a"
            assert source.resolve(subject_token="user-b") == "b"
        nested_request, exchange_a, exchange_b = (call.kwargs["data"] for call in mock_post.call_args_list)
        assert nested_request["grant_type"] == "client_credentials"
        assert exchange_a["client_assertion"] == exchange_b["client_assertion"] == "T1"

    def test_nested_client_token_is_requested_again_after_it_expires(self):
        # The nested token lives 10s, inside its 3600s buffer, so it is stale by the second exchange.
        source = self._source_authenticated_by_another_client(expiry_buffer_seconds=3600)
        responses = [
            _json(access_token="T1-old", expires_in=10),
            _json(access_token="a", expires_in=3600),
            _json(access_token="T1-new", expires_in=10),
            _json(access_token="b", expires_in=3600),
        ]
        with patch("httpx.post", side_effect=responses) as mock_post:
            source.resolve(subject_token="user-a")
            source.resolve(subject_token="user-b")
        assert mock_post.call_args_list[1].kwargs["data"]["client_assertion"] == "T1-old"
        assert mock_post.call_args_list[3].kwargs["data"]["client_assertion"] == "T1-new"

    def test_nested_client_error_propagates_before_the_exchange(self):
        source = self._source_authenticated_by_another_client()
        with patch("httpx.post", return_value=httpx.Response(401, text="invalid_client")) as mock_post:
            with pytest.raises(TokenRefreshError, match="invalid_client"):
                source.resolve(subject_token="user-a")
        assert mock_post.call_count == 1

    def test_empty_client_assertion_raises_before_the_exchange(self):
        source = self._source(client_secret=None, client_assertion_source=_FakeAssertionSource(""))
        with patch("httpx.post") as mock_post:
            with pytest.raises(TokenRefreshError):
                source.resolve(subject_token="user-a")
        mock_post.assert_not_called()

    @pytest.mark.parametrize("client_auth", INVALID_CLIENT_AUTH)
    def test_invalid_client_auth_raises(self, client_auth):
        with pytest.raises(OAuthConfigError):
            OAuthTokenExchangeSource(token_url=TOKEN_URL, client_id="client-1", **client_auth)

    def test_round_trip_with_client_assertion_source(self, monkeypatch):
        monkeypatch.setenv("PARENT_SECRET", "parent-secret")
        source = self._source(
            client_secret=None,
            client_assertion_source=OAuthClientCredentialsSource(
                token_url=TOKEN_URL,
                client_id="parent",
                client_secret=Secret.from_env_var("PARENT_SECRET"),
                extra_token_params={"fmi_path": "client-1"},
            ),
            client_assertion_type="urn:custom",
        )
        data = source.to_dict()
        nested = data["init_parameters"]["client_assertion_source"]
        assert nested["type"] == f"{SOURCES_MODULE}.OAuthClientCredentialsSource"
        restored = OAuthTokenExchangeSource.from_dict(source.to_dict())
        assert isinstance(restored.client_assertion_source, OAuthClientCredentialsSource)
        assert restored.client_assertion_source.client_secret.resolve_value() == "parent-secret"
        assert restored.to_dict() == data

    def test_from_dict_without_client_assertion_keys(self):
        # Dictionaries serialized before client-assertion authentication existed must keep loading.
        data = self._source(client_secret=None).to_dict()
        del data["init_parameters"]["client_assertion_source"]
        del data["init_parameters"]["client_assertion_type"]
        restored = OAuthTokenExchangeSource.from_dict(data)
        assert restored.client_assertion_source is None
        assert restored.client_assertion_type == JWT_BEARER_ASSERTION_TYPE

    @pytest.mark.asyncio
    async def test_resolve_async_with_nested_client_token(self):
        source = self._source_authenticated_by_another_client()
        responses = [_json(access_token="T1", expires_in=3600), _json(access_token="a", expires_in=3600)]
        with patch("httpx.AsyncClient.post", side_effect=responses) as mp:
            assert await source.resolve_async(subject_token="user-a") == "a"
        nested_request, exchange = (call.kwargs["data"] for call in mp.call_args_list)
        assert nested_request["grant_type"] == "client_credentials"
        assert exchange["client_assertion"] == "T1"
        assert exchange["client_assertion_type"] == JWT_BEARER_ASSERTION_TYPE

    @pytest.mark.asyncio
    async def test_empty_client_assertion_async_raises_before_the_exchange(self):
        source = self._source(client_secret=None, client_assertion_source=_FakeAssertionSource(""))
        with patch("httpx.AsyncClient.post") as mp:
            with pytest.raises(TokenRefreshError):
                await source.resolve_async(subject_token="user-a")
        mp.assert_not_called()


class TestOAuthStaticTokenSource:
    def test_resolve_returns_token(self):
        source = OAuthStaticTokenSource(token=Secret.from_token("pat-123"))
        assert source.resolve() == "pat-123"

    def test_empty_token_raises(self, monkeypatch):
        monkeypatch.delenv("EMPTY_TOKEN", raising=False)
        source = OAuthStaticTokenSource(token=Secret.from_env_var("EMPTY_TOKEN", strict=False))
        with pytest.raises(TokenRefreshError):
            source.resolve()

    def test_unresolvable_strict_token_raises_token_refresh_error(self, monkeypatch):
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        # A strict EnvVarSecret would raise ValueError on resolve; it must surface as TokenRefreshError.
        source = OAuthStaticTokenSource(token=Secret.from_env_var("MISSING_TOKEN"))
        with pytest.raises(TokenRefreshError):
            source.resolve()

    @pytest.mark.asyncio
    async def test_resolve_async_returns_token(self):
        source = OAuthStaticTokenSource(token=Secret.from_token("pat-123"))
        assert await source.resolve_async() == "pat-123"

    def test_round_trip(self, monkeypatch):
        monkeypatch.setenv("SLACK_TOKEN", "pat-123")
        source = OAuthStaticTokenSource(token=Secret.from_env_var("SLACK_TOKEN"))
        data = source.to_dict()
        assert data["type"] == f"{SOURCES_MODULE}.OAuthStaticTokenSource"
        restored = OAuthStaticTokenSource.from_dict(source.to_dict())
        assert restored.to_dict() == data


class TestOAuthPrivateKeyJWTSource:
    def _source(self, signing_material, **kwargs):
        params = {
            "client_id": "client-1",
            "audience": TOKEN_URL,
            "private_key": Secret.from_token(signing_material.private_key_pem),
        }
        params.update(kwargs)
        return OAuthPrivateKeyJWTSource(**params)

    @pytest.mark.parametrize("algorithm", ["RS256", "PS256"])
    def test_signs_assertion_with_client_claims(self, signing_material, algorithm):
        source = self._source(signing_material, algorithm=algorithm, lifetime_seconds=120)
        assertion = source.resolve()
        # `jwt.decode` verifies the signature, the audience, and that the assertion is currently valid.
        claims = jwt.decode(assertion, signing_material.public_key, algorithms=[algorithm], audience=TOKEN_URL)
        assert claims["iss"] == claims["sub"] == "client-1"
        assert claims["iat"] == claims["nbf"]
        assert claims["exp"] - claims["nbf"] == 120
        assert jwt.get_unverified_header(assertion) == {"alg": algorithm, "typ": "JWT"}

    def test_each_assertion_has_a_unique_jti(self, signing_material):
        source = self._source(signing_material)
        jtis = {
            jwt.decode(source.resolve(), signing_material.public_key, algorithms=["RS256"], audience=TOKEN_URL)["jti"]
            for _ in range(3)
        }
        assert len(jtis) == 3

    def test_certificate_and_key_id_headers(self, signing_material):
        source = self._source(
            signing_material, certificate=Secret.from_token(signing_material.certificate_pem), key_id="key-1"
        )
        header = jwt.get_unverified_header(source.resolve())
        assert header["x5t#S256"] == signing_material.thumbprint
        assert header["kid"] == "key-1"

    def test_certificate_chain_uses_the_first_certificate(self, signing_material):
        chain = signing_material.certificate_pem + "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n"
        source = self._source(signing_material, certificate=Secret.from_token(chain))
        assert jwt.get_unverified_header(source.resolve())["x5t#S256"] == signing_material.thumbprint

    @pytest.mark.parametrize(
        "certificate",
        ["not a certificate", "-----BEGIN CERTIFICATE-----\nnot-base64!\n-----END CERTIFICATE-----"],
    )
    def test_invalid_certificate_raises(self, signing_material, certificate):
        source = self._source(signing_material, certificate=Secret.from_token(certificate))
        with pytest.raises(TokenRefreshError):
            source.resolve()

    def test_unparseable_private_key_raises(self, signing_material):
        source = self._source(signing_material, private_key=Secret.from_token("not a key"))
        with pytest.raises(TokenRefreshError):
            source.resolve()

    def test_key_of_the_wrong_type_for_the_algorithm_raises(self, signing_material):
        source = self._source(signing_material, algorithm="ES256")  # an RSA key cannot sign with an EC algorithm
        with pytest.raises(TokenRefreshError):
            source.resolve()

    def test_private_key_defaults_to_env_var(self, signing_material, monkeypatch):
        monkeypatch.setenv("OAUTH_PRIVATE_KEY", signing_material.private_key_pem)
        assertion = OAuthPrivateKeyJWTSource(client_id="client-1", audience=TOKEN_URL).resolve()
        jwt.decode(assertion, signing_material.public_key, algorithms=["RS256"], audience=TOKEN_URL)

    @pytest.mark.parametrize("strict", [True, False])
    def test_unset_private_key_env_var_raises_token_refresh_error(self, monkeypatch, strict):
        # A strict secret raises on resolve and a lenient one resolves to None; both must surface as our error.
        monkeypatch.delenv("OAUTH_PRIVATE_KEY", raising=False)
        source = OAuthPrivateKeyJWTSource(
            client_id="client-1",
            audience=TOKEN_URL,
            private_key=Secret.from_env_var("OAUTH_PRIVATE_KEY", strict=strict),
        )
        with pytest.raises(TokenRefreshError):
            source.resolve()

    def test_unset_strict_certificate_env_var_raises_token_refresh_error(self, signing_material, monkeypatch):
        monkeypatch.delenv("MISSING_CERTIFICATE", raising=False)
        source = self._source(signing_material, certificate=Secret.from_env_var("MISSING_CERTIFICATE"))
        with pytest.raises(TokenRefreshError, match="certificate"):
            source.resolve()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"client_id": ""},
            {"audience": ""},
            {"algorithm": "none"},
            {"algorithm": "XX256"},
            {"lifetime_seconds": 0},
        ],
    )
    def test_invalid_config_raises(self, signing_material, kwargs):
        with pytest.raises(OAuthConfigError):
            self._source(signing_material, **kwargs)

    def test_missing_jwt_extra_raises_at_init(self, signing_material, monkeypatch):
        monkeypatch.setattr(sources.jwt_import, "check", Mock(side_effect=ImportError("install the jwt extra")))
        with pytest.raises(ImportError, match="install the jwt extra"):
            self._source(signing_material)

    def test_round_trip(self, signing_material, monkeypatch):
        monkeypatch.setenv("PRIVATE_KEY", signing_material.private_key_pem)
        monkeypatch.setenv("CERTIFICATE", signing_material.certificate_pem)
        source = OAuthPrivateKeyJWTSource(
            client_id="client-1",
            audience=TOKEN_URL,
            private_key=Secret.from_env_var("PRIVATE_KEY"),
            certificate=Secret.from_env_var("CERTIFICATE"),
            algorithm="PS256",
            key_id="key-1",
            lifetime_seconds=120,
        )
        data = source.to_dict()
        assert data["type"] == f"{SOURCES_MODULE}.OAuthPrivateKeyJWTSource"
        restored = OAuthPrivateKeyJWTSource.from_dict(source.to_dict())
        assert restored.to_dict() == data
        header = jwt.get_unverified_header(restored.resolve())
        assert header["alg"] == "PS256"
        assert header["x5t#S256"] == signing_material.thumbprint

    @pytest.mark.asyncio
    async def test_resolve_async_signs_assertion(self, signing_material):
        assertion = await self._source(signing_material).resolve_async()
        jwt.decode(assertion, signing_material.public_key, algorithms=["RS256"], audience=TOKEN_URL)
