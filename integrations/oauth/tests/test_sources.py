# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import httpx
import pytest
from haystack.utils import Secret

from haystack_integrations.utils.oauth import (
    OAuthConfigError,
    RefreshTokenSource,
    StaticTokenSource,
    TokenExchangeSource,
    TokenRefreshError,
)

SOURCES_MODULE = "haystack_integrations.utils.oauth.sources"
TOKEN_URL = "https://idp.example.com/oauth2/token"


def _json(status=200, **payload):
    return httpx.Response(status, json=payload)


class TestRefreshTokenSource:
    def _source(self, **kwargs):
        params = {
            "token_url": TOKEN_URL,
            "client_id": "client-1",
            "refresh_token": Secret.from_token("rt-1"),
            "client_secret": Secret.from_token("secret-1"),
        }
        params.update(kwargs)
        return RefreshTokenSource(**params)

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
            RefreshTokenSource(token_url="", client_id="c", refresh_token=Secret.from_token("rt"))

    def test_refresh_token_defaults_to_env_var(self, monkeypatch):
        monkeypatch.setenv("OAUTH_REFRESH_TOKEN", "env-rt")
        source = RefreshTokenSource(token_url=TOKEN_URL, client_id="client-1")
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
        source = RefreshTokenSource(
            token_url=TOKEN_URL,
            client_id="client-1",
            refresh_token=Secret.from_env_var("MS_REFRESH_TOKEN"),
            scopes=["a"],
        )
        data = source.to_dict()
        assert data["type"] == f"{SOURCES_MODULE}.RefreshTokenSource"
        restored = RefreshTokenSource.from_dict(source.to_dict())
        assert restored.to_dict() == data

    @pytest.mark.asyncio
    async def test_resolve_async_success_and_caches(self):
        source = self._source()
        with patch("httpx.AsyncClient.post", return_value=_json(access_token="acc-async", expires_in=3600)) as mp:
            token = await source.resolve_async()
            await source.resolve_async()
        assert token == "acc-async"
        assert mp.call_count == 1


class TestTokenExchangeSource:
    def _source(self, **kwargs):
        params = {
            "token_url": TOKEN_URL,
            "client_id": "client-1",
            "client_secret": Secret.from_token("secret-1"),
        }
        params.update(kwargs)
        return TokenExchangeSource(**params)

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
        source = TokenExchangeSource(
            token_url=TOKEN_URL,
            client_id="client-1",
            subject_token_param="assertion",
            subject_token_type="urn:ietf:params:oauth:token-type:jwt",
            requested_token_type="urn:ietf:params:oauth:token-type:access_token",
            scopes=["a"],
            extra_token_params={"requested_token_use": "on_behalf_of"},
        )
        data = source.to_dict()
        assert data["type"] == f"{SOURCES_MODULE}.TokenExchangeSource"
        assert data["init_parameters"]["subject_token_type"] == "urn:ietf:params:oauth:token-type:jwt"
        assert data["init_parameters"]["requested_token_type"] == "urn:ietf:params:oauth:token-type:access_token"
        restored = TokenExchangeSource.from_dict(source.to_dict())
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


class TestStaticTokenSource:
    def test_resolve_returns_token(self):
        source = StaticTokenSource(token=Secret.from_token("pat-123"))
        assert source.resolve() == "pat-123"

    def test_empty_token_raises(self, monkeypatch):
        monkeypatch.delenv("EMPTY_TOKEN", raising=False)
        source = StaticTokenSource(token=Secret.from_env_var("EMPTY_TOKEN", strict=False))
        with pytest.raises(TokenRefreshError):
            source.resolve()

    def test_unresolvable_strict_token_raises_token_refresh_error(self, monkeypatch):
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        # A strict EnvVarSecret would raise ValueError on resolve; it must surface as TokenRefreshError.
        source = StaticTokenSource(token=Secret.from_env_var("MISSING_TOKEN"))
        with pytest.raises(TokenRefreshError):
            source.resolve()

    @pytest.mark.asyncio
    async def test_resolve_async_returns_token(self):
        source = StaticTokenSource(token=Secret.from_token("pat-123"))
        assert await source.resolve_async() == "pat-123"

    def test_round_trip(self, monkeypatch):
        monkeypatch.setenv("SLACK_TOKEN", "pat-123")
        source = StaticTokenSource(token=Secret.from_env_var("SLACK_TOKEN"))
        data = source.to_dict()
        assert data["type"] == f"{SOURCES_MODULE}.StaticTokenSource"
        restored = StaticTokenSource.from_dict(source.to_dict())
        assert restored.to_dict() == data
