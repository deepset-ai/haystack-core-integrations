# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import time
from unittest.mock import MagicMock, patch

import pytest
from haystack.utils import Secret

from haystack_integrations.tools.mcp import (
    InMemoryTokenStorage,
    OAuthConfig,
    SSEServerInfo,
    StreamableHttpServerInfo,
    TokenData,
)
from haystack_integrations.tools.mcp.oauth import (
    TokenStorage,
    _fetch_client_credentials_token,
    ensure_oauth,
)

# ---------------------------------------------------------------------------
# TokenData
# ---------------------------------------------------------------------------


class TestTokenData:
    def test_valid_when_no_expiry(self):
        token = TokenData(access_token="tok", expires_at=0.0)
        assert token.is_valid()

    def test_valid_before_expiry(self):
        token = TokenData(access_token="tok", expires_at=time.time() + 3600)
        assert token.is_valid()

    def test_invalid_after_expiry(self):
        token = TokenData(access_token="tok", expires_at=time.time() - 1)
        assert not token.is_valid()

    def test_invalid_within_leeway(self):
        token = TokenData(access_token="tok", expires_at=time.time() + 10)
        assert not token.is_valid(leeway=30)

    def test_valid_outside_leeway(self):
        token = TokenData(access_token="tok", expires_at=time.time() + 120)
        assert token.is_valid(leeway=30)


# ---------------------------------------------------------------------------
# InMemoryTokenStorage
# ---------------------------------------------------------------------------


class TestInMemoryTokenStorage:
    def test_store_and_get(self):
        storage = InMemoryTokenStorage()
        token = TokenData(access_token="abc", expires_at=time.time() + 3600)
        storage.store_token("key1", token)
        assert storage.get_token("key1") is token

    def test_get_missing_returns_none(self):
        storage = InMemoryTokenStorage()
        assert storage.get_token("missing") is None

    def test_expired_token_auto_removed(self):
        storage = InMemoryTokenStorage()
        expired = TokenData(access_token="old", expires_at=time.time() - 1)
        storage.store_token("key", expired)
        assert storage.get_token("key") is None

    def test_clear_token(self):
        storage = InMemoryTokenStorage()
        token = TokenData(access_token="abc", expires_at=time.time() + 3600)
        storage.store_token("key", token)
        storage.clear_token("key")
        assert storage.get_token("key") is None

    def test_clear_missing_is_safe(self):
        storage = InMemoryTokenStorage()
        storage.clear_token("does-not-exist")  # should not raise

    def test_is_subclass_of_token_storage(self):
        assert issubclass(InMemoryTokenStorage, TokenStorage)


# ---------------------------------------------------------------------------
# OAuthConfig serialization
# ---------------------------------------------------------------------------


class TestOAuthConfigSerde:
    def test_to_dict_plain_secret(self):
        cfg = OAuthConfig(
            client_id="my-client",
            client_secret="plain-secret",
            token_endpoint="https://auth.example.com/token",
            scopes=["read", "write"],
        )
        d = cfg.to_dict()
        assert d["client_id"] == "my-client"
        assert d["client_secret"] == "plain-secret"
        assert d["token_endpoint"] == "https://auth.example.com/token"
        assert d["scopes"] == ["read", "write"]
        assert "token_storage" not in d  # not serialized

    def test_to_dict_secret_object(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "s3cr3t")
        cfg = OAuthConfig(
            client_id="id",
            client_secret=Secret.from_env_var("MY_SECRET"),
            token_endpoint="https://auth.example.com/token",
        )
        d = cfg.to_dict()
        # Should serialize as a Secret dict, not the resolved value
        assert isinstance(d["client_secret"], dict)
        assert d["client_secret"]["type"] == "env_var"

    def test_roundtrip_plain(self):
        cfg = OAuthConfig(
            client_id="cid",
            client_secret="cs",
            token_endpoint="https://tok.example.com",
            scopes=["a"],
        )
        restored = OAuthConfig.from_dict(cfg.to_dict())
        assert restored.client_id == cfg.client_id
        assert restored.client_secret == cfg.client_secret
        assert restored.token_endpoint == cfg.token_endpoint
        assert restored.scopes == cfg.scopes

    def test_roundtrip_secret(self, monkeypatch):
        monkeypatch.setenv("TEST_CS", "mypassword")
        cfg = OAuthConfig(
            client_id="cid",
            client_secret=Secret.from_env_var("TEST_CS"),
            token_endpoint="https://tok.example.com",
        )
        restored = OAuthConfig.from_dict(cfg.to_dict())
        assert isinstance(restored.client_secret, Secret)
        assert restored.client_secret.resolve_value() == "mypassword"

    def test_no_client_secret(self):
        cfg = OAuthConfig(client_id="cid", token_endpoint="https://tok.example.com")
        d = cfg.to_dict()
        assert d["client_secret"] is None
        restored = OAuthConfig.from_dict(d)
        assert restored.client_secret is None


# ---------------------------------------------------------------------------
# _fetch_client_credentials_token
# ---------------------------------------------------------------------------


class TestFetchClientCredentialsToken:
    def _mock_token_response(self, access_token="test-token", expires_in=3600, scope=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        body = {"access_token": access_token, "token_type": "Bearer", "expires_in": expires_in}
        if scope:
            body["scope"] = scope
        mock_resp.json.return_value = body
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_basic_client_credentials(self):
        cfg = OAuthConfig(
            client_id="cid",
            client_secret="cs",
            token_endpoint="https://auth.example.com/token",
        )
        with patch("haystack_integrations.tools.mcp.oauth.httpx.post") as mock_post:
            mock_post.return_value = self._mock_token_response()
            token = _fetch_client_credentials_token(cfg)

        assert token.access_token == "test-token"
        assert token.token_type == "Bearer"
        assert token.is_valid()

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["data"]
        assert payload["grant_type"] == "client_credentials"
        assert payload["client_id"] == "cid"
        assert payload["client_secret"] == "cs"

    def test_secret_client_secret_resolved(self, monkeypatch):
        monkeypatch.setenv("OAUTH_SECRET", "resolved-secret")
        cfg = OAuthConfig(
            client_id="cid",
            client_secret=Secret.from_env_var("OAUTH_SECRET"),
            token_endpoint="https://auth.example.com/token",
        )
        with patch("haystack_integrations.tools.mcp.oauth.httpx.post") as mock_post:
            mock_post.return_value = self._mock_token_response()
            _fetch_client_credentials_token(cfg)

        payload = mock_post.call_args.kwargs["data"]
        assert payload["client_secret"] == "resolved-secret"

    def test_scopes_joined(self):
        cfg = OAuthConfig(
            client_id="cid",
            token_endpoint="https://auth.example.com/token",
            scopes=["read", "write"],
        )
        with patch("haystack_integrations.tools.mcp.oauth.httpx.post") as mock_post:
            mock_post.return_value = self._mock_token_response()
            _fetch_client_credentials_token(cfg)

        payload = mock_post.call_args.kwargs["data"]
        assert payload["scope"] == "read write"

    def test_no_client_secret_not_sent(self):
        cfg = OAuthConfig(client_id="cid", token_endpoint="https://auth.example.com/token")
        with patch("haystack_integrations.tools.mcp.oauth.httpx.post") as mock_post:
            mock_post.return_value = self._mock_token_response()
            _fetch_client_credentials_token(cfg)

        payload = mock_post.call_args.kwargs["data"]
        assert "client_secret" not in payload

    def test_raises_without_endpoint(self):
        cfg = OAuthConfig(client_id="cid")
        with pytest.raises(ValueError, match="token_endpoint or authorization_server_url"):
            _fetch_client_credentials_token(cfg)

    def test_expires_at_set(self):
        cfg = OAuthConfig(client_id="cid", token_endpoint="https://auth.example.com/token")
        with patch("haystack_integrations.tools.mcp.oauth.httpx.post") as mock_post:
            mock_post.return_value = self._mock_token_response(expires_in=600)
            before = time.time()
            token = _fetch_client_credentials_token(cfg)
            after = time.time()

        assert before + 600 <= token.expires_at <= after + 600

    def test_no_expires_in_sets_zero(self):
        cfg = OAuthConfig(client_id="cid", token_endpoint="https://auth.example.com/token")
        resp = MagicMock()
        resp.json.return_value = {"access_token": "tok", "token_type": "Bearer"}
        resp.raise_for_status = MagicMock()
        with patch("haystack_integrations.tools.mcp.oauth.httpx.post", return_value=resp):
            token = _fetch_client_credentials_token(cfg)
        assert token.expires_at == 0.0

    def test_discovery_used_when_no_token_endpoint(self):
        cfg = OAuthConfig(
            client_id="cid",
            authorization_server_url="https://auth.example.com",
        )
        discovered = "https://auth.example.com/token"
        with (
            patch(
                "haystack_integrations.tools.mcp.oauth._discover_token_endpoint",
                return_value=discovered,
            ) as mock_disc,
            patch("haystack_integrations.tools.mcp.oauth.httpx.post") as mock_post,
        ):
            mock_post.return_value = self._mock_token_response()
            _fetch_client_credentials_token(cfg)

        mock_disc.assert_called_once_with("https://auth.example.com")


# ---------------------------------------------------------------------------
# ensure_oauth
# ---------------------------------------------------------------------------


class TestEnsureOAuth:
    def _make_server_info(self, with_oauth=True):
        if with_oauth:
            oauth = OAuthConfig(
                client_id="cid",
                client_secret="cs",
                token_endpoint="https://auth.example.com/token",
            )
        else:
            oauth = None
        info = MagicMock()
        info.oauth_config = oauth
        info.headers = None
        return info

    def test_noop_without_oauth_config(self):
        server_info = self._make_server_info(with_oauth=False)
        ensure_oauth(server_info)  # should not raise
        # headers unchanged
        assert server_info.headers is None

    def test_injects_authorization_header(self):
        server_info = self._make_server_info()
        mock_token = TokenData(access_token="fresh-token", token_type="Bearer", expires_at=time.time() + 3600)

        with patch("haystack_integrations.tools.mcp.oauth._fetch_client_credentials_token", return_value=mock_token):
            ensure_oauth(server_info)

        assert server_info.headers["Authorization"] == "Bearer fresh-token"

    def test_uses_cached_token(self):
        server_info = self._make_server_info()
        cached = TokenData(access_token="cached-token", expires_at=time.time() + 3600)
        server_info.oauth_config.token_storage.store_token("cid@https://auth.example.com/token", cached)

        with patch("haystack_integrations.tools.mcp.oauth._fetch_client_credentials_token") as mock_fetch:
            ensure_oauth(server_info)
            mock_fetch.assert_not_called()

        assert server_info.headers["Authorization"] == "Bearer cached-token"

    def test_refetches_expired_token(self):
        server_info = self._make_server_info()
        expired = TokenData(access_token="old", expires_at=time.time() - 1)
        server_info.oauth_config.token_storage.store_token("cid@https://auth.example.com/token", expired)
        new_token = TokenData(access_token="new-token", expires_at=time.time() + 3600)

        with patch("haystack_integrations.tools.mcp.oauth._fetch_client_credentials_token", return_value=new_token):
            ensure_oauth(server_info)

        assert server_info.headers["Authorization"] == "Bearer new-token"

    def test_creates_headers_dict_if_none(self):
        server_info = self._make_server_info()
        server_info.headers = None
        mock_token = TokenData(access_token="tok", expires_at=time.time() + 3600)

        with patch("haystack_integrations.tools.mcp.oauth._fetch_client_credentials_token", return_value=mock_token):
            ensure_oauth(server_info)

        assert isinstance(server_info.headers, dict)
        assert "Authorization" in server_info.headers

    def test_overwrites_existing_authorization_header(self):
        server_info = self._make_server_info()
        server_info.headers = {"Authorization": "Bearer old-static-token", "X-Custom": "val"}
        mock_token = TokenData(access_token="oauth-token", expires_at=time.time() + 3600)

        with patch("haystack_integrations.tools.mcp.oauth._fetch_client_credentials_token", return_value=mock_token):
            ensure_oauth(server_info)

        assert server_info.headers["Authorization"] == "Bearer oauth-token"
        assert server_info.headers["X-Custom"] == "val"


# ---------------------------------------------------------------------------
# SSEServerInfo / StreamableHttpServerInfo integration
# ---------------------------------------------------------------------------


class TestServerInfoOAuthField:
    def test_sse_server_info_accepts_oauth_config(self):
        cfg = OAuthConfig(client_id="cid", token_endpoint="https://auth.example.com/token")
        info = SSEServerInfo(url="http://example.com/sse", oauth_config=cfg)
        assert info.oauth_config is cfg

    def test_streamable_http_server_info_accepts_oauth_config(self):
        cfg = OAuthConfig(client_id="cid", token_endpoint="https://auth.example.com/token")
        info = StreamableHttpServerInfo(url="http://example.com/mcp", oauth_config=cfg)
        assert info.oauth_config is cfg

    def test_oauth_config_defaults_to_none(self):
        info = SSEServerInfo(url="http://example.com/sse")
        assert info.oauth_config is None

    def test_sse_server_info_serde_with_oauth(self):
        cfg = OAuthConfig(
            client_id="cid",
            client_secret="cs",
            token_endpoint="https://auth.example.com/token",
        )
        info = SSEServerInfo(url="http://example.com/sse", oauth_config=cfg)
        d = info.to_dict()
        assert "oauth_config" in d
        assert d["oauth_config"]["client_id"] == "cid"

    def test_streamable_http_server_info_serde_with_oauth(self):
        cfg = OAuthConfig(
            client_id="cid",
            token_endpoint="https://auth.example.com/token",
        )
        info = StreamableHttpServerInfo(url="http://example.com/mcp", oauth_config=cfg)
        d = info.to_dict()
        assert "oauth_config" in d
        assert d["oauth_config"]["client_id"] == "cid"

    def test_sse_server_info_from_dict_with_oauth(self):
        cfg = OAuthConfig(
            client_id="cid",
            client_secret="cs",
            token_endpoint="https://auth.example.com/token",
        )
        info = SSEServerInfo(url="http://example.com/sse", oauth_config=cfg)
        d = info.to_dict()
        restored = SSEServerInfo.from_dict(d)
        assert restored.oauth_config is not None
        assert restored.oauth_config.client_id == "cid"
        assert restored.oauth_config.client_secret == "cs"
        assert restored.oauth_config.token_endpoint == "https://auth.example.com/token"

    def test_server_info_without_oauth_serde_unchanged(self):
        info = SSEServerInfo(url="http://example.com/sse", token="tok")
        d = info.to_dict()
        restored = SSEServerInfo.from_dict(d)
        assert restored.oauth_config is None
        assert restored.token == "tok"
