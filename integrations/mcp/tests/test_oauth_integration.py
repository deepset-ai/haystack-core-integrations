# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end integration tests for OAuth Client Credentials flow.

Starts a real HTTP mock OAuth server on localhost using the stdlib http.server
and verifies the full token-fetch-and-inject path without mocking httpx.
"""

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from haystack.utils import Secret

from haystack_integrations.tools.mcp import InMemoryTokenStorage, OAuthConfig, SSEServerInfo, StreamableHttpServerInfo
from haystack_integrations.tools.mcp.oauth import TokenData, _fetch_client_credentials_token, ensure_oauth

# ---------------------------------------------------------------------------
# Mock OAuth server using stdlib http.server
# ---------------------------------------------------------------------------

VALID_CLIENT_ID = "test-client"
VALID_CLIENT_SECRET = "test-secret"
ISSUED_TOKEN = "mock-access-token-xyz"
SERVER_PORT = 19876

TOKEN_ENDPOINT = f"http://127.0.0.1:{SERVER_PORT}/oauth/token"
NO_EXPIRY_ENDPOINT = f"http://127.0.0.1:{SERVER_PORT}/oauth/no-expiry-token"
DISCOVERY_BASE = f"http://127.0.0.1:{SERVER_PORT}"


class _OAuthHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence request log output  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/.well-known/oauth-authorization-server":
            body = json.dumps({"token_endpoint": TOKEN_ENDPOINT}).encode()
            self._send(200, body)
        else:
            self._send(404, b"{}")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode()
        params = dict(urllib.parse.parse_qsl(raw))

        if self.path == "/oauth/token":
            if params.get("grant_type") != "client_credentials":
                self._send(400, json.dumps({"error": "unsupported_grant_type"}).encode())
                return
            if params.get("client_id") != VALID_CLIENT_ID or params.get("client_secret") != VALID_CLIENT_SECRET:
                self._send(401, json.dumps({"error": "invalid_client"}).encode())
                return
            scope = params.get("scope", "")
            body = json.dumps({
                "access_token": ISSUED_TOKEN,
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": scope,
            }).encode()
            self._send(200, body)

        elif self.path == "/oauth/no-expiry-token":
            if params.get("client_id") != VALID_CLIENT_ID:
                self._send(401, json.dumps({"error": "invalid_client"}).encode())
                return
            body = json.dumps({"access_token": "no-expiry-token", "token_type": "Bearer"}).encode()
            self._send(200, body)

        else:
            self._send(404, b"{}")

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="session", autouse=True)
def mock_oauth_server():
    """Start the mock OAuth server once for the whole test session."""
    server = HTTPServer(("127.0.0.1", SERVER_PORT), _OAuthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Wait until the server is accepting connections
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{SERVER_PORT}/.well-known/oauth-authorization-server", timeout=1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    yield
    server.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchClientCredentialsLive:
    def test_successful_token_fetch(self):
        cfg = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            client_secret=VALID_CLIENT_SECRET,
            token_endpoint=TOKEN_ENDPOINT,
        )
        token = _fetch_client_credentials_token(cfg)
        assert token.access_token == ISSUED_TOKEN
        assert token.token_type == "Bearer"
        assert token.is_valid()
        assert token.expires_at > time.time()

    def test_secret_object_resolved_correctly(self, monkeypatch):
        monkeypatch.setenv("OAUTH_CS", VALID_CLIENT_SECRET)
        cfg = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            client_secret=Secret.from_env_var("OAUTH_CS"),
            token_endpoint=TOKEN_ENDPOINT,
        )
        token = _fetch_client_credentials_token(cfg)
        assert token.access_token == ISSUED_TOKEN

    def test_wrong_credentials_raises(self):
        cfg = OAuthConfig(
            client_id="bad-client",
            client_secret="bad-secret",
            token_endpoint=TOKEN_ENDPOINT,
        )
        with pytest.raises(httpx.HTTPStatusError):
            _fetch_client_credentials_token(cfg)

    def test_scopes_sent_to_server(self):
        cfg = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            client_secret=VALID_CLIENT_SECRET,
            token_endpoint=TOKEN_ENDPOINT,
            scopes=["read", "write"],
        )
        token = _fetch_client_credentials_token(cfg)
        assert token.scope == "read write"

    def test_token_without_expiry_has_zero_expires_at(self):
        cfg = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            token_endpoint=NO_EXPIRY_ENDPOINT,
        )
        token = _fetch_client_credentials_token(cfg)
        assert token.access_token == "no-expiry-token"
        assert token.expires_at == 0.0
        assert token.is_valid()  # 0.0 means "never expires"

    def test_discovery_endpoint_resolves_token_url(self):
        cfg = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            client_secret=VALID_CLIENT_SECRET,
            authorization_server_url=DISCOVERY_BASE,
        )
        token = _fetch_client_credentials_token(cfg)
        assert token.access_token == ISSUED_TOKEN


class TestEnsureOAuthLive:
    def test_token_injected_into_sse_server_info(self):
        oauth = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            client_secret=VALID_CLIENT_SECRET,
            token_endpoint=TOKEN_ENDPOINT,
        )
        server_info = SSEServerInfo(url="http://example.com/sse", oauth_config=oauth)
        ensure_oauth(server_info)
        assert server_info.headers is not None
        assert server_info.headers["Authorization"] == f"Bearer {ISSUED_TOKEN}"

    def test_token_injected_into_streamable_http_server_info(self):
        oauth = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            client_secret=VALID_CLIENT_SECRET,
            token_endpoint=TOKEN_ENDPOINT,
        )
        server_info = StreamableHttpServerInfo(url="http://example.com/mcp", oauth_config=oauth)
        ensure_oauth(server_info)
        assert server_info.headers["Authorization"] == f"Bearer {ISSUED_TOKEN}"

    def test_token_cached_second_call_does_not_hit_server(self):
        storage = InMemoryTokenStorage()
        oauth = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            client_secret=VALID_CLIENT_SECRET,
            token_endpoint=TOKEN_ENDPOINT,
            token_storage=storage,
        )
        server_info = SSEServerInfo(url="http://example.com/sse", oauth_config=oauth)

        # First call — hits the real server and populates the cache
        ensure_oauth(server_info)
        first_header = server_info.headers["Authorization"]
        assert first_header == f"Bearer {ISSUED_TOKEN}"

        # Verify the token landed in the cache
        cache_key = f"{VALID_CLIENT_ID}@{TOKEN_ENDPOINT}"
        cached = storage.get_token(cache_key)
        assert cached is not None
        assert cached.access_token == ISSUED_TOKEN

        # Second call with the same storage — token still valid, no new fetch needed
        server_info2 = SSEServerInfo(url="http://example.com/sse2", oauth_config=oauth)
        ensure_oauth(server_info2)
        assert server_info2.headers["Authorization"] == first_header

    def test_expired_token_refetched_live(self):
        storage = InMemoryTokenStorage()
        oauth = OAuthConfig(
            client_id=VALID_CLIENT_ID,
            client_secret=VALID_CLIENT_SECRET,
            token_endpoint=TOKEN_ENDPOINT,
            token_storage=storage,
        )
        server_info = SSEServerInfo(url="http://example.com/sse", oauth_config=oauth)

        # Plant an expired token in cache
        expired = TokenData(access_token="expired-token", expires_at=time.time() - 1)
        cache_key = f"{oauth.client_id}@{oauth.token_endpoint}"
        storage.store_token(cache_key, expired)

        ensure_oauth(server_info)
        # Should have fetched a fresh token
        assert server_info.headers["Authorization"] == f"Bearer {ISSUED_TOKEN}"

    def test_noop_when_no_oauth_config(self):
        server_info = SSEServerInfo(url="http://example.com/sse")
        ensure_oauth(server_info)  # should not raise
        assert server_info.headers is None
