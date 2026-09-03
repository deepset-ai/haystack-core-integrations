# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Per-request credentials for MCP servers.

The property that matters, and the one a ``Secret`` cannot provide: the credential is read on every
HTTP request rather than once when the client is built. A pipeline that outlives a token's lifetime
therefore keeps working instead of failing until it is redeployed.
"""

import pytest
from haystack.utils import Secret

from haystack_integrations.tools.mcp import (
    ClientCredentialsTokenProvider,
    MCPTokenProvider,
    MCPTool,
    StreamableHttpServerInfo,
    TokenProviderAuth,
)
from haystack_integrations.tools.mcp.compatibility_layer import http_lib

TOKEN_URL = "https://auth.example.com/oauth/token"


class CountingProvider(MCPTokenProvider):
    """Hands out a new token each time it is asked, and counts how often that happens."""

    def __init__(self) -> None:
        self.calls = 0
        self.invalidations = 0

    def token(self) -> str:
        self.calls += 1
        return f"token-{self.calls}"

    def invalidate(self) -> None:
        self.invalidations += 1


class TestTokenProviderAuth:
    """The auth flow attaches a token per request and recovers from a rejected one in place."""

    @staticmethod
    def _client(provider: MCPTokenProvider, handler) -> "http_lib.Client":
        transport = http_lib.MockTransport(handler)
        return http_lib.Client(transport=transport, auth=TokenProviderAuth(provider))

    def test_attaches_a_bearer_token(self) -> None:
        provider = CountingProvider()
        seen = []

        def handler(request):
            seen.append(request.headers.get("Authorization"))
            return http_lib.Response(200)

        self._client(provider, handler).get("https://mcp.example.com/mcp")

        assert seen == ["Bearer token-1"]

    def test_asks_again_for_every_request(self) -> None:
        """A Secret is resolved once; this is the difference that keeps a long run alive."""
        provider = CountingProvider()
        client = self._client(provider, lambda _request: http_lib.Response(200))

        client.get("https://mcp.example.com/mcp")
        client.get("https://mcp.example.com/mcp")
        client.get("https://mcp.example.com/mcp")

        assert provider.calls == 3

    def test_refreshes_and_retries_within_the_rejected_request(self) -> None:
        """A 401 is recovered inside the failing request, so the MCP session is never torn down."""
        provider = CountingProvider()
        seen = []

        def handler(request):
            seen.append(request.headers.get("Authorization"))
            return http_lib.Response(401) if len(seen) == 1 else http_lib.Response(200)

        response = self._client(provider, handler).get("https://mcp.example.com/mcp")

        assert response.status_code == 200
        assert seen == ["Bearer token-1", "Bearer token-2"]
        assert provider.invalidations == 1

    def test_gives_up_after_one_retry(self) -> None:
        """A server that rejects every token is a configuration problem, not something to loop on."""
        provider = CountingProvider()
        attempts = []

        def handler(request):
            attempts.append(request.headers.get("Authorization"))
            return http_lib.Response(401)

        response = self._client(provider, handler).get("https://mcp.example.com/mcp")

        assert response.status_code == 401
        assert len(attempts) == 2


class TestClientCredentialsTokenProvider:
    """The OAuth2 machine-to-machine grant, which is what most self-hosted MCP servers expect."""

    @staticmethod
    def _provider(**kwargs) -> ClientCredentialsTokenProvider:
        defaults = {
            "token_url": TOKEN_URL,
            "client_id": "my-client",
            "client_secret": Secret.from_token("shh"),
        }
        return ClientCredentialsTokenProvider(**{**defaults, **kwargs})

    def test_caches_until_shortly_before_expiry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every HTTP request asks for a token, so fetching each time would add a round trip."""
        calls = []

        def fake_post(url, **_kwargs):
            calls.append(url)
            return http_lib.Response(200, json={"access_token": "at-1", "expires_in": 3600})

        monkeypatch.setattr("haystack_integrations.tools.mcp.mcp_client_credentials.httpx.post", fake_post)
        provider = self._provider()

        assert [provider.token() for _ in range(3)] == ["at-1", "at-1", "at-1"]
        assert len(calls) == 1

    def test_fetches_again_after_invalidation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A token can be rejected long before it was due to expire — revoked, for instance."""
        issued = iter(["at-1", "at-2"])
        monkeypatch.setattr(
            "haystack_integrations.tools.mcp.mcp_client_credentials.httpx.post",
            lambda _url, **_kwargs: http_lib.Response(200, json={"access_token": next(issued), "expires_in": 3600}),
        )
        provider = self._provider()

        first = provider.token()
        provider.invalidate()
        second = provider.token()

        assert (first, second) == ("at-1", "at-2")

    def test_sends_the_client_secret_as_basic_auth_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RFC 6749 requires every authorization server to accept Basic; not all accept the body."""
        captured = {}

        def fake_post(_url, **kwargs):
            captured.update(kwargs)
            return http_lib.Response(200, json={"access_token": "at", "expires_in": 60})

        monkeypatch.setattr("haystack_integrations.tools.mcp.mcp_client_credentials.httpx.post", fake_post)
        self._provider().token()

        assert captured["headers"]["Authorization"].startswith("Basic ")
        assert "client_secret" not in captured["data"]

    def test_can_send_the_client_secret_in_the_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def fake_post(_url, **kwargs):
            captured.update(kwargs)
            return http_lib.Response(200, json={"access_token": "at", "expires_in": 60})

        monkeypatch.setattr("haystack_integrations.tools.mcp.mcp_client_credentials.httpx.post", fake_post)
        self._provider(client_auth_method="client_secret_post").token()

        assert captured["data"]["client_secret"] == "shh"
        assert "Authorization" not in captured["headers"]

    def test_serializes_the_secret_by_reference(self) -> None:
        provider = self._provider(client_secret=Secret.from_env_var("MCP_CLIENT_SECRET"), scope="mcp:read")

        descriptor = provider.to_dict()

        assert descriptor["init_parameters"]["client_secret"] == {
            "type": "env_var",
            "env_vars": ["MCP_CLIENT_SECRET"],
            "strict": True,
        }

    def test_survives_a_serialization_round_trip(self) -> None:
        server_info = StreamableHttpServerInfo(
            url="https://mcp.example.com/mcp",
            token_provider=self._provider(client_secret=Secret.from_env_var("MCP_CLIENT_SECRET")),
        )

        restored = StreamableHttpServerInfo.from_dict(server_info.to_dict())

        assert isinstance(restored.token_provider, ClientCredentialsTokenProvider)
        assert restored.token_provider.token_url == TOKEN_URL


@pytest.mark.integration
class TestAgainstARealServer:
    """End-to-end over real HTTP against a real MCP server.

    This is the check worth having before a release: it proves the provider is consulted by the
    actual transport on every request, not merely that the plumbing type-checks.
    """

    def test_the_provider_is_consulted_for_each_tool_call(self, mcp_calculator_server) -> None:
        port = mcp_calculator_server("streamable-http")
        provider = CountingProvider()

        tool = MCPTool(
            name="add",
            server_info=StreamableHttpServerInfo(url=f"http://127.0.0.1:{port}/mcp", token_provider=provider),
        )

        assert tool.invoke(a=2, b=3)
        calls_after_first = provider.calls
        assert tool.invoke(a=4, b=5)

        # The point of the exercise: a Secret would have been read once, before the first call.
        assert provider.calls > calls_after_first
