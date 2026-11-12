# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import time
from unittest.mock import patch

import anyio
import pytest
from haystack.utils import Secret
from mcp.client.auth import OAuthFlowError, OAuthTokenError
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from haystack_integrations.tools.mcp import (
    InMemoryTokenStorage,
    MCPConnectionError,
    OAuthConfig,
    SSEServerInfo,
    StreamableHttpServerInfo,
)
from haystack_integrations.tools.mcp.compatibility_layer import http_lib
from haystack_integrations.tools.mcp.oauth import _FailFastAuth

SDK_SUPPORTS_ISSUER = "issuer" in inspect.signature(ClientCredentialsOAuthProvider.__init__).parameters


@pytest.fixture
def client_secret(monkeypatch):
    monkeypatch.setenv("TEST_MCP_CLIENT_SECRET", "super-secret")
    return Secret.from_env_var("TEST_MCP_CLIENT_SECRET")


@pytest.fixture
def oauth_config(client_secret):
    return OAuthConfig(client_id="my-client", client_secret=client_secret, scopes=["mcp:read", "mcp:write"])


class _FakeProvider:
    """Stand-in for an SDK OAuth provider: adds a header, or fails with an OAuth error."""

    requires_request_body = False
    requires_response_body = True

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def async_auth_flow(self, request):
        if self.error is not None:
            raise self.error
        request.headers["Authorization"] = "Bearer fake-token"
        yield request


class TestInMemoryTokenStorage:
    @pytest.mark.asyncio
    async def test_empty_by_default(self):
        storage = InMemoryTokenStorage()
        assert await storage.get_tokens() is None
        assert await storage.get_client_info() is None

    @pytest.mark.asyncio
    async def test_stores_tokens_and_client_info(self):
        storage = InMemoryTokenStorage()
        tokens = OAuthToken(access_token="access-token", token_type="Bearer", expires_in=3600)
        client_info = OAuthClientInformationFull(client_id="my-client", redirect_uris=None)

        await storage.set_tokens(tokens)
        await storage.set_client_info(client_info)

        assert await storage.get_tokens() == tokens
        assert await storage.get_client_info() == client_info


class TestOAuthConfig:
    def test_defaults(self, client_secret):
        config = OAuthConfig(client_id="my-client", client_secret=client_secret)

        assert config.issuer is None
        assert config.scopes is None
        assert config.token_endpoint_auth_method == "client_secret_basic"
        assert config.grant_type == "client_credentials"
        assert isinstance(config.token_storage, InMemoryTokenStorage)

    def test_each_config_gets_its_own_storage(self, client_secret):
        first = OAuthConfig(client_id="my-client", client_secret=client_secret)
        second = OAuthConfig(client_id="my-client", client_secret=client_secret)

        assert first.token_storage is not second.token_storage
        # token_storage is not part of equality, so configs with the same settings compare equal
        assert first == second

    def test_unsupported_grant_type(self, client_secret):
        with pytest.raises(ValueError, match="Unsupported grant_type"):
            OAuthConfig(client_id="my-client", client_secret=client_secret, grant_type="authorization_code")

    def test_unsupported_token_endpoint_auth_method(self, client_secret):
        with pytest.raises(ValueError, match="Unsupported token_endpoint_auth_method"):
            OAuthConfig(client_id="my-client", client_secret=client_secret, token_endpoint_auth_method="none")

    def test_to_dict(self, client_secret):
        config = OAuthConfig(
            client_id="my-client",
            client_secret=client_secret,
            issuer="https://auth.example.com",
            scopes=["mcp:read"],
            token_endpoint_auth_method="client_secret_post",
        )

        assert config.to_dict() == {
            "client_id": "my-client",
            "client_secret": {"type": "env_var", "env_vars": ["TEST_MCP_CLIENT_SECRET"], "strict": True},
            "issuer": "https://auth.example.com",
            "scopes": ["mcp:read"],
            "token_endpoint_auth_method": "client_secret_post",
            "grant_type": "client_credentials",
        }

    def test_to_dict_refuses_token_secret(self):
        config = OAuthConfig(client_id="my-client", client_secret=Secret.from_token("super-secret"))

        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            config.to_dict()

    def test_from_dict(self, oauth_config):
        restored = OAuthConfig.from_dict(oauth_config.to_dict())

        assert restored == oauth_config
        assert restored.client_secret.resolve_value() == "super-secret"
        assert isinstance(restored.token_storage, InMemoryTokenStorage)
        assert restored.token_storage is not oauth_config.token_storage

    def test_create_auth(self, oauth_config):
        provider = oauth_config.create_auth(server_url="https://mcp.example.com/mcp")

        assert isinstance(provider, ClientCredentialsOAuthProvider)
        assert provider.context.server_url == "https://mcp.example.com/mcp"
        assert provider.context.storage is oauth_config.token_storage
        assert provider.context.client_metadata.scope == "mcp:read mcp:write"
        assert provider.context.client_metadata.token_endpoint_auth_method == "client_secret_basic"

    def test_create_auth_with_unresolved_secret(self, monkeypatch):
        monkeypatch.delenv("UNSET_MCP_CLIENT_SECRET", raising=False)
        config = OAuthConfig(
            client_id="my-client", client_secret=Secret.from_env_var("UNSET_MCP_CLIENT_SECRET", strict=False)
        )

        with pytest.raises(ValueError, match="client_secret resolved to an empty value"):
            config.create_auth(server_url="https://mcp.example.com/mcp")

    @pytest.mark.skipif(not SDK_SUPPORTS_ISSUER, reason="The installed mcp SDK does not support `issuer`")
    def test_create_auth_with_issuer(self, client_secret):
        config = OAuthConfig(client_id="my-client", client_secret=client_secret, issuer="https://auth.example.com")

        provider = config.create_auth(server_url="https://mcp.example.com/mcp")

        assert provider._issuer == "https://auth.example.com"

    def test_create_auth_with_issuer_on_sdk_without_issuer_support(self, client_secret):
        class ProviderWithoutIssuer:
            def __init__(self, server_url, storage, client_id, client_secret, token_endpoint_auth_method, scope):
                pass

        config = OAuthConfig(client_id="my-client", client_secret=client_secret, issuer="https://auth.example.com")

        with patch(
            "haystack_integrations.tools.mcp.compatibility_layer.ClientCredentialsOAuthProvider", ProviderWithoutIssuer
        ):
            with pytest.raises(ValueError, match=r"OAuthConfig\.issuer needs"):
                config.create_auth(server_url="https://mcp.example.com/mcp")


class TestFailFastAuth:
    def test_copies_body_requirements_from_provider(self):
        auth = _FailFastAuth(_FakeProvider())

        assert auth.requires_request_body is False
        assert auth.requires_response_body is True

    @pytest.mark.asyncio
    async def test_delegates_to_provider(self):
        auth = _FailFastAuth(_FakeProvider())
        request = http_lib.Request("GET", "https://mcp.example.com/mcp")

        flow = auth.async_auth_flow(request)
        outgoing = await flow.__anext__()
        with pytest.raises(StopAsyncIteration):
            await flow.asend(http_lib.Response(200, request=outgoing))

        assert outgoing.headers["Authorization"] == "Bearer fake-token"
        assert auth.error is None

    @pytest.mark.asyncio
    async def test_oauth_error_is_recorded_and_cancels_scope(self):
        error = OAuthTokenError("Token exchange failed (401): invalid_client")
        auth = _FailFastAuth(_FakeProvider(error=error))
        scope = auth.failure_scope()

        with pytest.raises(OAuthTokenError):
            await auth.async_auth_flow(http_lib.Request("GET", "https://mcp.example.com/mcp")).__anext__()

        assert auth.error is error
        assert scope.cancel_called

    @pytest.mark.asyncio
    async def test_other_errors_do_not_cancel_scope(self):
        auth = _FailFastAuth(_FakeProvider(error=RuntimeError("boom")))
        scope = auth.failure_scope()

        with pytest.raises(RuntimeError):
            await auth.async_auth_flow(http_lib.Request("GET", "https://mcp.example.com/mcp")).__anext__()

        assert auth.error is None
        assert not scope.cancel_called

    @pytest.mark.asyncio
    async def test_failure_scope_resets_previous_error(self):
        auth = _FailFastAuth(_FakeProvider())
        auth.error = OAuthFlowError("previous failure")

        auth.failure_scope()

        assert auth.error is None


@pytest.mark.parametrize(
    ("server_info_class", "url"),
    [(SSEServerInfo, "https://mcp.example.com/sse"), (StreamableHttpServerInfo, "https://mcp.example.com/mcp")],
)
class TestServerInfoWithOAuth:
    def test_oauth_config_defaults_to_none(self, server_info_class, url):
        assert server_info_class(url=url).oauth_config is None

    def test_oauth_config_and_token_are_exclusive(self, server_info_class, url, oauth_config):
        with pytest.raises(ValueError, match="`token` and `oauth_config` can't be used together"):
            server_info_class(url=url, token="static-token", oauth_config=oauth_config)

    @pytest.mark.parametrize("header", ["Authorization", "authorization"])
    def test_oauth_config_and_authorization_header_are_exclusive(self, server_info_class, url, oauth_config, header):
        with pytest.raises(ValueError, match="`Authorization` header can't be used together with `oauth_config`"):
            server_info_class(url=url, headers={header: "Bearer static-token"}, oauth_config=oauth_config)

    def test_oauth_config_with_other_headers(self, server_info_class, url, oauth_config):
        server_info = server_info_class(url=url, headers={"X-Client-ID": "my-client"}, oauth_config=oauth_config)

        assert server_info.headers == {"X-Client-ID": "my-client"}

    def test_serde(self, server_info_class, url, oauth_config):
        server_info = server_info_class(url=url, oauth_config=oauth_config)

        data = server_info.to_dict()
        assert data["oauth_config"] == oauth_config.to_dict()
        assert "super-secret" not in json.dumps(data)

        restored = server_info_class.from_dict(data)
        assert restored == server_info
        assert isinstance(restored.oauth_config, OAuthConfig)
        assert restored.oauth_config.client_secret.resolve_value() == "super-secret"

    def test_from_dict_keeps_header_named_type(self, server_info_class, url):
        server_info = server_info_class(url=url, headers={"type": "haystack_integrations.tools.mcp.OAuthConfig"})

        restored = server_info_class.from_dict(server_info.to_dict())

        assert restored.headers == {"type": "haystack_integrations.tools.mcp.OAuthConfig"}


class TestClientsWithOAuth:
    @pytest.mark.asyncio
    async def test_streamable_http_client_passes_auth_and_reuses_it(self, oauth_config):
        client = StreamableHttpServerInfo(url="https://mcp.example.com/mcp", oauth_config=oauth_config).create_client()
        assert client._auth is None

        with patch(
            "haystack_integrations.tools.mcp.mcp_tool.create_mcp_http_client", side_effect=RuntimeError("stop")
        ) as mock_create_client:
            for _ in range(2):
                with pytest.raises(RuntimeError, match="stop"):
                    await client.connect()

        first_auth, second_auth = (call.kwargs["auth"] for call in mock_create_client.call_args_list)
        assert isinstance(first_auth, _FailFastAuth)
        assert first_auth is second_auth
        assert mock_create_client.call_args.kwargs["headers"] is None

    @pytest.mark.asyncio
    async def test_sse_client_passes_auth(self, oauth_config):
        client = SSEServerInfo(url="https://mcp.example.com/sse", oauth_config=oauth_config).create_client()

        with patch(
            "haystack_integrations.tools.mcp.mcp_tool.sse_client", side_effect=RuntimeError("stop")
        ) as mock_sse_client:
            with pytest.raises(RuntimeError, match="stop"):
                await client.connect()

        assert isinstance(mock_sse_client.call_args.kwargs["auth"], _FailFastAuth)

    @pytest.mark.asyncio
    async def test_no_auth_without_oauth_config(self):
        client = StreamableHttpServerInfo(url="https://mcp.example.com/mcp", token="static-token").create_client()

        with patch(
            "haystack_integrations.tools.mcp.mcp_tool.create_mcp_http_client", side_effect=RuntimeError("stop")
        ) as mock_create_client:
            with pytest.raises(RuntimeError, match="stop"):
                await client.connect()

        assert mock_create_client.call_args.kwargs["auth"] is None
        assert mock_create_client.call_args.kwargs["headers"] == {"Authorization": "Bearer static-token"}

    @pytest.mark.asyncio
    async def test_oauth_failure_aborts_initialization(self, oauth_config):
        client = StreamableHttpServerInfo(url="https://mcp.example.com/mcp", oauth_config=oauth_config).create_client()
        client._auth = _FailFastAuth(_FakeProvider())
        error = OAuthTokenError('Token exchange failed (401): {"error": "invalid_client"}')

        class SessionWithFailingAuth:
            """A session whose initialize request triggers a failed OAuth flow, and never gets a response."""

            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return None

            async def initialize(self):
                client._auth.error = error
                client._auth._scope.cancel()
                await anyio.sleep(30)

        start = time.monotonic()
        with patch("haystack_integrations.tools.mcp.mcp_tool.ClientSession", SessionWithFailingAuth):
            with pytest.raises(MCPConnectionError, match=r"OAuth authentication failed.*invalid_client") as exc_info:
                await client._initialize_session_with_transport((None, None), "HTTP server at test")

        assert exc_info.value.__cause__ is error
        assert time.monotonic() - start < 5
