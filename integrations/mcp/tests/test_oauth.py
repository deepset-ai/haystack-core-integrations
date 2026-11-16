# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import socket
import time
import webbrowser
from unittest.mock import patch
from urllib.parse import urlsplit

import anyio
import pytest
from haystack.utils import Secret
from mcp.client.auth import OAuthClientProvider, OAuthFlowError, OAuthTokenError
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
from haystack_integrations.tools.mcp.oauth import (
    _FailFastAuth,
    _LoopbackAuthorizationFlow,
    _PreRegisteredClientStorage,
    _resolve_redirect_uri,
)

SDK_SUPPORTS_ISSUER = "issuer" in inspect.signature(ClientCredentialsOAuthProvider.__init__).parameters


async def show_url(authorization_url: str) -> None:
    """Module-level `on_authorization_url` handler, so it can be serialized."""


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
            OAuthConfig(client_id="my-client", client_secret=client_secret, grant_type="password")

    @pytest.mark.parametrize("missing", ["client_id", "client_secret"])
    def test_client_credentials_needs_client_id_and_secret(self, client_secret, missing):
        kwargs = {"client_id": "my-client", "client_secret": client_secret, missing: None}

        with pytest.raises(ValueError, match="needs both `client_id` and `client_secret`"):
            OAuthConfig(**kwargs)

    @pytest.mark.parametrize(
        "option", [{"redirect_uri": "http://127.0.0.1:8765/callback"}, {"on_authorization_url": show_url}]
    )
    def test_client_credentials_rejects_authorization_code_options(self, client_secret, option):
        with pytest.raises(ValueError, match="only used by the 'authorization_code' grant"):
            OAuthConfig(client_id="my-client", client_secret=client_secret, **option)

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
            "redirect_uri": None,
            "callback_timeout": 300.0,
            "on_authorization_url": None,
        }

    def test_from_dict_without_authorization_code_fields(self, oauth_config):
        # Configurations serialized before the Authorization Code grant was added
        data = {
            "client_id": "my-client",
            "client_secret": {"type": "env_var", "env_vars": ["TEST_MCP_CLIENT_SECRET"], "strict": True},
            "issuer": None,
            "scopes": ["mcp:read", "mcp:write"],
            "token_endpoint_auth_method": "client_secret_basic",
            "grant_type": "client_credentials",
        }

        assert OAuthConfig.from_dict(data) == oauth_config

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


class TestOAuthConfigAuthorizationCode:
    def test_defaults(self):
        config = OAuthConfig(grant_type="authorization_code")

        assert config.client_id is None
        assert config.client_secret is None
        assert config.redirect_uri is None
        assert config.callback_timeout == 300.0
        assert config.on_authorization_url is None

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"issuer": "https://auth.example.com"}, "`issuer` is only supported"),
            ({"scopes": ["mcp:read"]}, "`scopes` is only supported"),
            ({"client_secret": Secret.from_token("secret")}, "`client_secret` needs the `client_id`"),
            ({"callback_timeout": 0}, "`callback_timeout` must be positive"),
            ({"redirect_uri": "https://127.0.0.1:8765/callback"}, "must be an http URL on a loopback host"),
            ({"redirect_uri": "http://app.example.com/callback"}, "must be an http URL on a loopback host"),
            ({"redirect_uri": "http://127.0.0.1:99999/callback"}, "Invalid port"),
        ],
    )
    def test_invalid_configuration(self, kwargs, error):
        with pytest.raises(ValueError, match=error):
            OAuthConfig(grant_type="authorization_code", **kwargs)

    @pytest.mark.parametrize(
        "redirect_uri", ["http://localhost:8765/callback", "http://127.0.0.1/cb", "http://[::1]:0/callback"]
    )
    def test_loopback_redirect_uris_are_accepted(self, redirect_uri):
        assert OAuthConfig(grant_type="authorization_code", redirect_uri=redirect_uri).redirect_uri == redirect_uri

    def test_to_dict_from_dict(self, client_secret):
        config = OAuthConfig(
            grant_type="authorization_code",
            client_id="my-client",
            client_secret=client_secret,
            token_endpoint_auth_method="client_secret_post",
            redirect_uri="http://localhost:8765/callback",
            callback_timeout=60.0,
            on_authorization_url=show_url,
        )

        data = config.to_dict()
        assert data == {
            "client_id": "my-client",
            "client_secret": {"type": "env_var", "env_vars": ["TEST_MCP_CLIENT_SECRET"], "strict": True},
            "issuer": None,
            "scopes": None,
            "token_endpoint_auth_method": "client_secret_post",
            "grant_type": "authorization_code",
            "redirect_uri": "http://localhost:8765/callback",
            "callback_timeout": 60.0,
            "on_authorization_url": "tests.test_oauth.show_url",
        }

        restored = OAuthConfig.from_dict(json.loads(json.dumps(data)))
        assert restored == config
        assert restored.on_authorization_url is show_url
        assert restored.client_secret.resolve_value() == "super-secret"

    def test_create_auth_registers_a_public_client_dynamically(self):
        config = OAuthConfig(grant_type="authorization_code")

        provider = config.create_auth(server_url="https://mcp.example.com/mcp")

        assert isinstance(provider, OAuthClientProvider)
        assert provider.context.server_url == "https://mcp.example.com/mcp"
        assert provider.context.storage is config.token_storage
        metadata = provider.context.client_metadata
        assert metadata.token_endpoint_auth_method == "none"
        assert metadata.grant_types == ["authorization_code", "refresh_token"]
        (redirect_uri,) = metadata.redirect_uris
        assert redirect_uri.host == "127.0.0.1"
        assert redirect_uri.port not in (None, 0)
        assert redirect_uri.path == "/callback"

    def test_free_port_is_picked_once_per_config(self):
        config = OAuthConfig(grant_type="authorization_code", redirect_uri="http://localhost:0/oauth")

        first = config.create_auth(server_url="https://mcp.example.com/mcp")
        second = config.create_auth(server_url="https://mcp.example.com/mcp")

        assert first.context.client_metadata.redirect_uris == second.context.client_metadata.redirect_uris
        assert str(first.context.client_metadata.redirect_uris[0]).startswith("http://localhost:")

    @pytest.mark.asyncio
    async def test_create_auth_uses_a_pre_registered_client(self, client_secret):
        config = OAuthConfig(
            grant_type="authorization_code",
            client_id="my-client",
            client_secret=client_secret,
            redirect_uri="http://localhost:8765/callback",
        )

        provider = config.create_auth(server_url="https://mcp.example.com/mcp")

        client_info = await provider.context.storage.get_client_info()
        assert client_info.client_id == "my-client"
        assert client_info.client_secret == "super-secret"
        assert client_info.token_endpoint_auth_method == "client_secret_basic"
        assert [str(uri) for uri in client_info.redirect_uris] == ["http://localhost:8765/callback"]
        # Nothing is written to the configured storage until the SDK stores something
        assert await config.token_storage.get_client_info() is None

    @pytest.mark.asyncio
    async def test_pre_registered_public_client(self):
        config = OAuthConfig(
            grant_type="authorization_code", client_id="my-client", redirect_uri="http://localhost:8765/callback"
        )

        provider = config.create_auth(server_url="https://mcp.example.com/mcp")

        client_info = await provider.context.storage.get_client_info()
        assert client_info.client_secret is None
        assert client_info.token_endpoint_auth_method == "none"


class TestPreRegisteredClientStorage:
    @pytest.mark.asyncio
    async def test_stored_client_info_takes_precedence(self):
        storage = InMemoryTokenStorage()
        pre_registered = OAuthClientInformationFull(client_id="pre-registered", redirect_uris=None)
        wrapped = _PreRegisteredClientStorage(storage=storage, client_info=pre_registered)

        assert await wrapped.get_client_info() == pre_registered

        stored = OAuthClientInformationFull(client_id="stored", redirect_uris=None)
        await wrapped.set_client_info(stored)
        assert await wrapped.get_client_info() == stored
        assert await storage.get_client_info() == stored

    @pytest.mark.asyncio
    async def test_tokens_are_delegated(self):
        storage = InMemoryTokenStorage()
        wrapped = _PreRegisteredClientStorage(
            storage=storage, client_info=OAuthClientInformationFull(client_id="pre-registered", redirect_uris=None)
        )
        tokens = OAuthToken(access_token="access-token", token_type="Bearer")

        await wrapped.set_tokens(tokens)

        assert await storage.get_tokens() == tokens
        assert await wrapped.get_tokens() == tokens


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _code_and_state(result) -> tuple[str, str | None]:
    # mcp v2 returns an AuthorizationCodeResult, v1 a (code, state) tuple
    return (result.code, result.state) if hasattr(result, "code") else result


async def _get(url: str):
    async with http_lib.AsyncClient() as client:
        return await client.get(url)


AUTHORIZATION_URL = "https://auth.example.com/authorize?client_id=my-client&state=expected-state"


class TestLoopbackAuthorizationFlow:
    @pytest.fixture
    def redirect_uri(self):
        return f"http://127.0.0.1:{_free_port()}/callback"

    @staticmethod
    def _flow(redirect_uri: str, **kwargs) -> _LoopbackAuthorizationFlow:
        kwargs = {"callback_timeout": 5.0, "on_authorization_url": show_url, **kwargs}
        return _LoopbackAuthorizationFlow(redirect_uri=redirect_uri, **kwargs)

    @staticmethod
    async def _callback_while(flow: _LoopbackAuthorizationFlow, send_requests) -> tuple:
        """Run `flow.callback()` while `send_requests` talks to the callback server."""
        results = {}

        async def callback() -> None:
            try:
                results["callback"] = await flow.callback()
            except OAuthFlowError as e:
                # Re-raised below, outside the task group, which would wrap it in an ExceptionGroup
                results["error"] = e

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(callback)
            results["responses"] = await send_requests()
        if "error" in results:
            raise results["error"]
        return results["callback"], results["responses"]

    @pytest.mark.asyncio
    async def test_captures_the_redirect_and_ignores_other_requests(self, redirect_uri):
        flow = self._flow(redirect_uri)
        await flow.redirect(AUTHORIZATION_URL)
        base = redirect_uri.removesuffix("/callback")

        async def send_requests():
            return [
                await _get(f"{base}/favicon.ico"),
                await _get(f"{redirect_uri}?code=injected&state=other-state"),
                await _get(f"{redirect_uri}?code=the-code&state=expected-state&iss=https://auth.example.com"),
            ]

        result, (wrong_path, wrong_state, redirect) = await self._callback_while(flow, send_requests)

        assert _code_and_state(result) == ("the-code", "expected-state")
        if hasattr(result, "iss"):
            assert result.iss == "https://auth.example.com"
        assert wrong_path.status_code == 404
        assert wrong_state.status_code == 400
        assert redirect.status_code == 200
        assert "Authorization complete" in redirect.text

    @pytest.mark.asyncio
    async def test_error_redirect_raises(self, redirect_uri):
        flow = self._flow(redirect_uri)
        await flow.redirect(AUTHORIZATION_URL)

        async def send_requests():
            return await _get(f"{redirect_uri}?error=access_denied&error_description=User+said+no&state=expected-state")

        with pytest.raises(OAuthFlowError, match=r"access_denied \(User said no\)"):
            await self._callback_while(flow, send_requests)

    @pytest.mark.asyncio
    async def test_redirect_without_code_raises(self, redirect_uri):
        flow = self._flow(redirect_uri)
        await flow.redirect(AUTHORIZATION_URL)

        async def send_requests():
            return await _get(f"{redirect_uri}?state=expected-state")

        with pytest.raises(OAuthFlowError, match="carries no authorization code"):
            await self._callback_while(flow, send_requests)

    @pytest.mark.asyncio
    async def test_timeout_without_terminal_raises(self, redirect_uri):
        flow = self._flow(redirect_uri, callback_timeout=0.2)
        await flow.redirect(AUTHORIZATION_URL)

        with patch("haystack_integrations.tools.mcp.oauth._is_interactive", return_value=False):
            with pytest.raises(OAuthFlowError, match="no interactive terminal"):
                await flow.callback()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pasted", ["{redirect_uri}?code=pasted-code&state=expected-state", "code=pasted-code"])
    async def test_timeout_falls_back_to_the_pasted_url(self, redirect_uri, pasted):
        flow = self._flow(redirect_uri, callback_timeout=0.2)
        await flow.redirect(AUTHORIZATION_URL)

        with (
            patch("haystack_integrations.tools.mcp.oauth._is_interactive", return_value=True),
            patch("builtins.input", return_value=f"  {pasted.format(redirect_uri=redirect_uri)}\n") as mock_input,
        ):
            result = await flow.callback()

        assert _code_and_state(result)[0] == "pasted-code"
        assert AUTHORIZATION_URL in mock_input.call_args.args[0]

    @pytest.mark.asyncio
    async def test_port_in_use_falls_back_to_the_pasted_url(self, redirect_uri, caplog):
        port = int(redirect_uri.rsplit(":", 1)[1].split("/")[0])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", port))
            taken.listen()
            # A long timeout: the fallback must not wait for it when nothing can be received
            flow = self._flow(redirect_uri, callback_timeout=60)
            await flow.redirect(AUTHORIZATION_URL)

            with (
                patch("haystack_integrations.tools.mcp.oauth._is_interactive", return_value=True),
                patch("builtins.input", return_value=f"{redirect_uri}?code=pasted-code&state=expected-state"),
            ):
                with anyio.fail_after(5):
                    result = await flow.callback()

        assert _code_and_state(result) == ("pasted-code", "expected-state")
        assert "Can't listen for the OAuth redirect" in caplog.text

    @pytest.mark.asyncio
    async def test_failing_handler_closes_the_callback_server(self, redirect_uri):
        async def failing_handler(_authorization_url: str) -> None:
            message = "UI unavailable"
            raise RuntimeError(message)

        flow = self._flow(redirect_uri, on_authorization_url=failing_handler)

        with pytest.raises(RuntimeError, match="UI unavailable"):
            await flow.redirect(AUTHORIZATION_URL)

        # The port is free again
        port = urlsplit(redirect_uri).port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))

    @pytest.mark.asyncio
    async def test_opens_the_browser(self, redirect_uri):
        flow = self._flow(redirect_uri, on_authorization_url=None)

        with patch("haystack_integrations.tools.mcp.oauth.webbrowser.open", return_value=True) as mock_open:
            await flow.redirect(AUTHORIZATION_URL)

        mock_open.assert_called_once_with(AUTHORIZATION_URL)

        async def send_requests():
            return await _get(f"{redirect_uri}?code=the-code&state=expected-state")

        result, _ = await self._callback_while(flow, send_requests)
        assert _code_and_state(result) == ("the-code", "expected-state")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("browser", [{"return_value": False}, {"side_effect": webbrowser.Error("no browser")}])
    async def test_logs_the_url_when_no_browser_can_be_opened(self, redirect_uri, caplog, browser):
        flow = self._flow(redirect_uri, on_authorization_url=None)

        with patch("haystack_integrations.tools.mcp.oauth.webbrowser.open", **browser):
            await flow.redirect(AUTHORIZATION_URL)

        assert "Couldn't open a browser" in caplog.text
        assert AUTHORIZATION_URL in caplog.text

        async def send_requests():
            return await _get(f"{redirect_uri}?code=the-code&state=expected-state")

        result, _ = await self._callback_while(flow, send_requests)
        assert _code_and_state(result) == ("the-code", "expected-state")


class TestResolveRedirectUri:
    def test_default(self):
        resolved = urlsplit(_resolve_redirect_uri(None))

        assert resolved.scheme == "http"
        assert resolved.hostname == "127.0.0.1"
        assert resolved.port
        assert resolved.path == "/callback"

    def test_fixed_port_is_kept(self):
        assert _resolve_redirect_uri("http://localhost:8765/oauth") == "http://localhost:8765/oauth"

    @pytest.mark.parametrize("redirect_uri", ["http://localhost:0/oauth", "http://localhost/oauth"])
    def test_free_port_is_filled_in(self, redirect_uri):
        resolved = urlsplit(_resolve_redirect_uri(redirect_uri))

        assert resolved.hostname == "localhost"
        assert resolved.port
        assert resolved.path == "/oauth"


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
