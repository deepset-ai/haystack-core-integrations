# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import pytest
from haystack.utils import Secret
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider

from haystack_integrations.tools.mcp import (
    MCPConnectionError,
    MCPTool,
    MCPToolset,
    OAuthConfig,
    SSEServerInfo,
    StreamableHttpServerInfo,
)

pytestmark = pytest.mark.integration

CLIENT_ID = "haystack-client"
CLIENT_SECRET = "haystack-secret"

# `issuer` is only accepted by mcp>=1.30.0 (v1) and mcp>=2.2.0 (v2)
SDK_SUPPORTS_ISSUER = "issuer" in inspect.signature(ClientCredentialsOAuthProvider.__init__).parameters

# An MCP server protected with OAuth, and the authorization server issuing its tokens, in a single process.
# The authorization server only knows the Client Credentials grant with HTTP Basic client authentication, and
# exposes `/stats` so tests can inspect the token requests it received.
_OAUTH_SERVER_TEMPLATE = """
import base64
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings

MCP_PORT, AS_PORT, TRANSPORT, EXPIRES_IN = {mcp_port}, {as_port}, "{transport}", {expires_in}
CLIENT_ID, CLIENT_SECRET = "{client_id}", "{client_secret}"
ISSUER = f"http://127.0.0.1:{{AS_PORT}}"
RESOURCE = f"http://127.0.0.1:{{MCP_PORT}}/{{'sse' if TRANSPORT == 'sse' else 'mcp'}}"

issued_tokens = {{}}
token_requests = []


class AuthServerHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/.well-known/oauth-authorization-server"):
            self._send_json(
                200,
                {{
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{{ISSUER}}/authorize",
                    "token_endpoint": f"{{ISSUER}}/token",
                    "grant_types_supported": ["client_credentials"],
                    "token_endpoint_auth_methods_supported": ["client_secret_basic"],
                }},
            )
        elif self.path == "/stats":
            self._send_json(200, {{"token_requests": token_requests}})
        else:
            self._send_json(404, {{"error": "not_found"}})

    def do_POST(self):
        if self.path != "/token":
            self._send_json(404, {{"error": "not_found"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        form = {{k: v[0] for k, v in urllib.parse.parse_qs(self.rfile.read(length).decode()).items()}}
        authorization = self.headers.get("Authorization", "")
        token_requests.append({{"form": form, "basic_auth": authorization.startswith("Basic ")}})

        expected = "Basic " + base64.b64encode(f"{{CLIENT_ID}}:{{CLIENT_SECRET}}".encode()).decode()
        if authorization != expected or form.get("grant_type") != "client_credentials":
            self._send_json(401, {{"error": "invalid_client", "error_description": "Client authentication failed"}})
            return

        token = f"access-token-{{len(token_requests)}}"
        issued_tokens[token] = time.time() + EXPIRES_IN
        self._send_json(200, {{"access_token": token, "token_type": "Bearer", "expires_in": EXPIRES_IN}})


class TokenVerifier:
    async def verify_token(self, token):
        expires_at = issued_tokens.get(token)
        if expires_at is None or time.time() > expires_at:
            return None
        return AccessToken(token=token, client_id=CLIENT_ID, scopes=[], expires_at=int(expires_at))


auth = AuthSettings(issuer_url=ISSUER, resource_server_url=RESOURCE)
try:
    from mcp.server import MCPServer  # mcp v2

    mcp = MCPServer("OAuth Calculator", token_verifier=TokenVerifier(), auth=auth)
    run_kwargs = {{"host": "127.0.0.1", "port": MCP_PORT}}
except ImportError:
    from mcp.server.fastmcp import FastMCP  # mcp v1

    mcp = FastMCP("OAuth Calculator", host="127.0.0.1", port=MCP_PORT, token_verifier=TokenVerifier(), auth=auth)
    run_kwargs = {{}}


@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    return a + b


if __name__ == "__main__":
    auth_server = ThreadingHTTPServer(("127.0.0.1", AS_PORT), AuthServerHandler)
    threading.Thread(target=auth_server.serve_forever, daemon=True).start()
    mcp.run(transport=TRANSPORT, **run_kwargs)
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(process: subprocess.Popen, port: int) -> None:
    deadline = time.time() + 30
    while True:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            message = f"OAuth MCP server process exited early (code {process.returncode}):\n{stderr}"
            raise RuntimeError(message)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError as e:
            if time.time() > deadline:
                message = f"OAuth MCP server did not start listening on port {port} within 30s"
                raise TimeoutError(message) from e
            time.sleep(0.2)


class OAuthServer:
    """Handle to a running OAuth-protected MCP server and its authorization server."""

    def __init__(self, url: str, issuer: str) -> None:
        self.url = url
        self.issuer = issuer

    def token_requests(self) -> list[dict]:
        with urllib.request.urlopen(f"{self.issuer}/stats", timeout=5) as response:  # noqa: S310
            return json.loads(response.read())["token_requests"]


@pytest.fixture
def oauth_mcp_server():
    """
    Factory fixture that starts an OAuth-protected MCP server exposing an `add` tool.

    Call it with a transport ("sse" or "streamable-http") and the lifetime of the issued tokens in seconds.
    """
    processes = []
    script_paths = []

    def _start(transport: str = "streamable-http", expires_in: int = 3600) -> OAuthServer:
        mcp_port, as_port = _free_port(), _free_port()
        script = _OAUTH_SERVER_TEMPLATE.format(
            mcp_port=mcp_port,
            as_port=as_port,
            transport=transport,
            expires_in=expires_in,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
            temp_file.write(script)
        script_paths.append(temp_file.name)

        process = subprocess.Popen(
            [sys.executable, temp_file.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        processes.append(process)
        _wait_for_port(process, mcp_port)
        _wait_for_port(process, as_port)

        path = "sse" if transport == "sse" else "mcp"
        return OAuthServer(url=f"http://127.0.0.1:{mcp_port}/{path}", issuer=f"http://127.0.0.1:{as_port}")

    yield _start

    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    for script_path in script_paths:
        if os.path.exists(script_path):
            os.remove(script_path)


@pytest.fixture
def client_secret(monkeypatch):
    monkeypatch.setenv("TEST_MCP_CLIENT_SECRET", CLIENT_SECRET)
    return Secret.from_env_var("TEST_MCP_CLIENT_SECRET")


def _oauth_config(server: OAuthServer, client_secret: Secret) -> OAuthConfig:
    return OAuthConfig(
        client_id=CLIENT_ID,
        client_secret=client_secret,
        issuer=server.issuer if SDK_SUPPORTS_ISSUER else None,
    )


def _add(tool: MCPTool, a: int, b: int) -> str:
    result = json.loads(tool.invoke(a=a, b=b))
    assert not result["isError"]
    return result["content"][0]["text"]


class TestOAuthClientCredentials:
    @pytest.mark.parametrize("transport", ["streamable-http", "sse"])
    def test_tool_call_on_oauth_protected_server(self, oauth_mcp_server, client_secret, mcp_tool_cleanup, transport):
        server = oauth_mcp_server(transport)
        server_info_class = SSEServerInfo if transport == "sse" else StreamableHttpServerInfo
        server_info = server_info_class(url=server.url, oauth_config=_oauth_config(server, client_secret))

        tool = mcp_tool_cleanup(MCPTool(name="add", server_info=server_info))

        assert _add(tool, 2, 3) == "5"
        token_requests = server.token_requests()
        assert len(token_requests) == 1
        assert token_requests[0]["basic_auth"]
        assert token_requests[0]["form"]["grant_type"] == "client_credentials"
        assert token_requests[0]["form"]["resource"] == server.url

    def test_new_token_is_fetched_after_expiry(self, oauth_mcp_server, client_secret, mcp_tool_cleanup):
        server = oauth_mcp_server("streamable-http", expires_in=2)
        server_info = StreamableHttpServerInfo(url=server.url, oauth_config=_oauth_config(server, client_secret))

        toolset = mcp_tool_cleanup(MCPToolset(server_info=server_info))
        toolset.warm_up()
        add_tool = next(tool for tool in toolset if tool.name == "add")

        assert _add(add_tool, 1, 1) == "2"
        assert len(server.token_requests()) == 1

        time.sleep(3)

        assert _add(add_tool, 2, 2) == "4"
        assert len(server.token_requests()) == 2

    def test_invalid_client_credentials(self, oauth_mcp_server, monkeypatch, mcp_tool_cleanup):
        server = oauth_mcp_server("streamable-http")
        monkeypatch.setenv("TEST_MCP_CLIENT_SECRET", "wrong-secret")
        server_info = StreamableHttpServerInfo(
            url=server.url,
            oauth_config=_oauth_config(server, Secret.from_env_var("TEST_MCP_CLIENT_SECRET")),
        )

        tool = mcp_tool_cleanup(MCPTool(name="add", server_info=server_info, connection_timeout=30))
        start = time.monotonic()
        with pytest.raises(MCPConnectionError, match="invalid_client"):
            tool.warm_up()

        # The OAuth error must abort the connection, not leave it waiting until connection_timeout
        assert time.monotonic() - start < 15

        assert server.token_requests()

    def test_serialized_toolset_contains_no_credentials(self, oauth_mcp_server, client_secret, mcp_tool_cleanup):
        server = oauth_mcp_server("streamable-http")
        server_info = StreamableHttpServerInfo(url=server.url, oauth_config=_oauth_config(server, client_secret))

        toolset = mcp_tool_cleanup(MCPToolset(server_info=server_info))
        toolset.warm_up()
        add_tool = next(tool for tool in toolset if tool.name == "add")
        assert _add(add_tool, 1, 2) == "3"

        serialized = json.dumps(toolset.to_dict())
        assert "access-token-" not in serialized
        assert CLIENT_SECRET not in serialized
        assert "TEST_MCP_CLIENT_SECRET" in serialized
        assert server_info.headers is None

        restored = MCPToolset.from_dict(json.loads(serialized))
        assert restored.server_info == server_info
