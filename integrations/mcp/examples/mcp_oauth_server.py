# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import secrets
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings

# Run this server first, before running the client mcp_oauth_client.py
# It starts an MCP server that only accepts requests carrying an OAuth access token, together with a minimal
# authorization server that issues those tokens with the Client Credentials grant.
#
# The authorization server is for demonstration only: in practice, tokens are issued by your identity provider
# (e.g. Auth0, Keycloak, Microsoft Entra ID), and the MCP server validates them.

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
ISSUER = "http://127.0.0.1:9000"
TOKEN_LIFETIME = 300  # seconds

# The credentials registered for the Haystack client
CLIENT_ID = "haystack-client"
CLIENT_SECRET = "haystack-secret"

issued_tokens: dict[str, float] = {}


class AuthorizationServer(BaseHTTPRequestHandler):
    """Serves the authorization server metadata (RFC 8414) and the token endpoint."""

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path.startswith("/.well-known/oauth-authorization-server"):
            self._send_json(
                200,
                {
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/authorize",
                    "token_endpoint": f"{ISSUER}/token",
                    "grant_types_supported": ["client_credentials"],
                    "token_endpoint_auth_methods_supported": ["client_secret_basic"],
                },
            )
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/token":
            self._send_json(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        form = dict(urllib.parse.parse_qsl(self.rfile.read(length).decode()))

        expected = "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        if self.headers.get("Authorization") != expected or form.get("grant_type") != "client_credentials":
            self._send_json(401, {"error": "invalid_client", "error_description": "Client authentication failed"})
            return

        token = secrets.token_urlsafe(32)
        issued_tokens[token] = time.time() + TOKEN_LIFETIME
        print(f"Issued an access token for {CLIENT_ID} (resource: {form.get('resource')})")
        self._send_json(200, {"access_token": token, "token_type": "Bearer", "expires_in": TOKEN_LIFETIME})


class TokenVerifier:
    """Lets the MCP server accept only tokens issued by the authorization server above."""

    async def verify_token(self, token: str) -> AccessToken | None:
        expires_at = issued_tokens.get(token)
        if expires_at is None or time.time() > expires_at:
            return None
        return AccessToken(token=token, client_id=CLIENT_ID, scopes=[], expires_at=int(expires_at))


# The MCP server advertises ISSUER as its authorization server (RFC 9728), so clients can discover where to get tokens
auth = AuthSettings(issuer_url=ISSUER, resource_server_url=MCP_SERVER_URL)
try:
    from mcp.server import MCPServer  # mcp v2

    mcp = MCPServer("MCP Calculator", token_verifier=TokenVerifier(), auth=auth)
    run_kwargs = {"host": "127.0.0.1", "port": 8000}
except ImportError:
    from mcp.server.fastmcp import FastMCP  # mcp v1

    mcp = FastMCP("MCP Calculator", host="127.0.0.1", port=8000, token_verifier=TokenVerifier(), auth=auth)
    run_kwargs = {}


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


if __name__ == "__main__":
    authorization_server = ThreadingHTTPServer(("127.0.0.1", 9000), AuthorizationServer)
    threading.Thread(target=authorization_server.serve_forever, daemon=True).start()
    mcp.run(transport="streamable-http", **run_kwargs)
