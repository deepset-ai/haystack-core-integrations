# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-request credentials for MCP servers.

A ``Secret`` is resolved once, when the client is built. That is fine for an API key but wrong for
an OAuth access token, which expires: a long-running pipeline holds a credential that was valid at
start-up and has no way to obtain a new one, so recovery means redeploying.

A token provider is asked for the credential on *every* HTTP request the transport makes instead,
and is told when a token was rejected so the next call can produce a fresh one. It is serialized
with the pipeline, so the descriptor -- not the secret -- is what gets stored.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

from haystack.core.serialization import generate_qualified_class_name

from haystack_integrations.tools.mcp.compatibility_layer import http_lib

UNAUTHORIZED = 401


class MCPTokenProvider(ABC):
    """
    Supplies a bearer token for an MCP server, freshly, on demand.

    Implementations are expected to cache: ``token`` is called once per HTTP request, so a provider
    that fetched remotely every time would add a round trip to every tool call.
    """

    @abstractmethod
    def token(self) -> str:
        """Returns a token believed to be currently valid."""

    @abstractmethod
    def invalidate(self) -> None:
        """Marks the last returned token as rejected, so the next ``token`` call obtains another."""

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the provider as a descriptor.

        :returns: The provider's qualified type and its init parameters
        """
        return {"type": generate_qualified_class_name(type(self)), "init_parameters": self._init_parameters()}

    def _init_parameters(self) -> dict[str, Any]:
        """The keyword arguments needed to rebuild this provider. Must not contain secrets."""
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPTokenProvider":
        """
        Rebuilds a provider from its descriptor.

        :param data: A descriptor produced by ``to_dict``
        :returns: The reconstructed provider
        """
        return cls(**(data.get("init_parameters") or {}))


class TokenProviderAuth(http_lib.Auth):  # type: ignore[misc,name-defined]
    """
    Applies a token provider to every request, and retries once when it is rejected.

    Typed against ``Any`` rather than ``httpx.Request``: the SDK uses ``httpx`` on v1 and ``httpx2``
    on v2, and the compatibility layer picks one at import time.

    ``httpx.Auth`` flows may inspect the response and yield a second request, which is what makes
    the refresh-and-retry happen inside the request that failed. The MCP session stays up: no
    reconnect, no lost transport, and nothing for the caller to handle.
    """

    # Tells httpx to buffer the response body so the flow can look at the status code.
    requires_response_body = False

    def __init__(self, provider: MCPTokenProvider) -> None:
        self._provider = provider

    def auth_flow(self, request: Any) -> Generator[Any, Any, None]:
        """Attaches a bearer token, and on a 401 obtains a new one and sends the request again."""
        request.headers["Authorization"] = f"Bearer {self._provider.token()}"
        response = yield request

        if response.status_code != UNAUTHORIZED:
            return

        self._provider.invalidate()
        request.headers["Authorization"] = f"Bearer {self._provider.token()}"
        yield request
