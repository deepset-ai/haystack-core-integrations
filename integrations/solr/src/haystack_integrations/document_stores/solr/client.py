# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal HTTP transport for Solr's JSON APIs."""

import contextlib
from typing import Any

import httpx
from haystack.utils import Secret

from .errors import SolrDocumentStoreError

#: Solr signals application-level failures in the response body as well as in the status code.
_SOLR_OK_STATUS = 0


class _SolrClient:
    """
    Lazily-initialized sync and async HTTP clients for a single Solr server.

    Solr's whole surface is JSON over HTTP, so the two `httpx` clients are the only backend-specific
    machinery the document store needs. Keeping them behind this class means the store never has to
    care which of the two it is running on.
    """

    def __init__(
        self,
        *,
        base_url: str,
        auth: tuple[Secret, Secret] | tuple[str, str] | None = None,
        verify_certs: bool = True,
        timeout: float = 30.0,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a client for one Solr server.

        :param base_url: the Solr base URL, for example `http://localhost:8983/solr`.
        :param auth: a username/password pair for basic authentication, as `Secret`s or plain strings.
        :param verify_certs: whether to verify TLS certificates.
        :param timeout: request timeout in seconds.
        :param client_kwargs: extra keyword arguments passed to both `httpx` clients.
        """
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.verify_certs = verify_certs
        self.timeout = timeout
        self.client_kwargs = client_kwargs or {}
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def resolved_auth(self) -> tuple[str, str] | None:
        """
        Resolve the configured credentials.

        :returns: a `(username, password)` pair, or `None` when no credentials are configured.
        :raises SolrDocumentStoreError: if only one of username and password resolves.
        """
        if self.auth is None:
            return None
        username, password = (value.resolve_value() if isinstance(value, Secret) else value for value in self.auth)
        if username is None and password is None:
            return None
        if username is None or password is None:
            msg = "auth requires both username and password to be set, but only one was provided."
            raise SolrDocumentStoreError(msg)
        return str(username), str(password)

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "auth": self.resolved_auth(),
            "verify": self.verify_certs,
            "timeout": self.timeout,
            **self.client_kwargs,
        }

    @property
    def client(self) -> httpx.Client:
        """The sync HTTP client, created on first access."""
        if self._client is None:
            self._client = httpx.Client(**self._client_kwargs())
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """The async HTTP client, created on first access."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(**self._client_kwargs())
        return self._async_client

    def close(self) -> None:
        """Close the sync client. The next request transparently opens a new one."""
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the async client. The next request transparently opens a new one."""
        if self._async_client is not None:
            with contextlib.suppress(Exception):
                await self._async_client.aclose()
            self._async_client = None

    def _build_request(self, path: str, params: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        query = {"wt": "json", **(params or {})}
        # httpx rejects None and renders bools as "True"/"False", neither of which Solr accepts.
        rendered = {
            key: ("true" if value is True else "false" if value is False else value)
            for key, value in query.items()
            if value is not None
        }
        return url, rendered

    @staticmethod
    def _handle_response(response: httpx.Response) -> dict[str, Any]:
        """
        Turn a Solr response into a payload dict, or raise.

        Solr reports failures both in the HTTP status and as an `error` object in the body, and the
        body carries the message worth showing, so it is preferred over the bare status.
        """
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("msg", response.text)
                msg = f"Solr request to {response.request.url} failed: {message}"
                raise SolrDocumentStoreError(msg)
            status = payload.get("responseHeader", {}).get("status")
            if status is not None and status != _SOLR_OK_STATUS:
                msg = f"Solr request to {response.request.url} returned status {status}: {response.text}"
                raise SolrDocumentStoreError(msg)

        if response.is_error:
            msg = f"Solr request to {response.request.url} failed with HTTP {response.status_code}: {response.text}"
            raise SolrDocumentStoreError(msg)

        if not isinstance(payload, dict):
            msg = f"Solr request to {response.request.url} returned a non-JSON body: {response.text[:200]}"
            raise SolrDocumentStoreError(msg)
        return payload

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
    ) -> dict[str, Any]:
        """
        Send a request to Solr.

        :param method: the HTTP method.
        :param path: the path below the Solr base URL.
        :param params: query parameters. `wt=json` is added automatically.
        :param json_body: an optional JSON request body.
        :returns: the parsed response payload.
        :raises SolrDocumentStoreError: if the request fails or Solr reports an error.
        """
        url, query = self._build_request(path, params)
        try:
            response = self.client.request(method, url, params=query, json=json_body)
        except httpx.HTTPError as error:
            msg = f"Could not reach Solr at {url}: {error}"
            raise SolrDocumentStoreError(msg) from error
        return self._handle_response(response)

    async def request_async(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
    ) -> dict[str, Any]:
        """
        Send a request to Solr asynchronously.

        :param method: the HTTP method.
        :param path: the path below the Solr base URL.
        :param params: query parameters. `wt=json` is added automatically.
        :param json_body: an optional JSON request body.
        :returns: the parsed response payload.
        :raises SolrDocumentStoreError: if the request fails or Solr reports an error.
        """
        url, query = self._build_request(path, params)
        try:
            response = await self.async_client.request(method, url, params=query, json=json_body)
        except httpx.HTTPError as error:
            msg = f"Could not reach Solr at {url}: {error}"
            raise SolrDocumentStoreError(msg) from error
        return self._handle_response(response)
