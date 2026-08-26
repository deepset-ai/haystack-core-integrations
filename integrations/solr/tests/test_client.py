# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from haystack.utils import Secret

from haystack_integrations.document_stores.solr.client import _SolrClient
from haystack_integrations.document_stores.solr.errors import SolrDocumentStoreError


def _client(handler, **kwargs) -> _SolrClient:
    client = _SolrClient(base_url="http://solr.test/solr", **kwargs)
    transport = httpx.MockTransport(handler)
    client._client = httpx.Client(transport=transport)
    client._async_client = httpx.AsyncClient(transport=transport)
    return client


class TestRequestBuilding:
    def test_adds_the_json_writer_type(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = request.url
            return httpx.Response(200, json={"responseHeader": {"status": 0}})

        _client(handler).request("GET", "admin/info/system")
        assert captured["url"].params["wt"] == "json"

    def test_renders_booleans_the_way_solr_expects(self):
        """httpx would send Python's "True"/"False", which Solr does not accept."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = request.url
            return httpx.Response(200, json={"responseHeader": {"status": 0}})

        _client(handler).request("GET", "admin/cores", params={"deleteIndex": True, "shy": False})
        assert captured["url"].params["deleteIndex"] == "true"
        assert captured["url"].params["shy"] == "false"

    def test_drops_none_parameters(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = request.url
            return httpx.Response(200, json={"responseHeader": {"status": 0}})

        _client(handler).request("GET", "admin/cores", params={"core": None, "action": "STATUS"})
        assert "core" not in captured["url"].params
        assert captured["url"].params["action"] == "STATUS"

    def test_joins_paths_without_doubling_slashes(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = request.url
            return httpx.Response(200, json={"responseHeader": {"status": 0}})

        _client(handler).request("GET", "/admin/info/system")
        assert str(captured["url"]).startswith("http://solr.test/solr/admin/info/system")


class TestErrorMapping:
    def test_prefers_the_solr_error_message_over_the_status(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"msg": "undefined field nope"}})

        with pytest.raises(SolrDocumentStoreError, match="undefined field nope"):
            _client(handler).request("GET", "x")

    def test_non_zero_response_header_status(self):
        """Solr can report failure in the body while still answering with HTTP 200."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"responseHeader": {"status": 500}})

        with pytest.raises(SolrDocumentStoreError, match="returned status 500"):
            _client(handler).request("GET", "x")

    def test_http_error_without_a_json_body(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="<html>Service Unavailable</html>")

        with pytest.raises(SolrDocumentStoreError, match="failed with HTTP 503"):
            _client(handler).request("GET", "x")

    def test_success_with_a_non_json_body(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json at all")

        with pytest.raises(SolrDocumentStoreError, match="non-JSON body"):
            _client(handler).request("GET", "x")

    def test_transport_failure(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            msg = "connection refused"
            raise httpx.ConnectError(msg)

        with pytest.raises(SolrDocumentStoreError, match="Could not reach Solr"):
            _client(handler).request("GET", "x")

    async def test_transport_failure_async(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            msg = "connection refused"
            raise httpx.ConnectError(msg)

        with pytest.raises(SolrDocumentStoreError, match="Could not reach Solr"):
            await _client(handler).request_async("GET", "x")

    async def test_solr_error_async(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"msg": "bad request"}})

        with pytest.raises(SolrDocumentStoreError, match="bad request"):
            await _client(handler).request_async("GET", "x")


class TestAuth:
    def test_no_auth(self):
        assert _SolrClient(base_url="http://solr.test/solr").resolved_auth() is None

    def test_plain_strings(self):
        client = _SolrClient(base_url="http://solr.test/solr", auth=("admin", "secret"))
        assert client.resolved_auth() == ("admin", "secret")

    def test_secrets(self, monkeypatch):
        monkeypatch.setenv("SOLR_USERNAME", "admin")
        monkeypatch.setenv("SOLR_PASSWORD", "secret")
        client = _SolrClient(
            base_url="http://solr.test/solr",
            auth=(Secret.from_env_var("SOLR_USERNAME"), Secret.from_env_var("SOLR_PASSWORD")),
        )
        assert client.resolved_auth() == ("admin", "secret")

    def test_half_configured_auth_is_refused(self):
        client = _SolrClient(base_url="http://solr.test/solr", auth=("admin", None))
        with pytest.raises(SolrDocumentStoreError, match="both username and password"):
            client.resolved_auth()


class TestLifecycle:
    def test_clients_are_created_lazily(self):
        client = _SolrClient(base_url="http://solr.test/solr")
        assert client._client is None
        assert client._async_client is None
        assert isinstance(client.client, httpx.Client)
        assert isinstance(client.async_client, httpx.AsyncClient)

    def test_the_same_client_is_reused(self):
        client = _SolrClient(base_url="http://solr.test/solr")
        assert client.client is client.client

    def test_close_is_safe_when_never_opened(self):
        _SolrClient(base_url="http://solr.test/solr").close()

    async def test_close_async_is_safe_when_never_opened(self):
        await _SolrClient(base_url="http://solr.test/solr").close_async()

    def test_close_allows_reopening(self):
        client = _SolrClient(base_url="http://solr.test/solr")
        first = client.client
        client.close()
        assert client._client is None
        assert client.client is not first

    async def test_close_async_allows_reopening(self):
        client = _SolrClient(base_url="http://solr.test/solr")
        first = client.async_client
        await client.close_async()
        assert client._async_client is None
        assert client.async_client is not first
