# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from everos_cloud.exceptions import ApiException
from haystack.dataclasses import ChatMessage
from urllib3.exceptions import HTTPError

from haystack_integrations.memory_stores.everos import EverOSMemoryStore, EverOSMemoryStoreError

from .test_memory_store import _response, _store_with_handler


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_search_retries_transient_status_through_real_sdk(status):
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return _response({}, status_code=status, request=request)
        return _response({"request_id": "ok", "data": {"episodes": []}}, request=request)

    store = _store_with_handler(handler)
    store.max_retries = 2
    with patch("haystack_integrations.memory_stores.everos.memory_store.time.sleep") as sleep:
        assert store.search_memories(query="query", user_id="alice") == []
    assert len(attempts) == 2
    sleep.assert_called_once_with(0.5)
    assert attempts[0].headers["Authorization"] == "Bearer test-token"
    store.close()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_search_does_not_retry_permanent_status(status):
    attempts = []

    def handler(request):
        attempts.append(request)
        return _response({}, status_code=status, request=request)

    store = _store_with_handler(handler)
    store.max_retries = 2
    with pytest.raises(EverOSMemoryStoreError, match=f"HTTP {status}"):
        store.search_memories(query="query", user_id="alice")
    assert len(attempts) == 1
    store.close()


@pytest.mark.parametrize("status", [500, 503])
def test_add_does_not_replay_ambiguous_writes(status):
    attempts = []

    def handler(request):
        attempts.append(request)
        return _response({}, status_code=status, request=request)

    store = _store_with_handler(handler)
    store.max_retries = 2
    with pytest.raises(EverOSMemoryStoreError, match=f"HTTP {status}"):
        store.add_memories(messages=[ChatMessage.from_user("test")], session_id="s", user_id="alice")
    assert len(attempts) == 1
    store.close()


def test_add_retries_rate_limit():
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return _response({}, status_code=429, request=request)
        return _response({"request_id": "ok", "data": {"message_count": 1, "status": "accumulated"}}, request=request)

    store = _store_with_handler(handler)
    store.max_retries = 2
    with patch("haystack_integrations.memory_stores.everos.memory_store.time.sleep"):
        assert (
            store.add_memories(messages=[ChatMessage.from_user("test")], session_id="s", user_id="alice")[
                "message_count"
            ]
            == 1
        )
    assert len(attempts) == 2
    store.close()


def test_retry_exhaustion_redacts_server_body():
    attempts = []

    def handler(request):
        attempts.append(request)
        return _response({"error": "private user text and secret-token"}, status_code=503, request=request)

    store = _store_with_handler(handler)
    store.max_retries = 2
    with patch("haystack_integrations.memory_stores.everos.memory_store.time.sleep"):
        with pytest.raises(EverOSMemoryStoreError, match="HTTP 503") as caught:
            store.search_memories(query="query", user_id="alice")
    assert len(attempts) == 3
    assert "private" not in str(caught.value)
    assert "secret-token" not in str(caught.value)
    assert caught.value.__suppress_context__
    store.close()


@pytest.mark.parametrize("read_only,expected_calls", [(True, 3), (False, 1)])
def test_transport_retry_depends_on_operation(read_only, expected_calls):
    store = EverOSMemoryStore(max_retries=2)
    with patch("haystack_integrations.memory_stores.everos.memory_store.time.sleep"):
        with patch.object(store, "warm_up", side_effect=HTTPError("offline")) as operation:
            with pytest.raises(EverOSMemoryStoreError, match="Could not reach"):
                store._request(operation, read_only=read_only)
    assert operation.call_count == expected_calls


@pytest.mark.parametrize("value,expected", [("2", 2.0), ("invalid", 0.5), ("120", None)])
def test_retry_after_is_respected_and_bounded(value, expected):
    error = ApiException(status=429)
    error.headers = {"Retry-After": value}
    assert EverOSMemoryStore._retry_delay(error, 0) == expected


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_retry_count(value):
    with pytest.raises(ValueError, match="max_retries"):
        EverOSMemoryStore(max_retries=value)
