# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import uuid
from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils import Secret
from urllib3.exceptions import HTTPError
from urllib3.response import HTTPResponse

from haystack_integrations.memory_stores.everos import EverOSMemoryStore, EverOSMemoryStoreError


def _response(body, *, status_code=200, request):
    return httpx.Response(status_code, json=body, request=request)


def _store_with_handler(handler):
    store = EverOSMemoryStore(api_key=Secret.from_token("test-token"), max_retries=0)

    def sdk_transport(method, url, **kwargs):
        request = httpx.Request(method, url, content=kwargs.get("body"), headers=kwargs.get("headers"))
        response = handler(request)
        return HTTPResponse(
            body=BytesIO(response.content),
            status=response.status_code,
            headers=dict(response.headers),
            preload_content=False,
        )

    store.client.api_client.rest_client.pool_manager.request = MagicMock(side_effect=sdk_transport)
    return store


class TestEverOSMemoryStore:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"base_url": " "}, "base_url"), ({"timeout": 0}, "timeout")],
    )
    def test_init_validates_configuration(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            EverOSMemoryStore(**kwargs)

    def test_init_is_lazy_and_serializable(self):
        store = EverOSMemoryStore(base_url="https://memory.example/api/v2", timeout=12.5)
        assert store._client is None
        assert store.to_dict() == {
            "type": "haystack_integrations.memory_stores.everos.memory_store.EverOSMemoryStore",
            "init_parameters": {
                "base_url": "https://memory.example/api/v2",
                "api_key": {"env_vars": ["EVEROS_CLOUD_API_KEY"], "strict": True, "type": "env_var"},
                "timeout": 12.5,
                "max_retries": 2,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.memory_stores.everos.memory_store.EverOSMemoryStore",
            "init_parameters": {
                "base_url": "https://memory.example",
                "api_key": {"env_vars": ["MY_EVEROS_KEY"], "strict": True, "type": "env_var"},
                "timeout": 8.0,
            },
        }
        store = EverOSMemoryStore.from_dict(data)
        assert store.base_url == "https://memory.example"
        assert store.api_key == Secret.from_env_var("MY_EVEROS_KEY")
        assert store.timeout == 8.0

    def test_warm_up_adds_bearer_token(self):
        store = EverOSMemoryStore(api_key=Secret.from_token("secret-token"))
        store.warm_up()
        assert store.client.api_client.configuration.access_token == "secret-token"
        assert store.client.api_client.configuration.retries == 0
        store.close()

    def test_default_store_targets_everos_cloud(self):
        store = EverOSMemoryStore()
        assert store.base_url == "https://api.evermind.ai"
        assert store.api_key == Secret.from_env_var("EVEROS_CLOUD_API_KEY")

    def test_client_property_warms_up_once(self):
        with patch("haystack_integrations.memory_stores.everos.memory_store.MemoryApi") as client_class:
            store = EverOSMemoryStore(api_key=Secret.from_token("test-token"))
            assert store.client is store.client
        client_class.assert_called_once()

    def test_add_memories_maps_roles_tool_calls_and_flushes(self):
        requests = []

        def handler(request):
            requests.append(request)
            if request.url.path.endswith("/memory/add"):
                return _response(
                    {"request_id": "add-request", "data": {"message_count": 2, "status": "accumulated"}},
                    request=request,
                )
            return _response({"request_id": "flush-request", "data": {"status": "extracted"}}, request=request)

        store = _store_with_handler(handler)
        result = store.add_memories(
            messages=[
                ChatMessage.from_system("do not persist"),
                ChatMessage.from_user("I prefer concise examples.", meta={"everos_timestamp_ms": 1234}),
                ChatMessage.from_assistant(
                    text=None,
                    tool_calls=[ToolCall(tool_name="lookup", arguments={"topic": "Haystack"}, id="call-1")],
                ),
            ],
            session_id="session-1",
            user_id="alice",
            agent_id="research-agent",
            app_id="haystack",
            project_id="demo",
            flush=True,
        )

        assert result == {
            "message_count": 2,
            "status": "accumulated",
            "request_id": "add-request",
            "flush_status": "extracted",
        }
        add_payload = json.loads(requests[0].content)
        assert add_payload["session_id"] == "session-1"
        assert add_payload["async_mode"] is False
        assert add_payload["messages"][0] == {
            "sender_id": "alice",
            "role": "user",
            "timestamp": 1234,
            "content": "I prefer concise examples.",
        }
        assert add_payload["messages"][1]["sender_id"] == "research-agent"
        assert add_payload["messages"][1]["tool_calls"] == [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"topic":"Haystack"}'},
            }
        ]
        assert json.loads(requests[1].content) == {
            "session_id": "session-1",
            "app_id": "haystack",
            "project_id": "demo",
        }

    def test_add_memories_uses_default_agent_id(self):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return _response(
                {"request_id": "one", "data": {"message_count": 1, "status": "accumulated"}}, request=request
            )

        store = _store_with_handler(handler)
        store.add_memories(messages=[ChatMessage.from_assistant("Hello")], session_id="session", user_id="alice")
        assert captured["messages"][0]["sender_id"] == "haystack-agent"

    def test_add_memories_requires_user_id_for_user_messages(self):
        store = EverOSMemoryStore()
        with pytest.raises(ValueError, match="user_id is required"):
            store.add_memories(messages=[ChatMessage.from_user("Hello")], session_id="session")

    def test_add_memories_requires_session_id(self):
        with pytest.raises(ValueError, match="session_id"):
            EverOSMemoryStore().add_memories(messages=[ChatMessage.from_user("Hello")], session_id="", user_id="alice")

    def test_add_memories_returns_skipped_for_only_system_messages(self):
        store = EverOSMemoryStore()
        result = store.add_memories(messages=[ChatMessage.from_system("system")], session_id="session", user_id="alice")
        assert result["status"] == "skipped"
        assert result["message_count"] == 0

    def test_add_memories_maps_tool_result_and_sender_name(self):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return _response(
                {"request_id": "one", "data": {"message_count": 1, "status": "accumulated"}}, request=request
            )

        tool_call = ToolCall(tool_name="lookup", arguments={}, id="call-result")
        message = ChatMessage.from_tool(
            tool_result={"answer": 42}, origin=tool_call, meta={"sender_name": "Docs Agent"}
        )
        store = _store_with_handler(handler)
        store.add_memories(messages=[message], session_id="session", user_id="alice", agent_id="agent-1")
        assert captured["messages"][0]["content"] == "{'answer': 42}"
        assert captured["messages"][0]["tool_call_id"] == "call-result"
        assert captured["messages"][0]["sender_name"] == "Docs Agent"

    def test_search_user_memory_formats_all_user_results(self):
        captured = {}
        response = {
            "request_id": "search-request",
            "data": {
                "episodes": [
                    {
                        "id": "ep-1",
                        "app_id": "default",
                        "project_id": "default",
                        "timestamp": "2026-08-31T00:00:00Z",
                        "subject": "Database",
                        "type": "Conversation",
                        "user_id": "alice",
                        "session_id": "session-1",
                        "episode": "Alice chose Qdrant for the prototype.",
                        "summary": "Vector database decision",
                        "atomic_facts": [{"id": "fact-1", "content": "Alice chose Qdrant.", "score": 0.9}],
                        "score": 0.82,
                    }
                ],
                "profiles": [
                    {
                        "id": "profile-1",
                        "app_id": "default",
                        "project_id": "default",
                        "user_id": "alice",
                        "profile_data": {"answer_style": "concise"},
                        "score": None,
                    }
                ],
                "agent_cases": [],
                "agent_skills": [],
                "unprocessed_messages": [],
            },
        }

        def handler(request):
            captured.update(json.loads(request.content))
            return _response(response, request=request)

        store = _store_with_handler(handler)
        memories = store.search_memories(
            query="What database does Alice use?",
            user_id="alice",
            session_id="session-1",
            include_profile=True,
            filters={"field": "timestamp", "operator": ">=", "value": 1000},
        )

        assert len(memories) == 2
        assert "Alice chose Qdrant" in (memories[0].text or "")
        assert "Relevant facts" in (memories[0].text or "")
        assert memories[0].meta["everos"]["memory_type"] == "episode"
        assert '"answer_style": "concise"' in (memories[1].text or "")
        assert memories[1].meta["everos"]["request_id"] == "search-request"
        assert captured["filters"] == {"AND": [{"timestamp": {"gte": 1000}}, {"session_id": "session-1"}]}
        assert captured["include_profile"] is True

    def test_search_agent_memory_formats_cases_and_skills(self):
        def handler(request):
            return _response(
                {
                    "request_id": "search-agent",
                    "data": {
                        "episodes": [],
                        "profiles": [],
                        "agent_cases": [
                            {
                                "id": "case-1",
                                "app_id": "default",
                                "project_id": "default",
                                "timestamp": "2026-08-31T00:00:00Z",
                                "session_id": "session-1",
                                "quality_score": 0.9,
                                "agent_id": "research-agent",
                                "task_intent": "Compare databases",
                                "approach": "Benchmark representative queries",
                                "key_insight": "Measure recall and latency together",
                                "score": 0.8,
                            }
                        ],
                        "agent_skills": [
                            {
                                "id": "skill-1",
                                "app_id": "default",
                                "project_id": "default",
                                "agent_id": "research-agent",
                                "name": "database-evaluation",
                                "confidence": 0.8,
                                "maturity_score": 0.7,
                                "description": "Evaluate candidate databases",
                                "content": "Run the benchmark suite and compare trade-offs.",
                                "score": 0.7,
                            }
                        ],
                        "unprocessed_messages": [],
                    },
                },
                request=request,
            )

        store = _store_with_handler(handler)
        memories = store.search_memories(query="How should I evaluate databases?", agent_id="research-agent")
        assert [memory.meta["everos"]["memory_type"] for memory in memories] == ["agent_case", "agent_skill"]
        assert "Benchmark representative queries" in (memories[0].text or "")
        assert "database-evaluation" in (memories[1].text or "")

    def test_search_includes_optional_thresholds_and_unprocessed_messages(self):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return _response(
                {
                    "request_id": "pending",
                    "data": {
                        "episodes": [],
                        "profiles": [],
                        "agent_cases": [],
                        "agent_skills": [],
                        "unprocessed_messages": [
                            {
                                "id": "msg-1",
                                "session_id": "s1",
                                "content": "Waiting for extraction",
                                "app_id": "default",
                                "project_id": "default",
                                "sender_id": "alice",
                                "role": "user",
                                "timestamp": "2026-08-31T00:00:00Z",
                            }
                        ],
                    },
                },
                request=request,
            )

        store = _store_with_handler(handler)
        memories = store.search_memories(
            query="pending",
            user_id="alice",
            top_k=100,
            radius=0.4,
            min_score=0.5,
            include_unprocessed=True,
        )
        assert memories[0].meta["everos"]["memory_type"] == "unprocessed_message"
        assert captured["radius"] == 0.4
        assert captured["min_score"] == 0.5

    def test_search_validates_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            EverOSMemoryStore().search_memories(query="query", user_id="alice", top_k=101)

    @pytest.mark.parametrize(
        ("query", "user_id", "agent_id", "match"),
        [
            ("", "alice", None, "non-empty query"),
            ("query", None, None, "Exactly one"),
            ("query", "alice", "agent", "Exactly one"),
        ],
    )
    def test_search_validates_query_and_owner(self, query, user_id, agent_id, match):
        with pytest.raises(ValueError, match=match):
            EverOSMemoryStore().search_memories(query=query, user_id=user_id, agent_id=agent_id)

    def test_http_error_uses_everos_error_envelope(self):
        def handler(request):
            return _response(
                {"request_id": "bad", "error": {"code": "INVALID_INPUT", "message": "bad owner"}},
                status_code=422,
                request=request,
            )

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="HTTP 422"):
            store.search_memories(query="query", user_id="alice")

    def test_flush_error_uses_standard_http_error(self):
        def handler(request):
            return _response({}, status_code=404, request=request)

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="HTTP 404"):
            store.flush_memories(session_id="session")

    def test_request_error_is_wrapped(self):
        def handler(_request):
            message = "offline"
            raise HTTPError(message)

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="Could not reach EverOS"):
            store.search_memories(query="query", user_id="alice")

    def test_non_json_response_is_wrapped(self):
        def handler(request):
            return httpx.Response(200, text="not-json", request=request)

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="invalid response"):
            store.search_memories(query="query", user_id="alice")

    def test_non_object_json_response_is_wrapped(self):
        def handler(request):
            return httpx.Response(200, json=["invalid"], request=request)

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="invalid response"):
            store.search_memories(query="query", user_id="alice")

    @pytest.mark.parametrize(
        "body",
        [
            {"request_id": "bad"},
            {"request_id": "bad", "data": {"message_count": "one", "status": "accumulated"}},
            {"request_id": "bad", "data": {"message_count": 1, "status": 1}},
        ],
    )
    def test_add_rejects_invalid_success_envelope(self, body):
        store = _store_with_handler(lambda request: _response(body, request=request))
        with pytest.raises(EverOSMemoryStoreError, match="invalid response"):
            store.add_memories(messages=[ChatMessage.from_user("Hello")], session_id="session", user_id="alice")

    def test_search_rejects_non_list_result_bucket(self):
        def handler(request):
            return _response(
                {
                    "request_id": "bad",
                    "data": {
                        "episodes": {"invalid": "not a list"},
                        "profiles": [],
                        "agent_cases": [],
                        "agent_skills": [],
                        "unprocessed_messages": [],
                    },
                },
                request=request,
            )

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="invalid response"):
            store.search_memories(query="query", user_id="alice")

    def test_close_resets_client(self):
        store = _store_with_handler(lambda request: _response({}, request=request))
        store.close()
        assert store._client is None


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("EVEROS_CLOUD_API_KEY"), reason="Set EVEROS_CLOUD_API_KEY for live tests.")
class TestEverOSMemoryStoreIntegration:
    def test_live_everos_cloud_memory_round_trip(self):
        base_url = os.environ.get("EVEROS_TEST_BASE_URL", "https://api.evermind.ai")
        token = uuid.uuid4().hex[:12]
        user_id = f"haystack-integration-{token}"
        session_id = f"haystack-integration-{token}"
        marker = f"The integration marker is cobalt-{token}."
        store = EverOSMemoryStore(
            base_url=base_url,
            api_key=Secret.from_env_var("EVEROS_CLOUD_API_KEY"),
            timeout=30,
        )

        try:
            result = store.add_memories(
                messages=[ChatMessage.from_user(marker)],
                session_id=session_id,
                user_id=user_id,
                flush=True,
            )
            assert result["message_count"] == 1
            assert result["flush_status"]

            memories = []
            for _ in range(8):
                memories = store.search_memories(
                    query=f"What is the integration marker ending in {token}?",
                    user_id=user_id,
                    session_id=session_id,
                    include_profile=True,
                    include_unprocessed=True,
                )
                if any(f"cobalt-{token}" in (memory.text or "") for memory in memories):
                    break
                time.sleep(1.5)

            assert any(f"cobalt-{token}" in (memory.text or "") for memory in memories), [
                {
                    "memory_type": memory.meta.get("everos", {}).get("memory_type"),
                    "text": memory.text,
                }
                for memory in memories
            ]
        finally:
            store.close()

    def test_live_cloud_default_add_is_searchable_and_user_scoped(self):
        base_url = os.environ.get("EVEROS_TEST_BASE_URL", "https://api.evermind.ai")
        token = uuid.uuid4().hex[:12]
        user_id = f"haystack-default-add-{token}"
        other_user_id = f"haystack-other-user-{token}"
        session_id = f"haystack-default-add-{token}"
        marker = f"My cloud notebook color is saffron-{token}."
        store = EverOSMemoryStore(
            base_url=base_url,
            api_key=Secret.from_env_var("EVEROS_CLOUD_API_KEY"),
            timeout=30,
        )

        try:
            result = store.add_memories(
                messages=[ChatMessage.from_user(marker)],
                session_id=session_id,
                user_id=user_id,
            )
            assert result["message_count"] == 1

            memories = []
            for _ in range(8):
                memories = store.search_memories(
                    query="What color is my cloud notebook?",
                    user_id=user_id,
                    method="hybrid",
                    include_profile=True,
                )
                if any(f"saffron-{token}" in (memory.text or "") for memory in memories):
                    break
                time.sleep(1.5)

            assert any(f"saffron-{token}" in (memory.text or "") for memory in memories), {
                "add_status": result["status"],
                "memories": [memory.text for memory in memories],
            }

            isolated = store.search_memories(
                query=f"saffron-{token}",
                user_id=other_user_id,
                method="keyword",
            )
            assert all(f"saffron-{token}" not in (memory.text or "") for memory in isolated)
        finally:
            store.close()

    @pytest.fixture
    def live_store(self):
        store = EverOSMemoryStore(base_url=os.environ.get("EVEROS_TEST_BASE_URL", "https://api.evermind.ai"))
        try:
            yield store
        finally:
            store.close()

    def test_live_filters_include_matching_session_and_exclude_other_session(self, live_store):
        token = uuid.uuid4().hex[:12]
        owner = f"haystack-filter-{token}"
        session = f"session-{token}"
        marker = f"amber-{token}"
        live_store.add_memories(
            messages=[ChatMessage.from_user(f"My favorite notebook is {marker}.")],
            session_id=session,
            user_id=owner,
            flush=True,
        )
        matching_filter = {
            "operator": "AND",
            "conditions": [
                {"field": "session_id", "operator": "==", "value": session},
                {"field": "sender_id", "operator": "==", "value": owner},
            ],
        }
        found = []
        for _ in range(12):
            found = live_store.search_memories(query=marker, user_id=owner, method="hybrid", filters=matching_filter)
            if any(marker in (message.text or "") for message in found):
                break
            time.sleep(2)
        assert any(marker in (message.text or "") for message in found), "Matching filter did not recall test memory."
        excluded = live_store.search_memories(
            query=marker,
            user_id=owner,
            filters={"field": "session_id", "operator": "==", "value": f"absent-{token}"},
        )
        assert not excluded, "A different session filter returned memory."

    def test_live_agent_memory_track_retrieves_case_and_isolates_owner(self, live_store):
        token = uuid.uuid4().hex[:12]
        agent = f"haystack-agent-{token}"
        session = f"agent-session-{token}"
        inspect_call = ToolCall(tool_name="inspect_deployment", arguments={"service": "test-api"}, id="inspect-1")
        patch_call = ToolCall(tool_name="update_probe", arguments={"port": 8000}, id="patch-1")
        verify_call = ToolCall(tool_name="check_readiness", arguments={"attempts": 3}, id="verify-1")
        result = live_store.add_memories(
            messages=[
                ChatMessage.from_user("Diagnose and resolve a deployment readiness failure."),
                ChatMessage.from_assistant(
                    "I will compare the readiness probe with the service configuration.", tool_calls=[inspect_call]
                ),
                ChatMessage.from_tool(
                    tool_result='{"probe_port":8080,"service_port":8000,"error":"connection refused"}',
                    origin=inspect_call,
                ),
                ChatMessage.from_assistant(
                    "The ports differ. Correct the probe instead of restarting the service.", tool_calls=[patch_call]
                ),
                ChatMessage.from_tool(tool_result='{"updated":true,"probe_port":8000}', origin=patch_call),
                ChatMessage.from_assistant(
                    "Now verify that the correction restores readiness.", tool_calls=[verify_call]
                ),
                ChatMessage.from_tool(tool_result='{"checks":[200,200,200],"ready":true}', origin=verify_call),
                ChatMessage.from_assistant(
                    "Task: restore service readiness after deployment. "
                    "I inspected the readiness probe and application logs, found the probe used port 8080 "
                    "while the service listened on port 8000, changed the probe to port 8000, "
                    "and verified three consecutive successful health checks. "
                    "Outcome: deployment recovered. Reusable lesson: compare probe and service ports before restarting."
                ),
            ],
            user_id=f"haystack-agent-user-{token}",
            agent_id=agent,
            session_id=session,
            flush=True,
        )
        assert result["message_count"] == 8
        memories = []
        for _ in range(20):
            memories = live_store.search_memories(
                query="How to fix a readiness probe using the wrong port?", agent_id=agent, method="hybrid"
            )
            if any(message.meta["everos"]["memory_type"] == "agent_case" for message in memories):
                break
            time.sleep(3)
        assert any(message.meta["everos"]["memory_type"] == "agent_case" for message in memories), {
            "reason": "No agent case recalled within the bounded polling window; extraction is not proven.",
            "add_status": result["status"],
            "flush_status": result["flush_status"],
            "request_id": result["request_id"],
            "test_session_id": session,
            "test_agent_id": agent,
            "returned_types": [message.meta["everos"]["memory_type"] for message in memories],
        }
        assert all(message.meta["everos"]["memory_type"] in {"agent_case", "agent_skill"} for message in memories)
        assert any("port" in (message.text or "").lower() for message in memories)
        isolated = live_store.search_memories(query="readiness probe port", agent_id=f"absent-agent-{token}")
        assert not isolated, "Another agent returned memory from the synthetic test."
