# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import httpx
import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils import Secret

from haystack_integrations.memory_stores.everos import EverOSMemoryStore, EverOSMemoryStoreError


def _response(body, *, status_code=200, request):
    return httpx.Response(status_code, json=body, request=request)


def _store_with_handler(handler):
    store = EverOSMemoryStore(api_key=Secret.from_token("test-token"))
    store._client = httpx.Client(transport=httpx.MockTransport(handler))
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
        with patch("haystack_integrations.memory_stores.everos.memory_store.httpx.Client") as client_class:
            store = EverOSMemoryStore(api_key=Secret.from_token("secret-token"))
            store.warm_up()
        headers = client_class.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-token"

    def test_default_store_targets_everos_cloud(self):
        store = EverOSMemoryStore()
        assert store.base_url == "https://api.evermind.ai"
        assert store.api_key == Secret.from_env_var("EVEROS_CLOUD_API_KEY")

    def test_client_property_warms_up_once(self):
        with patch("haystack_integrations.memory_stores.everos.memory_store.httpx.Client") as client_class:
            client = EverOSMemoryStore(api_key=Secret.from_token("test-token")).client
            _ = client
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
                                "agent_id": "research-agent",
                                "name": "database-evaluation",
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
                            {"id": "msg-1", "session_id": "s1", "content": "Waiting for extraction"}
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
        with pytest.raises(EverOSMemoryStoreError, match="INVALID_INPUT: bad owner"):
            store.search_memories(query="query", user_id="alice")

    def test_flush_error_uses_standard_http_error(self):
        def handler(request):
            return _response({}, status_code=404, request=request)

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="HTTP 404"):
            store.flush_memories(session_id="session")

    def test_request_error_is_wrapped(self):
        def handler(request):
            message = "offline"
            raise httpx.ConnectError(message, request=request)

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="Could not reach EverOS"):
            store.search_memories(query="query", user_id="alice")

    def test_non_json_response_is_wrapped(self):
        def handler(request):
            return httpx.Response(200, text="not-json", request=request)

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="non-JSON"):
            store.search_memories(query="query", user_id="alice")

    def test_non_object_json_response_is_wrapped(self):
        def handler(request):
            return httpx.Response(200, json=["invalid"], request=request)

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match="invalid JSON response object"):
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
                        "episodes": {},
                        "profiles": [],
                        "agent_cases": [],
                        "agent_skills": [],
                        "unprocessed_messages": [],
                    },
                },
                request=request,
            )

        store = _store_with_handler(handler)
        with pytest.raises(EverOSMemoryStoreError, match=r"data\.episodes"):
            store.search_memories(query="query", user_id="alice")

    def test_close_resets_client(self):
        store = _store_with_handler(lambda request: _response({}, request=request))
        store.close()
        assert store._client is None
