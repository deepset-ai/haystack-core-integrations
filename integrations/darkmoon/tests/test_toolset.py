# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
from haystack import component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from haystack.tools.errors import ToolInvocationError
from haystack.utils import Secret

from haystack_integrations.tools.darkmoon import DarkmoonAPIError, DarkmoonToolset

BASE = "http://darkmoon.test"

FINDINGS = {
    "total": 3,
    "stats": {"critical": 1, "high": 1, "medium": 1},
    "data": [
        {"title": "SQL injection", "severity": "critical"},
        {"title": "Stored XSS", "severity": "high"},
        {"title": "Verbose errors", "severity": "medium"},
    ],
}


class FakeDarkmoon:
    """In-memory stand-in for the Darkmoon Dashboard API, served through `httpx.MockTransport`."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.logins = 0
        self.campaigns: list[dict[str, Any]] = [{"id": "camp_old", "date": "2026-01-01"}]
        self.log_calls = 0
        self.finish_after = 2

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/v1/auth/login":
            self.logins += 1
            body = json.loads(request.content)
            if body["password"] != "pw":
                return httpx.Response(401, json={"detail": "bad credentials"})
            return httpx.Response(200, json={"token": "jwt-1"})
        if request.headers.get("Authorization") != "Bearer jwt-1":
            return httpx.Response(401, json={"detail": "unauthorized"})
        if path == "/api/v1/campaigns":
            return httpx.Response(200, json={"data": self.campaigns})
        if path == "/api/v1/vulnerabilities":
            return httpx.Response(200, json=FINDINGS)
        if path == "/api/v1/run/campaign":
            return httpx.Response(200, json={"run_id": "run-42"})
        if path == "/api/v1/run/logs/run-42":
            self.log_calls += 1
            if self.log_calls == 1:
                return httpx.Response(404, json={"detail": "not found"})
            events = [{"type": "run_completed"}] if self.log_calls >= self.finish_after + 1 else [{"type": "step"}]
            if self.log_calls > self.finish_after:
                self.campaigns.append({"id": "camp_new_example.com", "date": "2026-10-01"})
            return httpx.Response(200, json={"data": events})
        return httpx.Response(500, json={"detail": "unexpected"})


@pytest.fixture
def api():
    return FakeDarkmoon()


@pytest.fixture
def toolset(api, monkeypatch):
    monkeypatch.setenv("DARKMOON_USERNAME", "admin")
    monkeypatch.setenv("DARKMOON_PASSWORD", "pw")
    ts = DarkmoonToolset(base_url=BASE + "/")
    ts._client = httpx.Client(transport=httpx.MockTransport(api))
    return ts


def _tool(toolset: DarkmoonToolset, name: str) -> Tool:
    return next(t for t in toolset if t.name == name)


class TestInit:
    def test_tools(self, toolset):
        assert [t.name for t in toolset] == ["darkmoon_run_pentest", "darkmoon_get_findings", "darkmoon_list_campaigns"]
        assert all(isinstance(t, Tool) for t in toolset)
        assert toolset.base_url == BASE
        assert _tool(toolset, "darkmoon_run_pentest").parameters["required"] == ["target"]

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("DARKMOON_BASE_URL", "http://env.test/")
        assert DarkmoonToolset().base_url == "http://env.test"

    def test_missing_base_url(self, monkeypatch):
        monkeypatch.delenv("DARKMOON_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="DARKMOON_BASE_URL"):
            DarkmoonToolset()

    def test_invalid_max_findings(self):
        with pytest.raises(ValueError, match="max_findings"):
            DarkmoonToolset(base_url=BASE, max_findings=0)


class TestSerde:
    def test_roundtrip(self, monkeypatch):
        monkeypatch.setenv("DARKMOON_USERNAME", "admin")
        monkeypatch.setenv("DARKMOON_PASSWORD", "pw")
        ts = DarkmoonToolset(base_url=BASE, timeout=12.0, max_findings=7)
        data = ts.to_dict()
        assert data == {
            "type": "haystack_integrations.tools.darkmoon.toolset.DarkmoonToolset",
            "data": {
                "base_url": BASE,
                "username": {"type": "env_var", "env_vars": ["DARKMOON_USERNAME"], "strict": True},
                "password": {"type": "env_var", "env_vars": ["DARKMOON_PASSWORD"], "strict": True},
                "timeout": 12.0,
                "max_findings": 7,
            },
        }
        restored = DarkmoonToolset.from_dict(data)
        assert restored.base_url == BASE
        assert restored.timeout == 12.0
        assert restored.max_findings == 7
        assert restored.username == Secret.from_env_var("DARKMOON_USERNAME")
        assert [t.name for t in restored] == [t.name for t in ts]

    def test_agent_serde_does_not_leak_credentials(self, monkeypatch):
        monkeypatch.setenv("DARKMOON_USERNAME", "admin")
        monkeypatch.setenv("DARKMOON_PASSWORD", "pw")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=DarkmoonToolset(base_url=BASE))
        dumped = json.dumps(agent.to_dict())
        assert '"pw"' not in dumped
        assert "sk-test" not in dumped
        assert Agent.from_dict(agent.to_dict()).to_dict() == agent.to_dict()


class TestListAndFindings:
    def test_list_campaigns_logs_in_once(self, toolset, api):
        first = _tool(toolset, "darkmoon_list_campaigns").invoke()
        second = _tool(toolset, "darkmoon_list_campaigns").invoke()
        assert first == {"total": 1, "campaigns": [{"id": "camp_old", "date": "2026-01-01"}]}
        assert second == first
        assert api.logins == 1

    def test_get_findings(self, toolset):
        result = _tool(toolset, "darkmoon_get_findings").invoke(campaign_id="camp_old")
        assert result["campaign_id"] == "camp_old"
        assert result["total"] == 3
        assert result["stats"] == {"critical": 1, "high": 1, "medium": 1}
        assert len(result["findings"]) == 3
        assert result["truncated"] is False

    def test_get_findings_truncates_for_the_llm(self, toolset):
        toolset.max_findings = 2
        result = toolset.get_findings("camp_old")
        assert [f["title"] for f in result["findings"]] == ["SQL injection", "Stored XSS"]
        assert result["total"] == 3
        assert result["truncated"] is True

    def test_campaign_id_is_url_encoded(self, toolset, api):
        toolset.get_findings("a b&c")
        assert api.requests[-1].url.params["campaign_id"] == "a b&c"

    def test_empty_campaign_id(self, toolset):
        with pytest.raises(ValueError, match="campaign_id"):
            toolset.get_findings("  ")


class TestRunPentest:
    def test_start_only(self, toolset, api):
        result = toolset.run_pentest(
            "example.com", wait_for_completion=False, program="bounty", focus="auth, injection"
        )
        assert result == {"status": "started", "run_id": "run-42", "target": "example.com"}
        post = next(r for r in api.requests if r.url.path == "/api/v1/run/campaign")
        assert json.loads(post.content) == {
            "target": "example.com",
            "program": "bounty",
            "focus": ["auth", "injection"],
        }

    def test_wait_returns_findings_of_the_new_campaign(self, toolset, api):
        result = toolset.run_pentest("example.com", poll_interval_seconds=0)
        assert result["run_id"] == "run-42"
        assert result["campaign_id"] == "camp_new_example.com"
        assert result["timed_out"] is False
        assert result["total"] == 3
        vuln = next(r for r in api.requests if r.url.path == "/api/v1/vulnerabilities")
        assert vuln.url.params["campaign_id"] == "camp_new_example.com"

    def test_timeout_never_reports_a_stale_campaign(self, toolset, api):
        api.finish_after = 10_000
        result = toolset.run_pentest("example.com", max_wait_seconds=0, poll_interval_seconds=0)
        assert result["timed_out"] is True
        assert result["campaign_id"] is None
        assert result["findings"] == []

    def test_target_required(self, toolset):
        with pytest.raises(ValueError, match="target"):
            toolset.run_pentest(" ")


class TestErrors:
    def test_bad_credentials(self, api, monkeypatch):
        monkeypatch.setenv("DARKMOON_USERNAME", "admin")
        monkeypatch.setenv("DARKMOON_PASSWORD", "wrong")
        ts = DarkmoonToolset(base_url=BASE)
        ts._client = httpx.Client(transport=httpx.MockTransport(api))
        with pytest.raises(DarkmoonAPIError, match="401: bad credentials") as exc:
            ts.list_campaigns()
        assert "wrong" not in str(exc.value)

    def test_tool_invocation_wraps_errors(self, api, monkeypatch):
        monkeypatch.setenv("DARKMOON_USERNAME", "admin")
        monkeypatch.setenv("DARKMOON_PASSWORD", "wrong")
        ts = DarkmoonToolset(base_url=BASE)
        ts._client = httpx.Client(transport=httpx.MockTransport(api))
        with pytest.raises(ToolInvocationError):
            _tool(ts, "darkmoon_list_campaigns").invoke()

    def test_transport_error(self, monkeypatch):
        monkeypatch.setenv("DARKMOON_USERNAME", "admin")
        monkeypatch.setenv("DARKMOON_PASSWORD", "pw")

        def boom(request: httpx.Request) -> httpx.Response:
            msg = "connection refused"
            raise httpx.ConnectError(msg, request=request)

        ts = DarkmoonToolset(base_url=BASE)
        ts._client = httpx.Client(transport=httpx.MockTransport(boom))
        with pytest.raises(DarkmoonAPIError, match="request failed"):
            ts.list_campaigns()

    def test_close_resets_client_and_token(self, toolset):
        toolset.list_campaigns()
        toolset.close()
        assert toolset._client is None
        assert toolset._token is None


@component
class _ScriptedChatGenerator:
    """Asks for one `darkmoon_get_findings` call, then answers with the tool result it received."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None) -> dict[str, Any]:  # noqa: ARG002
        tool_result = messages[-1].tool_call_result
        if tool_result is not None:
            return {"replies": [ChatMessage.from_assistant(f"Darkmoon said: {tool_result.result}")]}
        call = ToolCall(tool_name="darkmoon_get_findings", arguments={"campaign_id": "camp_old"})
        return {"replies": [ChatMessage.from_assistant(tool_calls=[call])]}


def test_agent_calls_the_toolset(toolset):
    agent = Agent(chat_generator=_ScriptedChatGenerator(), tools=toolset)
    result = agent.run(messages=[ChatMessage.from_user("What did Darkmoon find?")])
    assert "SQL injection" in result["last_message"].text
