# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time
from typing import Any
from urllib.parse import quote

import httpx
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool, Toolset
from haystack.utils import Secret, deserialize_secrets_inplace

_TERMINAL_EVENTS = frozenset({"run_completed", "run_error"})
_DEFAULT_REQUEST_TIMEOUT = 60.0
_BASE_URL_ENV_VAR = "DARKMOON_BASE_URL"
_HTTP_ERROR_STATUS = 400


class DarkmoonAPIError(Exception):
    """Raised when the Darkmoon Dashboard API rejects or fails a call."""


class DarkmoonToolset(Toolset):
    """
    A Haystack `Toolset` that lets an `Agent` drive a self-hosted [Darkmoon](https://github.com/ASCIT31/Dark-Moon).

    Darkmoon is a GPL-3.0 autonomous AI penetration testing platform. The toolset exposes three tools to the LLM:

    - `darkmoon_run_pentest`: start a campaign against a target, optionally wait for it and return its findings.
    - `darkmoon_get_findings`: read the findings and severity statistics of a campaign.
    - `darkmoon_list_campaigns`: list the campaigns visible to the dashboard user.

    The tools call the Darkmoon Dashboard API of an instance you operate (`POST /api/v1/auth/login`, then
    `/api/v1/run/campaign`, `/api/v1/campaigns` and `/api/v1/vulnerabilities`) and reuse the returned JWT. There is
    no hosted public endpoint.

    ### Open source versus Pro

    The Darkmoon engine and CLI are open source (GPL-3.0). The Dashboard API used by this toolset belongs to the
    Pro edition of Darkmoon. Darkmoon's Pro remediation-to-pull-request feature is deliberately not exposed here.

    ### Responsible use

    Only run assessments against systems you own or are explicitly authorised to test. Findings can contain false
    positives and must be reviewed by a human.

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.tools.darkmoon import DarkmoonToolset

    # Requires DARKMOON_BASE_URL, DARKMOON_USERNAME, DARKMOON_PASSWORD and OPENAI_API_KEY
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=DarkmoonToolset())
    result = agent.run(messages=[ChatMessage.from_user("List my Darkmoon campaigns and summarise the latest one.")])
    print(result["last_message"].text)
    ```
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        username: Secret = Secret.from_env_var("DARKMOON_USERNAME"),
        password: Secret = Secret.from_env_var("DARKMOON_PASSWORD"),
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        max_findings: int = 50,
    ) -> None:
        """
        Create a DarkmoonToolset.

        :param base_url: Base URL of the Darkmoon Dashboard API, for example `http://localhost:8000`. If `None`,
            it is read from the `DARKMOON_BASE_URL` environment variable.
        :param username: The dashboard username. Read from the `DARKMOON_USERNAME` environment variable by default.
        :param password: The dashboard password. Read from the `DARKMOON_PASSWORD` environment variable by default.
        :param timeout: Timeout in seconds of each API request.
        :param max_findings: Maximum number of findings returned to the LLM by one tool call. The total and the
            severity statistics always cover all findings, and `truncated` tells the LLM when the list was cut.
        :raises ValueError: If no base URL is available, or `max_findings` is less than 1.
        """
        resolved_url = base_url or os.environ.get(_BASE_URL_ENV_VAR)
        if not resolved_url:
            msg = (
                f"A Darkmoon base URL is required: pass `base_url` or set the {_BASE_URL_ENV_VAR} environment variable."
            )
            raise ValueError(msg)
        if max_findings < 1:
            msg = f"max_findings must be at least 1, got {max_findings}."
            raise ValueError(msg)

        self.base_url = resolved_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self.max_findings = max_findings
        self._client: httpx.Client | None = None
        self._token: str | None = None
        self._lock = threading.Lock()

        super().__init__(tools=self._build_tools())

    def _build_tools(self) -> list[Tool]:
        return [
            Tool(
                name="darkmoon_run_pentest",
                description=(
                    "Start an autonomous Darkmoon penetration test against one target and return what it found. "
                    "Only use targets the user owns or is explicitly authorised to test. A run can take many "
                    "minutes. With wait_for_completion=false only the run id is returned; use "
                    "darkmoon_list_campaigns and darkmoon_get_findings later. Findings may be false positives and "
                    "need human review."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "The host, URL or scope to assess."},
                        "wait_for_completion": {
                            "type": "boolean",
                            "description": "Wait for the run to finish and return its findings. Defaults to true.",
                        },
                        "program": {
                            "type": "string",
                            "description": "Optional program name or rules of engagement note.",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Optional comma separated focus areas, for example 'auth, injection'.",
                        },
                        "severity": {"type": "string", "description": "Optional minimum severity to report."},
                        "max_wait_seconds": {
                            "type": "integer",
                            "description": "Maximum seconds to wait for the run. Defaults to 1800.",
                        },
                    },
                    "required": ["target"],
                },
                function=self.run_pentest,
            ),
            Tool(
                name="darkmoon_get_findings",
                description=(
                    "Return the findings (vulnerabilities) Darkmoon recorded for one campaign, with the total and "
                    "severity statistics. Findings may be false positives and need human review."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_id": {
                            "type": "string",
                            "description": "The Darkmoon campaign id. Use darkmoon_list_campaigns to find ids.",
                        }
                    },
                    "required": ["campaign_id"],
                },
                function=self.get_findings,
            ),
            Tool(
                name="darkmoon_list_campaigns",
                description="List the Darkmoon pentest campaigns visible to the dashboard user, with their ids.",
                parameters={"type": "object", "properties": {}},
                function=self.list_campaigns,
            ),
        ]

    def _http(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout)
            return self._client

    def close(self) -> None:
        """Close the underlying HTTP client. The toolset reconnects if a tool is invoked again."""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
            self._token = None

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None, *, authenticated: bool = True
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._login()}"
        try:
            response = self._http().request(method, f"{self.base_url}{path}", headers=headers, json=body)
        except httpx.HTTPError as e:
            msg = f"request failed: {e!s}"
            raise DarkmoonAPIError(msg) from e
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        if response.status_code >= _HTTP_ERROR_STATUS:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            msg = f"API error {response.status_code}: {detail if isinstance(detail, str) and detail else 'failed'}"
            raise DarkmoonAPIError(msg)
        return payload

    def _login(self) -> str:
        if self._token:
            return self._token
        payload = self._request(
            "POST",
            "/api/v1/auth/login",
            {"username": self.username.resolve_value(), "password": self.password.resolve_value()},
            authenticated=False,
        )
        token = payload.get("token") if isinstance(payload, dict) else None
        if not token:
            msg = "login did not return a token"
            raise DarkmoonAPIError(msg)
        self._token = str(token)
        return self._token

    def _campaigns(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/v1/campaigns")
        data = payload.get("data") if isinstance(payload, dict) else None
        return data if isinstance(data, list) else []

    def _findings(self, campaign_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/api/v1/vulnerabilities?campaign_id={quote(campaign_id, safe='')}")
        body = payload if isinstance(payload, dict) else {}
        findings = body.get("data") or []
        return {
            "campaign_id": campaign_id,
            "total": body.get("total") or len(findings),
            "stats": body.get("stats") or {},
            "findings": findings[: self.max_findings],
            "truncated": len(findings) > self.max_findings,
        }

    def list_campaigns(self) -> dict[str, Any]:
        """
        List the Darkmoon campaigns visible to the dashboard user.

        :returns: A dictionary with the `total` count and the `campaigns` list.
        :raises DarkmoonAPIError: If the API call fails.
        """
        campaigns = self._campaigns()
        return {"total": len(campaigns), "campaigns": campaigns}

    def get_findings(self, campaign_id: str) -> dict[str, Any]:
        """
        Return the findings Darkmoon recorded for a campaign.

        :param campaign_id: The Darkmoon campaign id.
        :returns: A dictionary with `campaign_id`, `total`, severity `stats`, the `findings` list and `truncated`.
        :raises ValueError: If `campaign_id` is empty.
        :raises DarkmoonAPIError: If the API call fails.
        """
        campaign_id = (campaign_id or "").strip()
        if not campaign_id:
            msg = "campaign_id is required"
            raise ValueError(msg)
        return self._findings(campaign_id)

    def run_pentest(
        self,
        target: str,
        wait_for_completion: bool = True,
        program: str | None = None,
        focus: str | None = None,
        severity: str | None = None,
        max_wait_seconds: int = 1800,
        poll_interval_seconds: float = 5.0,
    ) -> dict[str, Any]:
        """
        Start an autonomous Darkmoon pentest against one target.

        :param target: The host, URL or scope to assess.
        :param wait_for_completion: If `True`, poll the run log until it finishes (or `max_wait_seconds` elapses)
            and return the findings of the campaign created by this run. If `False`, return the run id only.
        :param program: Optional program name or rules of engagement note.
        :param focus: Optional comma separated focus areas, for example `"auth, injection"`.
        :param severity: Optional minimum severity to report.
        :param max_wait_seconds: Maximum seconds to wait for the run.
        :param poll_interval_seconds: Seconds between run status checks.
        :returns: With `wait_for_completion` false, a dictionary with `status`, `run_id` and `target`. Otherwise a
            dictionary with `run_id`, `campaign_id`, `timed_out`, `total`, `stats`, `findings` and `truncated`.
        :raises ValueError: If `target` is empty.
        :raises DarkmoonAPIError: If the API call fails.
        """
        target = (target or "").strip()
        if not target:
            msg = "target is required"
            raise ValueError(msg)

        params: dict[str, Any] = {"target": target}
        if program and program.strip():
            params["program"] = program.strip()
        areas = [p.strip() for p in (focus or "").split(",") if p.strip()]
        if areas:
            params["focus"] = areas
        if severity and severity.strip():
            params["severity"] = severity.strip()

        known_ids = {c.get("id") for c in self._campaigns()}
        handle = self._request("POST", "/api/v1/run/campaign", params)
        run_id = handle.get("run_id") if isinstance(handle, dict) else None
        if not run_id:
            msg = "no run id returned"
            raise DarkmoonAPIError(msg)
        if not wait_for_completion:
            return {"status": "started", "run_id": run_id, "target": target}

        timed_out = self._wait_for_run(str(run_id), max_wait_seconds, poll_interval_seconds)
        campaign = self._resolve_campaign(known_ids, target)
        result: dict[str, Any] = {
            "run_id": run_id,
            "campaign_id": campaign.get("id") if campaign else None,
            "timed_out": timed_out,
            "total": 0,
            "stats": {},
            "findings": [],
            "truncated": False,
        }
        if campaign and campaign.get("id"):
            result.update(self._findings(str(campaign["id"])))
            result["run_id"] = run_id
            result["timed_out"] = timed_out
        return result

    def _wait_for_run(self, run_id: str, max_wait_seconds: int, poll_interval: float) -> bool:
        """Poll the run log until a terminal event. Returns `True` if the wait timed out."""
        deadline = time.monotonic() + max_wait_seconds
        path = f"/api/v1/run/logs/{quote(run_id, safe='')}"
        while True:
            try:
                payload = self._request("GET", path)
            except DarkmoonAPIError as e:
                # The log does not exist until the run writes its first event.
                if "404" not in str(e):
                    raise
                payload = {}
            events = payload.get("data") if isinstance(payload, dict) else None
            if any(isinstance(ev, dict) and ev.get("type") in _TERMINAL_EVENTS for ev in events or []):
                return False
            if time.monotonic() >= deadline:
                return True
            time.sleep(poll_interval)

    def _resolve_campaign(self, known_ids: set, target: str) -> dict[str, Any] | None:
        """
        Find the campaign created by this run.

        The trigger endpoint returns a run id only, so only campaigns that did not exist before the run are
        considered: a stale campaign is never reported as the result of this run.
        """
        fresh = [c for c in self._campaigns() if c.get("id") not in known_ids]
        if not fresh:
            return None
        fresh.sort(key=lambda c: str(c.get("date") or ""), reverse=True)
        host = target.lower()
        for campaign in fresh:
            if host in str(campaign.get("id", "")).lower():
                return campaign
        return fresh[0]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the toolset to a dictionary.

        :returns: Dictionary with serialized data. Credentials are serialized as `Secret` descriptors.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "base_url": self.base_url,
                "username": self.username.to_dict(),
                "password": self.password.to_dict(),
                "timeout": self.timeout,
                "max_findings": self.max_findings,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DarkmoonToolset":
        """
        Deserialize the toolset from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized toolset.
        """
        inner = data["data"]
        deserialize_secrets_inplace(inner, keys=["username", "password"])
        return cls(**inner)
