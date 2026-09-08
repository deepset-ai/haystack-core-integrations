# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

FAKE_TOKEN = "mrs_FAKE_TOKEN_DO_NOT_USE_9f8c7b"


@dataclass
class CapturedRequest:
    method: str
    url: httpx.URL
    headers: httpx.Headers
    params: dict[str, Any] | None
    body: Any
    timeout: httpx.Timeout


@dataclass
class HTTPRecorder:
    response_json: Any = field(default_factory=lambda: {"ok": True})
    response_text: str | None = None
    response_status: int = 200
    response_headers: dict[str, str] | None = None
    requests: list[CapturedRequest] = field(default_factory=list)

    def response(self, method: str, url: str, kwargs: dict[str, Any], timeout: httpx.Timeout) -> httpx.Response:
        params = kwargs.get("params")
        request = httpx.Request(method, url, headers=kwargs.get("headers"), params=params, json=kwargs.get("json"))
        self.requests.append(
            CapturedRequest(
                method=method,
                url=request.url,
                headers=request.headers,
                params=params,
                body=kwargs.get("json"),
                timeout=timeout,
            )
        )
        headers = self.response_headers
        if self.response_text is not None:
            return httpx.Response(
                self.response_status,
                text=self.response_text,
                headers=headers,
                request=request,
            )
        return httpx.Response(self.response_status, json=self.response_json, headers=headers, request=request)


@pytest.fixture
def http_recorder(monkeypatch):
    recorder = HTTPRecorder()

    def sync_request(client, method, url, **kwargs):
        return recorder.response(method, url, kwargs, client.timeout)

    async def async_request(client, method, url, **kwargs):
        return recorder.response(method, url, kwargs, client.timeout)

    monkeypatch.setattr(httpx.Client, "request", sync_request)
    monkeypatch.setattr(httpx.AsyncClient, "request", async_request)
    return recorder
