# TEMPORARY DIAGNOSTIC - DO NOT MERGE
# Probes the Llama API with the CI LLAMA_API_KEY at the raw-HTTP level to find
# why every chat completion returns 403 "not available for your application".
# Compares the OpenAI-compat endpoint vs the native endpoint and dumps full
# status / headers / body so we can tell an endpoint/format problem apart from
# an account-wide entitlement problem.
import os

import httpx
import pytest

KEY = os.environ.get("LLAMA_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

# Headers worth surfacing (request id, rate limit, quota, auth hints).
INTERESTING = (
    "x-request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "retry-after",
    "www-authenticate",
    "cf-ray",
    "x-error",
)


def _summarize(label, resp):
    hdrs = {k: v for k, v in resp.headers.items() if k.lower() in INTERESTING}
    body = resp.text
    if len(body) > 1200:
        body = body[:1200] + "...<truncated>"
    return f"--- {label} ---\nHTTP {resp.status_code}\nheaders: {hdrs}\nbody: {body}"


def _post_chat(label, url, payload):
    try:
        r = httpx.post(url, headers=HEADERS, json=payload, timeout=30.0)
        return _summarize(label, r)
    except Exception as e:
        return f"--- {label} ---\nEXCEPTION {type(e).__name__}: {e}"


def _get(label, url):
    try:
        r = httpx.get(url, headers=HEADERS, timeout=30.0)
        return _summarize(label, r)
    except Exception as e:
        return f"--- {label} ---\nEXCEPTION {type(e).__name__}: {e}"


@pytest.mark.integration
@pytest.mark.flaky(reruns=0)
@pytest.mark.skipif(not KEY, reason="Export an env var called LLAMA_API_KEY to run this test.")
def test_diagnostic_raw_probe():
    model = "Llama-4-Scout-17B-16E-Instruct-FP8"
    msgs = [{"role": "user", "content": "hi"}]
    report = [f"key_len={len(KEY)} key_prefix={KEY[:4]!r}"]

    # Model listings on both endpoints (native ids may differ from compat ids).
    report.append(_get("GET compat /compat/v1/models", "https://api.llama.com/compat/v1/models"))
    report.append(_get("GET native /v1/models", "https://api.llama.com/v1/models"))

    # Chat completion via OpenAI-compat endpoint (what the integration uses today).
    report.append(
        _post_chat(
            "POST compat /compat/v1/chat/completions",
            "https://api.llama.com/compat/v1/chat/completions",
            {"model": model, "messages": msgs, "max_tokens": 1, "stream": False},
        )
    )

    # Chat completion via the NATIVE Llama endpoint (different request schema).
    report.append(
        _post_chat(
            "POST native /v1/chat/completions",
            "https://api.llama.com/v1/chat/completions",
            {"model": model, "messages": msgs, "max_completion_tokens": 1},
        )
    )

    header = "\n\n===== META LLAMA RAW DIAGNOSTIC =====\n"
    msg = header + "\n\n".join(report) + "\n====================================="
    raise AssertionError(msg)
