# TEMPORARY DIAGNOSTIC - DO NOT MERGE
# Lists the models the LLAMA_API_KEY can actually access and probes each known
# candidate with a minimal chat completion, to find why the live tests get a
# 403 "model is not available for your application" for Llama-4-Scout.
import os

import pytest
from openai import OpenAI

CANDIDATES = [
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-4-Scout-17B-16E-Instruct-FP8",
    "Llama-3.3-70B-Instruct",
    "Llama-3.3-8B-Instruct",
]


@pytest.mark.integration
@pytest.mark.flaky(reruns=0)
@pytest.mark.skipif(
    not os.environ.get("LLAMA_API_KEY", None),
    reason="Export an env var called LLAMA_API_KEY to run this test.",
)
def test_diagnostic_list_and_probe_models():
    client = OpenAI(
        api_key=os.environ["LLAMA_API_KEY"],
        base_url="https://api.llama.com/compat/v1/",
    )

    report = []

    # 1) What does the key see via the OpenAI-compatible /models endpoint?
    try:
        listed = client.models.list()
        ids = [m.id for m in listed.data]
        report.append("LISTED MODELS (" + str(len(ids)) + "): " + ", ".join(sorted(ids)))
    except Exception as e:
        report.append(f"models.list() FAILED -> {type(e).__name__}: {e}")

    # 2) Probe each known candidate with a minimal chat completion.
    for model in CANDIDATES:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            report.append(f"PROBE OK   {model} -> returned model id: {r.model}")
        except Exception as e:
            report.append(f"PROBE FAIL {model} -> {type(e).__name__}: {e}")

    # Force the report into the CI failure summary.
    msg = "\n\n===== META LLAMA DIAGNOSTIC =====\n" + "\n".join(report) + "\n================================="
    raise AssertionError(msg)
