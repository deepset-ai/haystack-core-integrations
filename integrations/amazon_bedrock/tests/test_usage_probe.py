# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Throwaway probe: run the chat generator live and dump the real `meta["usage"]` to the GitHub Step Summary,
so we can see exactly what each provider reports for the Agent `context_tokens` work. Not a real test.
"""

import json
import os

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

_QUERY = [ChatMessage.from_user("What is 17*24? Reply with just the number.")]

# A spread of Bedrock model families: Anthropic (cache accounting), DeepSeek R1 (reasoning), Llama (plain).
_MODELS = [
    "global.anthropic.claude-sonnet-4-6",
    "us.deepseek.r1-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
]


def _dump_usage(label, build):
    try:
        usage = build().run(_QUERY)["replies"][-1].meta.get("usage")
        body = json.dumps(usage, indent=2, default=str)
    except Exception as e:  # noqa: BLE001 - probe must never hard-fail; record the error instead
        body = f"ERROR: {type(e).__name__}: {e}"
    block = f"### {label}\n\n```json\n{body}\n```\n"
    print(block)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(block + "\n")


@pytest.mark.integration
def test_probe_usage_shapes():
    for model in _MODELS:
        _dump_usage(f"Amazon Bedrock {model}", lambda m=model: AmazonBedrockChatGenerator(model=m))
