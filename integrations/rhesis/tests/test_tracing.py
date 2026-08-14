# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pytest
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.connectors.rhesis import RhesisConnector


@component
class StubChatGenerator:
    """Minimal ChatGenerator-shaped component for integration tests without an LLM API key."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        reply = ChatMessage.from_assistant(
            "Berlin is the capital of Germany.",
            meta={
                "model": "stub-model",
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            },
        )
        return {"replies": [reply]}


@pytest.fixture
def basic_pipeline(monkeypatch):
    monkeypatch.setenv("RHESIS_API_KEY", os.environ.get("RHESIS_API_KEY", "test-key"))
    monkeypatch.setenv("RHESIS_BASE_URL", os.environ.get("RHESIS_BASE_URL", "http://localhost:8080"))
    monkeypatch.setenv("HAYSTACK_RHESIS_ENFORCE_FLUSH", "false")

    pipe = Pipeline()
    pipe.add_component("tracer", RhesisConnector("Chat example"))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", StubChatGenerator())
    pipe.connect("prompt_builder.prompt", "llm.messages")
    return pipe


@pytest.mark.skipif(
    not os.environ.get("RHESIS_API_KEY"),
    reason="Missing required environment variable: RHESIS_API_KEY",
)
@pytest.mark.integration
def test_tracing_integration(basic_pipeline):
    messages = [
        ChatMessage.from_system("Always respond in German."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]
    response = basic_pipeline.run(
        data={
            "prompt_builder": {"template_variables": {"location": "Berlin"}, "template": messages},
            "tracer": {"invocation_context": {"session_id": "sess-integration"}},
        }
    )
    assert "Berlin" in response["llm"]["replies"][0].text
    assert response["tracer"]["trace_id"]
    # trace_url may be empty when RHESIS_FRONTEND_URL is not configured
    assert "name" in response["tracer"]


@pytest.mark.integration
def test_fail_open_unreachable_backend(monkeypatch, caplog):
    monkeypatch.setenv("RHESIS_API_KEY", "test-key")
    monkeypatch.setenv("RHESIS_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("HAYSTACK_RHESIS_ENFORCE_FLUSH", "false")

    pipe = Pipeline()
    pipe.add_component("tracer", RhesisConnector("Chat example"))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", StubChatGenerator())
    pipe.connect("prompt_builder.prompt", "llm.messages")

    messages = [ChatMessage.from_user("Tell me about {{location}}")]
    with caplog.at_level(logging.WARNING):
        response = pipe.run(
            data={
                "prompt_builder": {"template_variables": {"location": "Berlin"}, "template": messages},
            }
        )
    assert "Berlin" in response["llm"]["replies"][0].text
