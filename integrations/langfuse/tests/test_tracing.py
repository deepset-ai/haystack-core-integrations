# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from urllib.parse import urlparse
from typing import Optional

import pytest
import requests
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from requests.auth import HTTPBasicAuth

from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack_integrations.components.generators.cohere import CohereChatGenerator
from haystack_integrations.tracing.langfuse import LangfuseSpan, DefaultSpanHandler
from haystack_integrations.tracing.langfuse.tracer import _COMPONENT_OUTPUT_KEY

# don't remove (or move) this env var setting from here, it's needed to turn tracing on
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"


def poll_langfuse(url: str):
    """Utility function to poll Langfuse API until the trace is ready"""
    # Initial wait for trace creation
    time.sleep(10)

    auth = HTTPBasicAuth(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])

    attempts = 5
    delay = 1

    res = None
    while attempts > 0:
        res = requests.get(url, auth=auth)
        if res.status_code == 200:
            return res

        attempts -= 1
        if attempts > 0:
            time.sleep(delay)
            delay *= 2

    return res


@pytest.fixture
def pipeline_with_env_vars(llm_class, expected_trace):
    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseConnector(name=f"Chat example - {expected_trace}", public=True))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", llm_class())
    pipe.connect("prompt_builder.prompt", "llm.messages")
    return pipe


@pytest.mark.integration
@pytest.mark.parametrize(
    "llm_class, env_var, expected_trace",
    [
        (OpenAIChatGenerator, "OPENAI_API_KEY", "OpenAI"),
        (AnthropicChatGenerator, "ANTHROPIC_API_KEY", "Anthropic"),
        (CohereChatGenerator, "COHERE_API_KEY", "Cohere"),
    ],
)
@pytest.mark.parametrize("pipeline_fixture", ["pipeline_with_env_vars"])
def test_tracing_integration(llm_class, env_var, expected_trace, pipeline_fixture, request):
    if not all([os.environ.get("LANGFUSE_SECRET_KEY"), os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get(env_var)]):
        pytest.skip(f"Missing required environment variables: LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, or {env_var}")

    pipe = request.getfixturevalue(pipeline_fixture)
    messages = [
        ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    response = pipe.run(
        data={
            "prompt_builder": {"template_variables": {"location": "Berlin"}, "template": messages},
            "tracer": {"invocation_context": {"user_id": "user_42"}},
        }
    )
    assert "Berlin" in response["llm"]["replies"][0].text
    assert response["tracer"]["trace_url"]
    assert response["tracer"]["trace_id"]

    trace_url = response["tracer"]["trace_url"]
    uuid = os.path.basename(urlparse(trace_url).path)
    url = f"https://cloud.langfuse.com/api/public/traces/{uuid}"

    res = poll_langfuse(url)
    assert res.status_code == 200, f"Failed to retrieve data from Langfuse API: {res.status_code}"

    res_json = res.json()
    assert res_json["name"] == f"Chat example - {expected_trace}"
    assert isinstance(res_json["input"], dict)
    assert res_json["input"]["tracer"]["invocation_context"]["user_id"] == "user_42"
    assert isinstance(res_json["output"], dict)
    assert isinstance(res_json["metadata"], dict)
    assert isinstance(res_json["observations"], list)
    assert res_json["observations"][0]["type"] == "GENERATION"


def test_pipeline_serialization(monkeypatch):
    """Test that a pipeline with secrets can be properly serialized and deserialized"""

    # Set test env vars
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_api_key")

    # Create pipeline with OpenAI LLM
    pipe = Pipeline()
    pipe.add_component(
        "tracer",
        LangfuseConnector(
            name="Chat example - OpenAI",
            public=True,
            secret_key=Secret.from_env_var("LANGFUSE_SECRET_KEY"),
            public_key=Secret.from_env_var("LANGFUSE_PUBLIC_KEY"),
        ),
    )
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", OpenAIChatGenerator())
    pipe.connect("prompt_builder.prompt", "llm.messages")

    # Serialize
    serialized = pipe.to_dict()

    # Check serialized secrets
    tracer_params = serialized["components"]["tracer"]["init_parameters"]
    assert isinstance(tracer_params["secret_key"], dict)
    assert tracer_params["secret_key"]["type"] == "env_var"
    assert tracer_params["secret_key"]["env_vars"] == ["LANGFUSE_SECRET_KEY"]
    assert isinstance(tracer_params["public_key"], dict)
    assert tracer_params["public_key"]["type"] == "env_var"
    assert tracer_params["public_key"]["env_vars"] == ["LANGFUSE_PUBLIC_KEY"]

    # Deserialize
    new_pipe = Pipeline.from_dict(serialized)

    # Verify pipeline is the same
    assert new_pipe == pipe


class QualityCheckSpanHandler(DefaultSpanHandler):
    """Extends default handler to add quality checks with warning levels."""

    def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
        # First do the default handling (model, usage, etc.)
        super().handle(span, component_type)

        # Then add our custom quality checks
        if component_type == "OpenAIChatGenerator":
            output = span._data.get(_COMPONENT_OUTPUT_KEY, {})
            replies = output.get("replies", [])

            if not replies:
                span._span.update(level="ERROR", status_message="No response received")
                return

            reply = replies[0]
            if "error" in reply.meta:
                span._span.update(level="ERROR", status_message=f"OpenAI error: {reply.meta['error']}")
            elif len(reply.text) > 10:
                span._span.update(level="WARNING", status_message="Response too long (> 10 chars)")
            else:
                span._span.update(level="DEFAULT", status_message="Success")


@pytest.mark.integration
def test_custom_span_handler():
    """Test that custom span handler properly sets Langfuse levels."""
    if not all(
        [os.environ.get("LANGFUSE_SECRET_KEY"), os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("OPENAI_API_KEY")]
    ):
        pytest.skip("Missing required environment variables")

    pipe = Pipeline()
    pipe.add_component(
        "tracer",
        LangfuseConnector(
            name="Quality Check Example",
            public=True,
            span_handler=QualityCheckSpanHandler(),
        ),
    )
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", OpenAIChatGenerator())
    pipe.connect("prompt_builder.prompt", "llm.messages")

    # Test short response
    messages = [
        ChatMessage.from_system("Respond with exactly 3 words."),
        ChatMessage.from_user("What is Berlin?"),
    ]

    response = pipe.run(
        data={
            "prompt_builder": {"template_variables": {}, "template": messages},
            "tracer": {"invocation_context": {"user_id": "test_user"}},
        }
    )

    trace_url = response["tracer"]["trace_url"]
    uuid = os.path.basename(urlparse(trace_url).path)
    url = f"https://cloud.langfuse.com/api/public/traces/{uuid}"

    res = poll_langfuse(url)
    assert res.status_code == 200, f"Failed to retrieve data from Langfuse API: {res.status_code}"

    content = str(res.content)
    assert "WARNING" in content
    assert "Response too long" in content
