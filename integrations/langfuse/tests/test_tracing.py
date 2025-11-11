# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

import pytest
import requests
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from requests.auth import HTTPBasicAuth

from haystack_integrations.components.connectors.langfuse import LangfuseConnector

# We use the try/except block to prevent import errors when running unit tests with lowest direct dependencies,
# where the subdependencies of LLM integrations might not be respected.
try:
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
except ImportError:
    AnthropicChatGenerator = None
try:
    from haystack_integrations.components.generators.cohere import CohereChatGenerator
except ImportError:
    CohereChatGenerator = None

# don't remove (or move) this env var setting from here, it's needed to turn tracing on
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")


def poll_langfuse(url: str):
    """Utility function to poll Langfuse API until the trace is ready"""
    # Initial wait for trace creation
    time.sleep(10)

    auth = HTTPBasicAuth(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])

    attempts = 5
    delay = 1

    res = None
    while attempts > 0:
        res = requests.get(url, auth=auth, timeout=60.0)
        if res.status_code == 200:
            return res

        attempts -= 1
        if attempts > 0:
            time.sleep(delay)
            delay *= 2

    return res


@pytest.fixture
def basic_pipeline(llm_class, expected_trace):
    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseConnector(name=f"Chat example - {expected_trace}", public=True))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", llm_class())
    pipe.connect("prompt_builder.prompt", "llm.messages")
    return pipe


@pytest.mark.skipif(
    not all([os.environ.get("LANGFUSE_SECRET_KEY"), os.environ.get("LANGFUSE_PUBLIC_KEY")]),
    reason="Missing required environment variables: LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY",
)
@pytest.mark.integration
@pytest.mark.parametrize(
    "llm_class, env_var, expected_trace",
    [
        (OpenAIChatGenerator, "OPENAI_API_KEY", "OpenAI"),
        (AnthropicChatGenerator, "ANTHROPIC_API_KEY", "Anthropic"),
        (CohereChatGenerator, "COHERE_API_KEY", "Cohere"),
    ],
)
def test_tracing_integration(env_var, expected_trace, basic_pipeline):
    if not all([os.environ.get("LANGFUSE_SECRET_KEY"), os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get(env_var)]):
        pytest.skip(f"Missing required environment variable: {env_var}")

    messages = [
        ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    response = basic_pipeline.run(
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
    url = f"{os.environ['LANGFUSE_HOST']}/api/public/traces/{uuid}"

    res = poll_langfuse(url)
    assert res.status_code == 200, f"Failed to retrieve data from Langfuse API: {res.status_code}"

    res_json = res.json()
    assert res_json["name"] == f"Chat example - {expected_trace}"
    assert isinstance(res_json["input"], dict)
    assert res_json["input"]["tracer"]["invocation_context"]["user_id"] == "user_42"
    assert isinstance(res_json["output"], dict)
    assert isinstance(res_json["metadata"], dict)
    assert isinstance(res_json["observations"], list)
    # at least one observation should be a generation
    assert any(obs["type"] == "GENERATION" for obs in res_json["observations"])


@pytest.mark.skipif(
    not all(
        [
            os.environ.get("LANGFUSE_SECRET_KEY"),
            os.environ.get("LANGFUSE_PUBLIC_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        ]
    ),
    reason="Missing required environment variables: LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY",
)
@pytest.mark.integration
def test_tracing_with_sub_pipelines():
    @component
    class SubGenerator:
        def __init__(self):
            self.sub_pipeline = Pipeline()
            self.sub_pipeline.add_component("llm", OpenAIChatGenerator())

        @component.output_types(replies=List[ChatMessage])
        def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
            return {"replies": self.sub_pipeline.run(data={"llm": {"messages": messages}})["llm"]["replies"]}

    @component
    class SubPipeline:
        def __init__(self):
            self.sub_pipeline = Pipeline()
            self.sub_pipeline.add_component("prompt_builder", ChatPromptBuilder())
            self.sub_pipeline.add_component("sub_llm", SubGenerator())
            self.sub_pipeline.connect("prompt_builder.prompt", "sub_llm.messages")

        @component.output_types(replies=List[ChatMessage])
        def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
            return {
                "replies": self.sub_pipeline.run(
                    data={"prompt_builder": {"template": messages, "template_variables": {"location": "Berlin"}}}
                )["sub_llm"]["replies"]
            }

    # Create the main pipeline
    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseConnector(name="Sub-pipeline example"))
    pipe.add_component("sub_pipeline", SubPipeline())

    msgs = [
        ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]
    response = pipe.run(
        data={"sub_pipeline": {"messages": msgs}, "tracer": {"invocation_context": {"user_id": "user_42"}}}
    )

    assert "Berlin" in response["sub_pipeline"]["replies"][0].text
    assert response["tracer"]["trace_url"]
    assert response["tracer"]["trace_id"]

    trace_url = response["tracer"]["trace_url"]
    uuid = os.path.basename(urlparse(trace_url).path)
    url = f"{os.environ['LANGFUSE_HOST']}/api/public/traces/{uuid}"

    res = poll_langfuse(url)
    assert res.status_code == 200, f"Failed to retrieve data from Langfuse API: {res.status_code}"

    res_json = res.json()
    assert res_json["name"] == "Sub-pipeline example"
    assert isinstance(res_json["input"], dict)
    assert isinstance(res_json["output"], dict)
    assert isinstance(res_json["metadata"], dict)
    assert isinstance(res_json["observations"], list)

    observations = res_json["observations"]
    assert len(observations) == 8

    haystack_pipeline_run_observations = [obs for obs in observations if obs["name"] == "haystack.pipeline.run"]
    # There should be two observations for the haystack.pipeline.run span: one for each sub pipeline
    # Main pipeline is stored under the name "Sub-pipeline example"
    assert len(haystack_pipeline_run_observations) == 2
    assert "prompt_builder" in str(haystack_pipeline_run_observations[0])
    assert "llm" in str(haystack_pipeline_run_observations[1])


@pytest.mark.skipif(
    not all(
        [
            os.environ.get("LANGFUSE_SECRET_KEY"),
            os.environ.get("LANGFUSE_PUBLIC_KEY"),
        ]
    ),
    reason="Missing required environment variables: LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY",
)
@pytest.mark.integration
def test_context_cleanup_after_nested_failures():
    """
    Test that tracer context is properly cleaned up even when nested operations fail.

    This test addresses a critical bug where failing nested operations (like inner pipelines)
    could corrupt the tracing context, leaving stale spans that affect subsequent operations.
    The fix ensures proper cleanup through try/finally blocks.

    Before the fix: context would retain spans after failures (length > 0)
    After the fix: context is always cleaned up (length == 0)
    """

    @component
    class FailingParser:
        @component.output_types(result=str)
        def run(self, data: str):
            # This will fail with ValueError when data is not valid JSON
            parsed = json.loads(data)
            return {"result": parsed["key"]}

    @component
    class ComponentWithNestedPipeline:
        def __init__(self):
            # This simulates IntentClassifier's internal pipeline
            self.internal_pipeline = Pipeline()
            self.internal_pipeline.add_component("parser", FailingParser())

        @component.output_types(result=str)
        def run(self, input_data: str):
            # Run nested pipeline - this is where corruption occurs
            result = self.internal_pipeline.run({"parser": {"data": input_data}})
            return {"result": result["parser"]["result"]}

    tracer = LangfuseConnector("test")

    main_pipeline = Pipeline()
    main_pipeline.add_component("nested_component", ComponentWithNestedPipeline())
    main_pipeline.add_component("tracer", tracer)

    # Test 1: First run will fail and should clean up context
    try:
        main_pipeline.run({"nested_component": {"input_data": "invalid json"}})
    except Exception:  # noqa: S110
        pass  # Expected to fail

    # Critical assertion: context should be empty after failed operation
    assert len(tracer.tracer._context) == 0

    # Test 2: Second run should work normally with clean context
    main_pipeline.run({"nested_component": {"input_data": '{"key": "valid"}'}})

    # Critical assertion: context should be empty after successful operation
    assert len(tracer.tracer._context) == 0
