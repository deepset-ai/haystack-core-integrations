import os
import time
from urllib.parse import urlparse

import pytest
import requests
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from requests.auth import HTTPBasicAuth
import httpx

from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack_integrations.components.generators.cohere import CohereChatGenerator

# don't remove (or move) this env var setting from here, it's needed to turn tracing on
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"


@pytest.fixture
def pipeline_with_env_vars(llm_class, expected_trace):
    """Pipeline factory using environment variables for Langfuse authentication"""
    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseConnector(name=f"Chat example - {expected_trace}", public=True))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", llm_class())
    pipe.connect("prompt_builder.prompt", "llm.messages")
    return pipe


@pytest.fixture
def pipeline_with_secrets(llm_class, expected_trace):
    """Pipeline factory using Secret objects for Langfuse authentication"""
    pipe = Pipeline()
    pipe.add_component(
        "tracer",
        LangfuseConnector(
            name=f"Chat example - {expected_trace}",
            public=True,
            secret_key=Secret.from_env_var("LANGFUSE_SECRET_KEY"),
            public_key=Secret.from_env_var("LANGFUSE_PUBLIC_KEY"),
        ),
    )
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", llm_class())
    pipe.connect("prompt_builder.prompt", "llm.messages")
    return pipe


@pytest.fixture
def pipeline_with_custom_client(llm_class, expected_trace):
    """Pipeline factory using custom httpx client for Langfuse"""
    pipe = Pipeline()
    custom_client = httpx.Client(timeout=30.0)  # Custom timeout of 30 seconds
    pipe.add_component(
        "tracer",
        LangfuseConnector(
            name=f"Chat example - {expected_trace}",
            public=True,
            secret_key=Secret.from_env_var("LANGFUSE_SECRET_KEY"),
            public_key=Secret.from_env_var("LANGFUSE_PUBLIC_KEY"),
            httpx_client=custom_client,
        ),
    )
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
@pytest.mark.parametrize(
    "pipeline_fixture", ["pipeline_with_env_vars", "pipeline_with_secrets", "pipeline_with_custom_client"]
)
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

    trace_url = response["tracer"]["trace_url"]
    uuid = os.path.basename(urlparse(trace_url).path)
    url = f"https://cloud.langfuse.com/api/public/traces/{uuid}"

    # Poll the Langfuse API a bit as the trace might not be ready right away
    attempts = 5
    delay = 1
    while attempts >= 0:
        res = requests.get(
            url, auth=HTTPBasicAuth(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
        )
        if attempts > 0 and res.status_code != 200:
            attempts -= 1
            time.sleep(delay)
            delay *= 2
            continue
        assert res.status_code == 200, f"Failed to retrieve data from Langfuse API: {res.status_code}"

        # check if the trace contains the expected LLM name
        assert expected_trace in str(res.content)
        # check if the trace contains the expected generation span
        assert "GENERATION" in str(res.content)
        # check if the trace contains the expected user_id
        assert "user_42" in str(res.content)
        break


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
