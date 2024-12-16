import os
import random
import time
from urllib.parse import urlparse

import pytest
import requests
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from requests.auth import HTTPBasicAuth

from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack_integrations.components.generators.cohere import CohereChatGenerator

# don't remove (or move) this env var setting from here, it's needed to turn tracing on
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"


@pytest.mark.integration
@pytest.mark.parametrize(
    "llm_class, env_var, expected_trace",
    [
        (OpenAIChatGenerator, "OPENAI_API_KEY", "OpenAI"),
        (AnthropicChatGenerator, "ANTHROPIC_API_KEY", "Anthropic"),
        (CohereChatGenerator, "COHERE_API_KEY", "Cohere"),
    ],
)
def test_tracing_integration(llm_class, env_var, expected_trace):
    if not all([os.environ.get("LANGFUSE_SECRET_KEY"), os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get(env_var)]):
        pytest.skip(f"Missing required environment variables: LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, or {env_var}")

    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseConnector(name=f"Chat example - {expected_trace}", public=True))
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", llm_class())
    pipe.connect("prompt_builder.prompt", "llm.messages")

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
