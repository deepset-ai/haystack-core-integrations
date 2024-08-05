import os

# don't remove (or move) this env var setting from here, it's needed to turn tracing on
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from urllib.parse import urlparse

import pytest
import requests

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from requests.auth import HTTPBasicAuth

from haystack_integrations.components.connectors.langfuse import LangfuseConnector


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("LANGFUSE_SECRET_KEY", None) and not os.environ.get("LANGFUSE_PUBLIC_KEY", None),
    reason="Export an env var called LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY containing Langfuse credentials.",
)
def test_tracing_integration():

    pipe = Pipeline()
    pipe.add_component("tracer", LangfuseConnector(name="Chat example", public=True))  # public so anyone can verify run
    pipe.add_component("prompt_builder", ChatPromptBuilder())
    pipe.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))

    pipe.connect("prompt_builder.prompt", "llm.messages")

    messages = [
        ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    response = pipe.run(data={"prompt_builder": {"template_variables": {"location": "Berlin"}, "template": messages}})
    assert "Berlin" in response["llm"]["replies"][0].content
    assert response["tracer"]["trace_url"]
    url = "https://cloud.langfuse.com/api/public/traces/"
    trace_url = response["tracer"]["trace_url"]
    parsed_url = urlparse(trace_url)
    # trace id is the last part of the path (after the last '/')
    uuid = os.path.basename(parsed_url.path)
    try:
        # GET request with Basic Authentication on the Langfuse API
        response = requests.get(
            url + uuid, auth=HTTPBasicAuth(os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY"))
        )

        assert response.status_code == 200, f"Failed to retrieve data from Langfuse API: {response.status_code}"
    except requests.exceptions.RequestException as e:
        assert False, f"Failed to retrieve data from Langfuse API: {e}"
