# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from unittest.mock import Mock

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from haystack_integrations.tracing.langfuse import DefaultSpanHandler


class CustomSpanHandler(DefaultSpanHandler):
    def handle(self, span, component_type=None):
        pass


class TestLangfuseConnector:
    def test_run(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")

        langfuse_connector = LangfuseConnector(
            name="Chat example - OpenAI",
            public=True,
            secret_key=Secret.from_env_var("LANGFUSE_SECRET_KEY"),
            public_key=Secret.from_env_var("LANGFUSE_PUBLIC_KEY"),
        )

        mock_tracer = Mock()
        mock_tracer.get_trace_url.return_value = "https://example.com/trace"
        mock_tracer.get_trace_id.return_value = "12345"
        langfuse_connector.tracer = mock_tracer

        response = langfuse_connector.run(invocation_context={"some_key": "some_value"})
        assert response["name"] == "Chat example - OpenAI"
        assert response["trace_url"] == "https://example.com/trace"
        assert response["trace_id"] == "12345"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")

        langfuse_connector = LangfuseConnector(name="Chat example - OpenAI")
        serialized = langfuse_connector.to_dict()

        assert serialized == {
            "type": "haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector",
            "init_parameters": {
                "name": "Chat example - OpenAI",
                "public": False,
                "secret_key": {
                    "type": "env_var",
                    "env_vars": ["LANGFUSE_SECRET_KEY"],
                    "strict": True,
                },
                "public_key": {
                    "type": "env_var",
                    "env_vars": ["LANGFUSE_PUBLIC_KEY"],
                    "strict": True,
                },
                "span_handler": None,
                "host": None,
                "langfuse_client_kwargs": None,
            },
        }

    def test_to_dict_with_params(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")

        langfuse_connector = LangfuseConnector(
            name="Chat example - OpenAI",
            public=True,
            secret_key=Secret.from_env_var("LANGFUSE_SECRET_KEY"),
            public_key=Secret.from_env_var("LANGFUSE_PUBLIC_KEY"),
            span_handler=CustomSpanHandler(),
            host="https://example.com",
            langfuse_client_kwargs={"timeout": 30.0},
        )

        serialized = langfuse_connector.to_dict()
        assert serialized == {
            "type": "haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector",
            "init_parameters": {
                "name": "Chat example - OpenAI",
                "public": True,
                "secret_key": {
                    "type": "env_var",
                    "env_vars": ["LANGFUSE_SECRET_KEY"],
                    "strict": True,
                },
                "public_key": {
                    "type": "env_var",
                    "env_vars": ["LANGFUSE_PUBLIC_KEY"],
                    "strict": True,
                },
                "span_handler": {
                    "type": "tests.test_langfuse_connector.CustomSpanHandler",
                    "data": {
                        "type": "tests.test_langfuse_connector.CustomSpanHandler",
                        "init_parameters": {},
                    },
                },
                "host": "https://example.com",
                "langfuse_client_kwargs": {"timeout": 30.0},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")

        data = {
            "type": "haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector",
            "init_parameters": {
                "name": "Chat example - OpenAI",
                "public": False,
                "secret_key": {
                    "type": "env_var",
                    "env_vars": ["LANGFUSE_SECRET_KEY"],
                    "strict": True,
                },
                "public_key": {
                    "type": "env_var",
                    "env_vars": ["LANGFUSE_PUBLIC_KEY"],
                    "strict": True,
                },
                "span_handler": None,
                "host": None,
                "langfuse_client_kwargs": None,
            },
        }
        langfuse_connector = LangfuseConnector.from_dict(data)
        assert langfuse_connector.name == "Chat example - OpenAI"
        assert langfuse_connector.public is False
        assert langfuse_connector.secret_key == Secret.from_env_var("LANGFUSE_SECRET_KEY")
        assert langfuse_connector.public_key == Secret.from_env_var("LANGFUSE_PUBLIC_KEY")
        assert langfuse_connector.span_handler is None
        assert langfuse_connector.host is None
        assert langfuse_connector.langfuse_client_kwargs is None

    def test_from_dict_with_params(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")

        data = {
            "type": "haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector",
            "init_parameters": {
                "name": "Chat example - OpenAI",
                "public": True,
                "secret_key": {
                    "type": "env_var",
                    "env_vars": ["LANGFUSE_SECRET_KEY"],
                    "strict": True,
                },
                "public_key": {
                    "type": "env_var",
                    "env_vars": ["LANGFUSE_PUBLIC_KEY"],
                    "strict": True,
                },
                "span_handler": {
                    "type": "tests.test_langfuse_connector.CustomSpanHandler",
                    "data": {
                        "type": "tests.test_langfuse_connector.CustomSpanHandler",
                        "init_parameters": {},
                    },
                },
                "host": "https://example.com",
                "langfuse_client_kwargs": {"timeout": 30.0},
            },
        }

        langfuse_connector = LangfuseConnector.from_dict(data)
        assert langfuse_connector.name == "Chat example - OpenAI"
        assert langfuse_connector.public is True
        assert langfuse_connector.secret_key == Secret.from_env_var("LANGFUSE_SECRET_KEY")
        assert langfuse_connector.public_key == Secret.from_env_var("LANGFUSE_PUBLIC_KEY")
        assert isinstance(langfuse_connector.span_handler, CustomSpanHandler)
        assert langfuse_connector.host == "https://example.com"
        assert langfuse_connector.langfuse_client_kwargs == {"timeout": 30.0}

    def test_pipeline_serialization(self, monkeypatch):
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
