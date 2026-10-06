# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from unittest.mock import Mock, patch

import httpx
import pytest
from haystack import Pipeline, tracing
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tracing.tracer import NullTracer
from haystack.utils import Secret

from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from haystack_integrations.tracing.langfuse import DefaultSpanHandler


@pytest.fixture
def mock_langfuse(monkeypatch):
    monkeypatch.setattr(tracing.tracer, "actual_tracer", NullTracer())
    with patch("haystack_integrations.components.connectors.langfuse.langfuse_connector.Langfuse") as constructor:
        constructor.return_value.get_trace_url.return_value = "https://example.com/trace"
        constructor.return_value.get_current_trace_id.return_value = "12345"
        yield constructor


class CustomSpanHandler(DefaultSpanHandler):
    def handle(self, span, component_type=None):
        pass


class TestRun:
    def test_run(self, mock_langfuse):
        langfuse_connector = LangfuseConnector(
            name="Chat example - OpenAI",
            public=True,
            secret_key=Secret.from_token("secret"),
            public_key=Secret.from_token("public"),
        )

        response = langfuse_connector.run(invocation_context={"some_key": "some_value"})
        assert response["name"] == "Chat example - OpenAI"
        assert response["trace_url"] == "https://example.com/trace"
        assert response["trace_id"] == "12345"
        mock_langfuse.assert_called_once()
        assert tracing.tracer.actual_tracer is langfuse_connector.tracer


class TestSerialization:
    def test_to_dict(self):
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

    def test_to_dict_with_params(self):
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
                    "init_parameters": {},
                },
                "host": "https://example.com",
                "langfuse_client_kwargs": {"timeout": 30.0},
            },
        }

    def test_from_dict(self):
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

    def test_from_dict_without_span_handler(self):
        # All keys that would point to None (span_handler, host, langfuse_client_kwargs) are intentionally absent
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

    def test_from_dict_with_params(self):
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
                    "init_parameters": {},
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

    def test_from_dict_with_legacy_span_handler_format(self):
        # Pipelines serialized before this fix wrap span_handler as {"type": ..., "data": {...}}

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
        assert isinstance(langfuse_connector.span_handler, CustomSpanHandler)

    def test_pipeline_serialization(self, monkeypatch):
        # Set test env vars
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


class TestComponentLifecycle:
    @pytest.mark.parametrize("key", ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"])
    def test_key_resolved_at_warm_up_not_init(self, key, monkeypatch, mock_langfuse):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
        monkeypatch.delenv(key)
        connector = LangfuseConnector("test")
        assert connector.tracer is None
        mock_langfuse.assert_not_called()
        assert not tracing.is_tracing_enabled()

        with pytest.raises(ValueError, match=key):
            connector.warm_up()
        assert connector.tracer is None
        mock_langfuse.assert_not_called()

        monkeypatch.setenv(key, "available-now")
        connector.warm_up()
        assert connector.tracer is not None
        mock_langfuse.assert_called_once()

    def test_warm_up_passes_configuration_and_is_idempotent(self, mock_langfuse):
        client = Mock(spec=httpx.Client)
        handler = CustomSpanHandler()
        connector = LangfuseConnector(
            "configured",
            public=True,
            public_key=Secret.from_token("public"),
            secret_key=Secret.from_token("secret"),
            httpx_client=client,
            span_handler=handler,
            host="https://example.com",
            langfuse_client_kwargs={"timeout": 30, "host": "https://override.example.com"},
        )
        connector.warm_up()
        first_tracer = connector.tracer
        connector.warm_up()
        assert connector.tracer is first_tracer
        mock_langfuse.assert_called_once_with(
            public_key="public",
            secret_key="secret",
            httpx_client=client,
            host="https://override.example.com",
            timeout=30,
        )
        assert handler.tracer is mock_langfuse.return_value
        assert connector.tracer is not None
        assert connector.tracer._name == "configured"
        assert connector.tracer._public is True
