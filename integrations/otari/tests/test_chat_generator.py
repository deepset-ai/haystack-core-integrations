# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.utils.auth import Secret
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from haystack_integrations.components.generators.otari import OtariChatGenerator

from .conftest import OTARI_API_BASE_URL, requires_api_key

DEFAULT_MODEL = "openai:gpt-5-mini"
DEFAULT_API_BASE_URL = "https://api.otari.ai/api/v1"
COMPONENT_TYPE = "haystack_integrations.components.generators.otari.chat.chat_generator.OtariChatGenerator"


class CalendarEvent(BaseModel):
    event_name: str
    event_date: str


@pytest.fixture
def mock_chat_completion():
    """Mock the Otari (OpenAI-compatible) chat completion response, including Otari's cost fields."""
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        mock_chat_completion_create.return_value = ChatCompletion(
            id="foo",
            model="gpt-5-mini-2025-08-07",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=1750162525,
            usage=CompletionUsage(
                prompt_tokens=57,
                completion_tokens=40,
                total_tokens=97,
                cost_usd="0.000279",
                pricing_source="defaults",
            ),
        )
        yield mock_chat_completion_create


class TestOtariChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        component = OtariChatGenerator()
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == DEFAULT_MODEL
        assert component.api_base_url == DEFAULT_API_BASE_URL
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.tools is None

    def test_init_with_parameters(self, tools):
        component = OtariChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="anthropic:claude-sonnet-4-6",
            streaming_callback=print_streaming_chunk,
            api_base_url="http://localhost:8000/api/v1",
            generation_kwargs={"max_tokens": 10},
            tools=tools,
            timeout=10,
            max_retries=2,
        )
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "anthropic:claude-sonnet-4-6"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "http://localhost:8000/api/v1"
        assert component.generation_kwargs == {"max_tokens": 10}
        assert component.tools == tools
        assert component.timeout == 10
        assert component.max_retries == 2

    def test_missing_api_key_fails_at_warm_up(self, monkeypatch):
        monkeypatch.delenv("OTARI_API_KEY", raising=False)
        component = OtariChatGenerator()
        with pytest.raises(ValueError, match="OTARI_API_KEY"):
            component.warm_up()

    def test_warm_up_warns_once_about_eu_key_on_default_url(self, caplog):
        component = OtariChatGenerator(api_key=Secret.from_token("otk_v1_eu_test"))
        component.warm_up()
        component.warm_up()
        assert caplog.text.count("belongs to otari.ai's EU region") == 1
        assert "https://eu.api.otari.ai/api/v1" in caplog.text

    @pytest.mark.parametrize(
        ("api_key", "api_base_url"),
        [
            ("otk_v1_eu_test", "https://eu.api.otari.ai/api/v1"),
            ("otk_v1_us_test", DEFAULT_API_BASE_URL),
            ("tk-local-gateway-key", DEFAULT_API_BASE_URL),
        ],
    )
    def test_warm_up_does_not_warn_otherwise(self, caplog, api_key, api_base_url):
        component = OtariChatGenerator(api_key=Secret.from_token(api_key), api_base_url=api_base_url)
        component.warm_up()
        assert "EU region" not in caplog.text

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "test-api-key")
        data = OtariChatGenerator().to_dict()

        assert data["type"] == COMPONENT_TYPE
        # the OpenAI-only organization and tools_strict parameters of the parent are not serialized
        assert data["init_parameters"] == {
            "api_key": {"env_vars": ["OTARI_API_KEY"], "strict": True, "type": "env_var"},
            "model": DEFAULT_MODEL,
            "streaming_callback": None,
            "api_base_url": DEFAULT_API_BASE_URL,
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = OtariChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="openai:gpt-5",
            streaming_callback=print_streaming_chunk,
            api_base_url="http://localhost:8000/api/v1",
            generation_kwargs={"max_tokens": 10, "response_format": CalendarEvent},
            timeout=10,
            max_retries=10,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        init_parameters = component.to_dict()["init_parameters"]

        assert init_parameters["api_key"] == {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"}
        assert init_parameters["model"] == "openai:gpt-5"
        assert init_parameters["streaming_callback"] == "haystack.components.generators.utils.print_streaming_chunk"
        assert init_parameters["api_base_url"] == "http://localhost:8000/api/v1"
        assert init_parameters["timeout"] == 10
        assert init_parameters["max_retries"] == 10
        assert init_parameters["http_client_kwargs"] == {"proxy": "http://localhost:8080"}
        # a Pydantic response_format is converted to OpenAI's JSON schema format, everything else is passed through
        assert init_parameters["generation_kwargs"]["max_tokens"] == 10
        response_format = init_parameters["generation_kwargs"]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "CalendarEvent"
        assert response_format["json_schema"]["schema"]["required"] == ["event_name", "event_date"]

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "fake-api-key")
        component = OtariChatGenerator.from_dict(
            {
                "type": COMPONENT_TYPE,
                "init_parameters": {
                    "api_key": {"env_vars": ["OTARI_API_KEY"], "strict": True, "type": "env_var"},
                    "model": "openai:gpt-5",
                    "api_base_url": "http://localhost:8000/api/v1",
                    "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                    "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                    "timeout": 10,
                    "max_retries": 10,
                    "tools": None,
                    "http_client_kwargs": {"proxy": "http://localhost:8080"},
                },
            }
        )

        assert component.api_key == Secret.from_env_var("OTARI_API_KEY")
        assert component.model == "openai:gpt-5"
        assert component.api_base_url == "http://localhost:8000/api/v1"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.timeout == 10
        assert component.max_retries == 10
        assert component.tools is None
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "fake-api-key")
        component = OtariChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = component.run(chat_messages)

        # the generation kwargs are passed on to the Otari endpoint
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert isinstance(reply, ChatMessage)
        assert reply.text == "Hello world!"
        # Otari adds its cost fields to the usage object, and they reach the reply's meta
        assert reply.meta["usage"]["cost_usd"] == "0.000279"
        assert reply.meta["usage"]["pricing_source"] == "defaults"

    def test_run_with_extra_headers(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("OTARI_API_KEY", "fake-api-key")
        component = OtariChatGenerator()
        component.run(chat_messages, generation_kwargs={"extra_headers": {"Idempotency-Key": "request-1"}})

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["extra_headers"] == {"Idempotency-Key": "request-1"}

    def test_serde_in_pipeline(self, monkeypatch, tools):
        monkeypatch.setenv("ENV_VAR", "test-key")
        generator = OtariChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            generation_kwargs={"temperature": 0.7},
            streaming_callback=print_streaming_chunk,
            tools=tools,
        )
        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        component_dict = pipeline.to_dict()["components"]["generator"]
        assert component_dict["type"] == COMPONENT_TYPE
        assert component_dict["init_parameters"]["generation_kwargs"] == {"temperature": 0.7}

        new_pipeline = Pipeline.loads(pipeline.dumps())
        assert new_pipeline == pipeline
        loaded_generator = new_pipeline.get_component("generator")
        assert loaded_generator.model == generator.model
        assert loaded_generator.generation_kwargs == generator.generation_kwargs
        assert loaded_generator.streaming_callback is print_streaming_chunk
        assert [tool.name for tool in loaded_generator.tools] == [tool.name for tool in tools]

    @requires_api_key
    @pytest.mark.integration
    def test_live_run(self):
        component = OtariChatGenerator(api_base_url=OTARI_API_BASE_URL)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-5-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        # the local test gateway prices the model with Otari's bundled prices, otari.ai with the organization's
        assert float(message.meta["usage"]["cost_usd"]) > 0
        assert message.meta["usage"]["pricing_source"] in ("defaults", "deployment", "organization")

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_streaming(self):
        chunks = []

        def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        component = OtariChatGenerator(
            api_base_url=OTARI_API_BASE_URL,
            streaming_callback=callback,
            generation_kwargs={"stream_options": {"include_usage": True}},
        )
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        message = results["replies"][0]
        assert "Paris" in message.text
        assert message.meta["finish_reason"] == "stop"
        assert len(chunks) > 1
        assert "Paris" in "".join(chunk.content for chunk in chunks if chunk.content)
        assert float(message.meta["usage"]["cost_usd"]) > 0

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = OtariChatGenerator(api_base_url=OTARI_API_BASE_URL, tools=tools)
        results = component.run(messages=initial_messages, generation_kwargs={"tool_choice": "auto"})

        assert len(results["replies"]) == 1
        tool_message = results["replies"][0]
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)
        assert tool_message.meta["finish_reason"] == "tool_calls"

        # the model requests the tool once per city, in a single reply
        tool_calls = tool_message.tool_calls
        assert len(tool_calls) == 2
        assert all(tool_call.id and tool_call.tool_name == "weather" for tool_call in tool_calls)
        assert sorted(tool_call.arguments["city"] for tool_call in tool_calls) == ["Berlin", "Paris"]

        # pass the tool results back to the model to get the final response
        results = component.run(
            [
                initial_messages[0],
                tool_message,
                ChatMessage.from_tool(tool_result="22° C and sunny", origin=tool_calls[0]),
                ChatMessage.from_tool(tool_result="16° C and windy", origin=tool_calls[1]),
            ]
        )
        final_message = results["replies"][0]
        assert final_message.is_from(ChatRole.ASSISTANT)
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_with_response_format(self):
        component = OtariChatGenerator(
            api_base_url=OTARI_API_BASE_URL, generation_kwargs={"response_format": CalendarEvent}
        )
        results = component.run([ChatMessage.from_user("The marketing summit takes place on October 12th.")])

        event = json.loads(results["replies"][0].text)
        assert "marketing summit" in event["event_name"].lower()
        assert isinstance(event["event_date"], str)
