# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import patch

import pytest
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, ImageContent, StreamingChunk
from haystack.utils.auth import Secret
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from haystack_integrations.components.generators.hetzner.chat.chat_generator import HetznerChatGenerator

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_API_BASE_URL = "https://inference.hetzner.com/api/v1"
COMPONENT_TYPE = "haystack_integrations.components.generators.hetzner.chat.chat_generator.HetznerChatGenerator"

requires_api_key = pytest.mark.skipif(
    not os.environ.get("HETZNER_API_KEY", None),
    reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
)

# an 8x8 red PNG, small enough to inline and enough to exercise the multimodal request path
RED_SQUARE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEUlEQVR42mO4o6GBFTEMLQkAe3tLAYZNzu4AAAAASUVORK5CYII="


class CalendarEvent(BaseModel):
    event_name: str
    event_date: str


@pytest.fixture
def mock_chat_completion():
    """Mock the Hetzner (OpenAI-compatible) chat completion response and reuse it for tests."""
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        mock_chat_completion_create.return_value = ChatCompletion(
            id="foo",
            model=DEFAULT_MODEL,
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
            usage=CompletionUsage(prompt_tokens=57, completion_tokens=40, total_tokens=97),
        )
        yield mock_chat_completion_create


class TestHetznerChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "test-api-key")
        component = HetznerChatGenerator()
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == DEFAULT_MODEL
        assert component.api_base_url == DEFAULT_API_BASE_URL
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_with_unsupported_model_warns(self, caplog, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "test-api-key")
        component = HetznerChatGenerator(model="some-model-not-served-by-hetzner")
        # unknown models are passed on to the API, the user is only warned
        assert component.model == "some-model-not-served-by-hetzner"
        assert "not in the list of models known to be served by the Hetzner Inference API" in caplog.text

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "test-api-key")
        data = HetznerChatGenerator().to_dict()

        assert data["type"] == COMPONENT_TYPE
        assert data["init_parameters"] == {
            "api_key": {"env_vars": ["HETZNER_API_KEY"], "strict": True, "type": "env_var"},
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
        component = HetznerChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="Qwen3.8-27B",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "response_format": CalendarEvent},
            timeout=10,
            max_retries=10,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        init_parameters = component.to_dict()["init_parameters"]

        assert init_parameters["api_key"] == {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"}
        assert init_parameters["model"] == "Qwen3.8-27B"
        assert init_parameters["streaming_callback"] == "haystack.components.generators.utils.print_streaming_chunk"
        assert init_parameters["api_base_url"] == "test-base-url"
        assert init_parameters["timeout"] == 10
        assert init_parameters["max_retries"] == 10
        assert init_parameters["http_client_kwargs"] == {"proxy": "http://localhost:8080"}
        # a Pydantic response_format is converted to OpenAI's JSON schema format, everything else is passed through
        assert init_parameters["generation_kwargs"]["max_tokens"] == 10
        response_format = init_parameters["generation_kwargs"]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "CalendarEvent"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"]["required"] == ["event_name", "event_date"]
        assert response_format["json_schema"]["schema"]["additionalProperties"] is False

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "fake-api-key")
        component = HetznerChatGenerator.from_dict(
            {
                "type": COMPONENT_TYPE,
                "init_parameters": {
                    "api_key": {"env_vars": ["HETZNER_API_KEY"], "strict": True, "type": "env_var"},
                    "model": DEFAULT_MODEL,
                    "api_base_url": "test-base-url",
                    "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                    "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                    "timeout": 10,
                    "max_retries": 10,
                    "tools": None,
                    "http_client_kwargs": {"proxy": "http://localhost:8080"},
                },
            }
        )

        assert component.api_key == Secret.from_env_var("HETZNER_API_KEY")
        assert component.model == DEFAULT_MODEL
        assert component.api_base_url == "test-base-url"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.timeout == 10
        assert component.max_retries == 10
        assert component.tools is None
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "fake-api-key")
        component = HetznerChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = component.run(chat_messages)

        # the generation kwargs are passed on to the Hetzner endpoint
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)
        assert response["replies"][0].text == "Hello world!"

    def test_serde_in_pipeline(self, monkeypatch, tools):
        monkeypatch.setenv("ENV_VAR", "test-key")
        generator = HetznerChatGenerator(
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

        # the Tool serialization format is owned by haystack-ai and varies across its versions, so
        # round-trip the tools through YAML instead of pinning their serialized form
        new_pipeline = Pipeline.loads(pipeline.dumps())
        assert new_pipeline == pipeline
        loaded_generator = new_pipeline.get_component("generator")
        assert loaded_generator.model == generator.model
        assert loaded_generator.generation_kwargs == generator.generation_kwargs
        assert loaded_generator.streaming_callback is print_streaming_chunk
        assert [tool.name for tool in loaded_generator.tools] == [tool.name for tool in tools]
        assert [tool.parameters for tool in loaded_generator.tools] == [tool.parameters for tool in tools]

    @requires_api_key
    @pytest.mark.integration
    def test_live_run(self):
        results = HetznerChatGenerator().run([ChatMessage.from_user("What's the capital of France")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert "Paris" in message.text
        assert DEFAULT_MODEL in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_streaming(self):
        chunks = []

        def callback(chunk: StreamingChunk) -> None:
            chunks.append(chunk)

        component = HetznerChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        message = results["replies"][0]
        assert "Paris" in message.text
        assert message.meta["finish_reason"] == "stop"
        assert len(chunks) > 1
        assert "Paris" in "".join(chunk.content for chunk in chunks if chunk.content)

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = HetznerChatGenerator(tools=tools)
        results = component.run(messages=initial_messages, generation_kwargs={"tool_choice": "auto"})

        assert len(results["replies"]) == 1
        tool_message = results["replies"][0]
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)
        assert tool_message.meta["finish_reason"] == "tool_calls"

        # the model requests the tool once per city, in a single reply
        tool_calls = tool_message.tool_calls
        assert len(tool_calls) == 2
        assert all(tool_call.id and tool_call.tool_name == "weather" for tool_call in tool_calls)
        assert sorted((tool_call.arguments for tool_call in tool_calls), key=lambda args: args["city"]) == [
            {"city": "Berlin"},
            {"city": "Paris"},
        ]

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
        component = HetznerChatGenerator(generation_kwargs={"response_format": CalendarEvent})
        results = component.run([ChatMessage.from_user("The marketing summit takes place on October 12th.")])

        event = json.loads(results["replies"][0].text)
        assert "marketing summit" in event["event_name"].lower()
        assert isinstance(event["event_date"], str)

    @requires_api_key
    @pytest.mark.integration
    def test_live_run_with_image(self):
        image = ImageContent(base64_image=RED_SQUARE_PNG, mime_type="image/png")
        component = HetznerChatGenerator()
        results = component.run([ChatMessage.from_user(content_parts=["What color is this image?", image])])

        message = results["replies"][0]
        # the model names the color in its own words, so accept any reddish description
        assert any(color in message.text.lower() for color in ("red", "crimson", "scarlet"))
        assert message.meta["finish_reason"] == "stop"
