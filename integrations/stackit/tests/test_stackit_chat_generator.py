# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils.auth import Secret
from openai import OpenAIError
from pydantic import BaseModel

from haystack_integrations.components.generators.stackit.chat.chat_generator import STACKITChatGenerator


class CalendarEvent(BaseModel):
    event_name: str
    event_date: str
    event_location: str


class TestSTACKITChatGenerator:
    def test_supported_models(self):
        """SUPPORTED_MODELS is a non-empty list of strings."""
        models = STACKITChatGenerator.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")
        component = STACKITChatGenerator(model="google/gemma-3-27b-it")
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "google/gemma-3-27b-it"
        assert component.api_base_url == "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_with_parameters(self):
        component = STACKITChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="google/gemma-3-27b-it",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "google/gemma-3-27b-it"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("STACKIT_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            # haystack-ai 2.x raises at init; haystack-ai >= 3.0 raises when the client is created in warm_up
            component = STACKITChatGenerator(model="google/gemma-3-27b-it")
            component.warm_up()

    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")
        component = STACKITChatGenerator(model="google/gemma-3-27b-it")
        component.warm_up()  # with haystack-ai >= 3.0 the client is created during warm-up
        assert component.client.api_key == "test-api-key"

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")
        component = STACKITChatGenerator(model="google/gemma-3-27b-it")
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
            "model": "google/gemma-3-27b-it",
            "streaming_callback": None,
            "api_base_url": "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = STACKITChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="google/gemma-3-27b-it",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={
                "max_tokens": 10,
                "some_test_param": "test-params",
                "response_format": CalendarEvent,
            },
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "https://proxy.example.com:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "google/gemma-3-27b-it",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {
                "max_tokens": 10,
                "some_test_param": "test-params",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CalendarEvent",
                        "strict": True,
                        "schema": {
                            "properties": {
                                "event_name": {"title": "Event Name", "type": "string"},
                                "event_date": {"title": "Event Date", "type": "string"},
                                "event_location": {"title": "Event Location", "type": "string"},
                            },
                            "required": ["event_name", "event_date", "event_location"],
                            "title": "CalendarEvent",
                            "type": "object",
                            "additionalProperties": False,
                        },
                    },
                },
            },
            "timeout": 10.0,
            "max_retries": 2,
            "http_client_kwargs": {"proxy": "https://proxy.example.com:8080"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.stackit.chat.chat_generator.STACKITChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["STACKIT_API_KEY"], "strict": True, "type": "env_var"},
                "model": "google/gemma-3-27b-it",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = STACKITChatGenerator.from_dict(data)
        assert component.model == "google/gemma-3-27b-it"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("STACKIT_API_KEY")

    def test_run_with_params(self, mock_chat_completion, monkeypatch):
        chat_messages = [
            ChatMessage.from_system("You are a helpful assistant"),
            ChatMessage.from_user("What's the capital of France"),
        ]
        monkeypatch.setenv("STACKIT_API_KEY", "fake-api-key")
        component = STACKITChatGenerator(
            model="google/gemma-3-27b-it", generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_string_input(self, mock_chat_completion: MagicMock) -> None:

        component = STACKITChatGenerator(model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"))
        response = component.run("What's the capital of France?")

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "What's the capital of France?"}]

        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)

    def test_run_with_empty_messages(self, mock_chat_completion):
        component = STACKITChatGenerator(model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"))
        assert component.run([]) == {"replies": []}
        mock_chat_completion.assert_not_called()

    def test_run_with_reasoning(self, reasoning_completion):
        component = STACKITChatGenerator(model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"))
        with patch("openai.resources.chat.completions.Completions.create", return_value=reasoning_completion):
            response = component.run([ChatMessage.from_user("What is 2 + 2?")])

        message = response["replies"][0]
        assert message.text == "4"
        assert message.reasoning.reasoning_text == "We need a brief reply. 2+2=4."
        assert message.meta["model"] == "openai/gpt-oss-20b"
        assert message.meta["finish_reason"] == "stop"
        assert message.meta["usage"] == reasoning_completion.usage.model_dump()

    def test_run_with_reasoning_streaming(self, reasoning_chunks):
        chunks = []
        component = STACKITChatGenerator(
            model="openai/gpt-oss-20b", api_key=Secret.from_token("test-api-key"), streaming_callback=chunks.append
        )
        with patch("openai.resources.chat.completions.Completions.create", return_value=iter(reasoning_chunks)):
            response = component.run([ChatMessage.from_user("What is 2 + 2?")])

        assert len(chunks) == 17
        assert chunks[0].content == "" and chunks[0].index is None
        assert [c.reasoning.reasoning_text for c in chunks[1:14]] == [
            "We",
            " need",
            " a",
            " brief",
            " reply",
            ".",
            " ",
            "2",
            "+",
            "2",
            "=",
            "4",
            ".",
        ]
        assert [c.start for c in chunks[1:14]] == [True] + [False] * 12
        assert all(c.index == 0 and c.content == "" and not c.tool_calls for c in chunks[1:14])
        assert chunks[14].content == "4" and chunks[14].start and chunks[14].reasoning is None
        assert chunks[14].index == 1
        assert chunks[15].finish_reason == "stop"
        assert chunks[16].meta["usage"] == reasoning_chunks[-1].usage.model_dump()
        assert all(c.meta["model"] == "openai/gpt-oss-20b" and c.component_info is not None for c in chunks)
        message = response["replies"][0]
        assert message.text == "4"
        assert message.reasoning.reasoning_text == "We need a brief reply. 2+2=4."
        assert message.meta["finish_reason"] == "stop"
        assert message.meta["usage"] == reasoning_chunks[-1].usage.model_dump()

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self) -> None:
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = STACKITChatGenerator(model="google/gemma-3-27b-it")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "paris" in message.text.lower()
        assert "google/gemma-3-27b-it" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = STACKITChatGenerator(model="google/gemma-3-27b-it", streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "paris" in message.text.lower()

        assert "google/gemma-3-27b-it" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "paris" in callback.responses.lower()

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [False, True])
    def test_live_run_with_reasoning(self, streaming):
        chunks = []
        component = STACKITChatGenerator(
            model="openai/gpt-oss-20b",
            generation_kwargs={"max_completion_tokens": 256, "reasoning_effort": "low"},
            streaming_callback=chunks.append if streaming else None,
        )
        message = component.run([ChatMessage.from_user("What is 2 + 2? Reply briefly.")])["replies"][0]
        assert "4" in message.text
        assert message.reasoning.reasoning_text
        assert message.meta["finish_reason"] == "stop"
        if streaming:
            assert "".join(chunk.content for chunk in chunks) == message.text
            assert "".join(chunk.reasoning.reasoning_text for chunk in chunks if chunk.reasoning) == (
                message.reasoning.reasoning_text
            )

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self):
        chat_messages = [
            ChatMessage.from_system("You are a helpful assistant"),
            ChatMessage.from_user("What's the capital of France"),
        ]
        component = STACKITChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_response_format_json_schema(self):
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "CapitalCity",
                "strict": True,
                "schema": {
                    "title": "CapitalCity",
                    "type": "object",
                    "properties": {
                        "city": {"title": "City", "type": "string"},
                        "country": {"title": "Country", "type": "string"},
                    },
                    "required": ["city", "country"],
                    "additionalProperties": False,
                },
            },
        }

        chat_messages = [ChatMessage.from_user("What's the capital of France?")]
        comp = STACKITChatGenerator(
            model="google/gemma-3-27b-it", generation_kwargs={"response_format": response_schema}
        )
        results = comp.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "paris" in msg["city"].lower()
        assert isinstance(msg["country"], str)
        assert "france" in msg["country"].lower()
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("STACKIT_API_KEY", None),
        reason="Export an env var called STACKIT_API_KEY containing the STACKIT API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_response_format_pydantic_model(self):
        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October12th at the Hilton Hotel downtown.")
        ]
        component = STACKITChatGenerator(
            model="google/gemma-3-27b-it",
            generation_kwargs={"response_format": CalendarEvent},
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "marketing summit" in msg["event_name"].lower()
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)
