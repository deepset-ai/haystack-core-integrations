# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils.auth import Secret
from openai import OpenAIError

from haystack_integrations.components.generators.stackit.chat.chat_generator import STACKITChatGenerator


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

    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("STACKIT_API_KEY", "test-api-key")
        component = STACKITChatGenerator(model="google/gemma-3-27b-it")
        component.warm_up()
        assert component._client.api_key == "test-api-key"

    def test_warm_up_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("STACKIT_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            component = STACKITChatGenerator(model="google/gemma-3-27b-it")
            component.warm_up()

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

        restored = STACKITChatGenerator.from_dict(data)
        assert restored.to_dict() == data
        assert restored.api_key.resolve_value() == "test-api-key"

    @pytest.mark.parametrize("tools_strict", [None, False, True])
    def test_from_dict(self, monkeypatch, tools_strict):
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
        if tools_strict is not None:
            data["init_parameters"]["tools_strict"] = tools_strict
        component = STACKITChatGenerator.from_dict(data)
        assert component.model == "google/gemma-3-27b-it"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("STACKIT_API_KEY")

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
