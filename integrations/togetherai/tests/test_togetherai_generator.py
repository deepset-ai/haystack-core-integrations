# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import pytz
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret
from openai import OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from haystack_integrations.components.generators.togetherai.generator import TogetherAIGenerator


@pytest.fixture
def mock_chat_completion():
    """
    Mock the Together AI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage=CompletionUsage(
                prompt_tokens=57,
                completion_tokens=40,
                total_tokens=97,
            ),
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


class TestTogetherAIGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "test-api-key")
        component = TogetherAIGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.client.timeout == 30
        assert component.client.max_retries == 5

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            TogetherAIGenerator()

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = TogetherAIGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=40.0,
            max_retries=1,
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.client.timeout == 40.0
        assert component.client.max_retries == 1
        assert component.api_base_url == "test-base-url"

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "test-api-key")
        component = TogetherAIGenerator(
            api_key=Secret.from_env_var("TOGETHER_API_KEY"),
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.togetherai.generator.TogetherAIGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["TOGETHER_API_KEY"], "strict": True, "type": "env_var"},
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "streaming_callback": None,
                "system_prompt": None,
                "api_base_url": "https://api.together.xyz/v1",
                "generation_kwargs": {},
                "timeout": None,
                "max_retries": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "test-api-key")
        component = TogetherAIGenerator(
            api_key=Secret.from_env_var("TOGETHER_API_KEY"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=10.0,
            max_retries=2,
            system_prompt="test-system-prompt",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.togetherai.generator.TogetherAIGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["TOGETHER_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "system_prompt": "test-system-prompt",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10.0,
                "max_retries": 2,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.togetherai.generator.TogetherAIGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["TOGETHER_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "system_prompt": None,
                "api_base_url": "test-base-url",
                "timeout": None,
                "max_retries": None,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = TogetherAIGenerator.from_dict(data)
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("TOGETHER_API_KEY")
        assert component.system_prompt is None
        assert component.timeout is None
        assert component.max_retries is None

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.togetherai.generator.TogetherAIGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["TOGETHER_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "api_base_url": "test-base-url",
                "system_prompt": None,
                "timeout": None,
                "max_retries": None,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            TogetherAIGenerator.from_dict(data)

    def test_run(self, mock_chat_completion):
        component = TogetherAIGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(prompt="What's Natural Language Processing?")

        # Verify the mock was called
        mock_chat_completion.assert_called_once()

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, str) for reply in response["replies"]]

    def test_run_with_params(self, mock_chat_completion):
        component = TogetherAIGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(prompt="What's Natural Language Processing?")

        # check that the component calls the Together AI API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, str) for reply in response["replies"]]

    @pytest.mark.skipif(
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        component = TogetherAIGenerator()
        results = component.run(prompt="What's the capital of France?")
        assert len(results["replies"]) == 1
        assert len(results["meta"]) == 1
        response: str = results["replies"][0]
        assert "Paris" in response

        metadata = results["meta"][0]
        assert "meta-llama/Llama-3.3-70B-Instruct-Turbo" in metadata["model"]
        assert metadata["finish_reason"] == "stop"

        assert "usage" in metadata
        assert "prompt_tokens" in metadata["usage"] and metadata["usage"]["prompt_tokens"] > 0
        assert "completion_tokens" in metadata["usage"] and metadata["usage"]["completion_tokens"] > 0
        assert "total_tokens" in metadata["usage"] and metadata["usage"]["total_tokens"] > 0

    def test_run_with_wrong_model(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = OpenAIError("Invalid model name")

        generator = TogetherAIGenerator(api_key=Secret.from_token("test-api-key"), model="something-obviously-wrong")

        generator.client = mock_client

        with pytest.raises(OpenAIError):
            generator.run(prompt="Whatever")

    @pytest.mark.skipif(
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_with_system_prompt(self):
        generator = TogetherAIGenerator(
            model="openai/gpt-oss-120b", system_prompt="Answer in Italian using only one word."
        )
        result = generator.run(prompt="What's the capital of Italy?")
        assert "roma" in result["replies"][0].lower()

    @pytest.mark.skipif(
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming_with_include_usage(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = TogetherAIGenerator(
            streaming_callback=callback, generation_kwargs={"stream_options": {"include_usage": True}}
        )
        results = component.run(prompt="What's the capital of France?")

        # Basic response validation
        assert len(results["replies"]) == 1
        assert len(results["meta"]) == 1
        response: str = results["replies"][0]
        assert "Paris" in response

        # Metadata validation
        metadata = results["meta"][0]
        assert "meta-llama/Llama-3.3-70B-Instruct-Turbo" in metadata["model"]
        assert metadata["finish_reason"] == "stop"

        # Basic usage validation
        assert isinstance(metadata.get("usage"), dict), "meta.usage not a dict"
        usage = metadata["usage"]
        assert "prompt_tokens" in usage and usage["prompt_tokens"] > 0
        assert "completion_tokens" in usage and usage["completion_tokens"] > 0

        # Streaming callback validation
        assert callback.counter > 1
        assert "Paris" in callback.responses
