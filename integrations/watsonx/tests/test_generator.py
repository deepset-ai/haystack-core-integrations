# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import logging
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

from haystack_integrations.components.generators.watsonx.chat.chat_generator import WatsonxChatGenerator
from haystack_integrations.components.generators.watsonx.generator import WatsonxGenerator

logger = logging.getLogger(__name__)


class TestWatsonxGenerator:
    @pytest.fixture
    def mock_watsonx(self, monkeypatch):
        """Fixture for setting up common mocks"""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-api-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "fake-project-id")

        with patch(
            "haystack_integrations.components.generators.watsonx.chat.chat_generator.ModelInference"
        ) as mock_model:
            mock_model_instance = MagicMock()
            mock_model_instance.chat = MagicMock(
                return_value={
                    "choices": [
                        {
                            "message": {"content": "This is a generated response", "role": "assistant"},
                            "finish_reason": "completed",
                            "index": 0,
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                }
            )
            mock_model_instance.achat = AsyncMock(
                return_value={
                    "choices": [
                        {
                            "message": {"content": "Async generated response", "role": "assistant"},
                            "finish_reason": "completed",
                            "index": 0,
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                }
            )
            mock_model_instance.chat_stream = MagicMock(
                return_value=[
                    {"choices": [{"delta": {"content": "Streaming"}, "finish_reason": None, "index": 0}]},
                    {"choices": [{"delta": {"content": " response"}, "finish_reason": "completed", "index": 0}]},
                ]
            )

            class MockAsyncGenerator:
                def __init__(self):
                    self._count = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    self._count += 1
                    if self._count == 1:
                        return {"choices": [{"delta": {"content": "Async streaming"}, "finish_reason": None}]}
                    elif self._count == 2:
                        return {"choices": [{"delta": {"content": " response"}, "finish_reason": "completed"}]}
                    else:
                        raise StopAsyncIteration

            async def mock_achat_stream(*args, **kwargs):
                return MockAsyncGenerator()

            mock_model_instance.achat_stream = mock_achat_stream
            mock_model.return_value = mock_model_instance

            yield {"model": mock_model, "model_instance": mock_model_instance}

    def test_init_default(self, mock_watsonx):
        generator = WatsonxGenerator(project_id=Secret.from_token("fake-project-id"))

        _, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-3-3-8b-instruct"
        assert kwargs["project_id"] == "fake-project-id"
        assert kwargs["verify"] is None

        assert generator.model == "ibm/granite-3-3-8b-instruct"
        assert isinstance(generator.project_id, Secret)
        assert generator.project_id.resolve_value() == "fake-project-id"
        assert generator.api_base_url == "https://us-south.ml.cloud.ibm.com"

    def test_init_with_all_params(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
            api_base_url="https://custom-url.com",
            generation_kwargs={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
            verify=False,
        )

        _, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-3-3-8b-instruct"
        assert kwargs["project_id"] == "test-project"
        assert kwargs["verify"] is False

        assert isinstance(generator.project_id, Secret)
        assert generator.project_id.resolve_value() == "test-project"

    def test_init_fails_without_project_id(self, mock_watsonx):
        os.environ.pop("WATSONX_PROJECT_ID", None)
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
            WatsonxGenerator(api_key=Secret.from_token("test-api-key"))

    def test_to_dict(self, mock_watsonx):
        generator = WatsonxGenerator(
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            generation_kwargs={"max_tokens": 100},
        )
        data = generator.to_dict()
        expected = {
            "type": "haystack_integrations.components.generators.watsonx.generator.WatsonxGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-3-3-8b-instruct",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "api_base_url": "https://us-south.ml.cloud.ibm.com",
                "generation_kwargs": {"max_tokens": 100},
                "verify": None,
                "system_prompt": None,
                "timeout": None,
                "max_retries": None,
                "streaming_callback": None,
            },
        }
        assert data == expected

    def test_from_dict(self, mock_watsonx):
        data = {
            "type": "haystack_integrations.components.generators.watsonx.generator.WatsonxGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-3-3-8b-instruct",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "api_base_url": "https://us-south.ml.cloud.ibm.com",
                "generation_kwargs": {"max_tokens": 100},
                "verify": None,
                "system_prompt": None,
                "timeout": None,
                "max_retries": None,
                "streaming_callback": None,
            },
        }

        generator = WatsonxGenerator.from_dict(data)
        assert generator.api_key == Secret.from_env_var("WATSONX_API_KEY")
        assert generator.model == "ibm/granite-3-3-8b-instruct"
        assert generator.project_id == Secret.from_env_var("WATSONX_PROJECT_ID")
        assert generator.api_base_url == "https://us-south.ml.cloud.ibm.com"
        assert generator.generation_kwargs == {"max_tokens": 100}
        assert generator.verify is None
        assert generator.system_prompt is None
        assert generator.timeout is None
        assert generator.max_retries is None
        assert generator.streaming_callback is None

    def test_run_with_prompt_only(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
        )

        result = generator.run(prompt="Test prompt")

        assert "replies" in result
        assert "meta" in result
        assert len(result["replies"]) == 1
        assert len(result["meta"]) == 1
        assert result["replies"][0] == "This is a generated response"
        assert result["meta"][0]["finish_reason"] == "completed"
        assert "usage" in result["meta"][0]

        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}], params={}
        )

    def test_run_with_system_prompt(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
        )

        result = generator.run(prompt="Test prompt", system_prompt="You are a helpful assistant.")

        assert len(result["replies"]) == 1
        assert result["replies"][0] == "This is a generated response"

        expected_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test prompt"},
        ]
        mock_watsonx["model_instance"].chat.assert_called_once_with(messages=expected_messages, params={})

    def test_run_with_generation_kwargs(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
            generation_kwargs={"max_tokens": 100, "temperature": 0.7},
        )

        result = generator.run(prompt="Test prompt", generation_kwargs={"top_p": 0.9})

        assert len(result["replies"]) == 1
        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}],
            params={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
        )

    def test_run_with_streaming(self, mock_watsonx):
        generator = WatsonxGenerator(model="ibm/granite-13b-instruct-v2", project_id=Secret.from_token("test-project"))

        mock_callback = MagicMock()
        result = generator.run(prompt="Test prompt", streaming_callback=mock_callback)

        assert mock_callback.call_count == 2

        first_call_arg = mock_callback.call_args_list[0].args[0]
        assert isinstance(first_call_arg, StreamingChunk)
        assert first_call_arg.content == "Streaming"

        assert len(result["replies"]) == 1
        assert result["replies"][0] == "Streaming response"

    def test_prepare_messages_user_only(self, mock_watsonx):
        generator = WatsonxGenerator(project_id=Secret.from_token("test-project"))

        messages = generator._prepare_messages("Hello world")

        assert len(messages) == 1
        assert messages[0].role.value == "user"
        assert messages[0].text == "Hello world"

    def test_prepare_messages_with_system(self, mock_watsonx):
        generator = WatsonxGenerator(project_id=Secret.from_token("test-project"))

        messages = generator._prepare_messages("Hello world", "You are helpful")

        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert messages[0].text == "You are helpful"
        assert messages[1].role.value == "user"
        assert messages[1].text == "Hello world"

    def test_convert_chat_response_to_generator_format(self, mock_watsonx):
        generator = WatsonxGenerator(project_id=Secret.from_token("test-project"))
        chat_response = {
            "replies": [
                ChatMessage.from_assistant("First response", meta={"model": "test-model", "tokens": 10}),
                ChatMessage.from_assistant("Second response", meta={"model": "test-model", "tokens": 15}),
            ]
        }
        result = generator._convert_chat_response_to_generator_format(chat_response["replies"])
        assert "replies" in result
        assert "meta" in result
        assert len(result["replies"]) == 2
        assert len(result["meta"]) == 2
        assert result["replies"][0] == "First response"
        assert result["replies"][1] == "Second response"
        assert result["meta"][0]["tokens"] == 10
        assert result["meta"][1]["tokens"] == 15

    def test_convert_empty_chat_response(self, mock_watsonx):
        generator = WatsonxGenerator(project_id=Secret.from_token("test-project"))
        result = generator._convert_chat_response_to_generator_format([])
        assert result["replies"] == []
        assert result["meta"] == []

    @pytest.mark.asyncio
    async def test_run_async_with_prompt_only(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
        )

        result = await generator.run_async(prompt="Test prompt")

        assert "replies" in result
        assert "meta" in result
        assert len(result["replies"]) == 1
        assert len(result["meta"]) == 1
        assert result["replies"][0] == "Async generated response"
        assert result["meta"][0]["finish_reason"] == "completed"

        mock_watsonx["model_instance"].achat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}], params={}
        )

    @pytest.mark.asyncio
    async def test_run_async_with_system_prompt(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
        )

        result = await generator.run_async(prompt="Test prompt", system_prompt="You are a helpful assistant.")

        assert len(result["replies"]) == 1
        assert result["replies"][0] == "Async generated response"

        expected_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test prompt"},
        ]
        mock_watsonx["model_instance"].achat.assert_called_once_with(messages=expected_messages, params={})

    @pytest.mark.asyncio
    async def test_run_async_streaming(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
        )

        mock_callback = AsyncMock()
        result = await generator.run_async(prompt="Test prompt", streaming_callback=mock_callback)

        assert mock_callback.await_count == 2

        first_chunk = mock_callback.call_args_list[0].args[0]
        assert isinstance(first_chunk, StreamingChunk)
        assert first_chunk.content == "Async streaming"

        assert len(result["replies"]) == 1
        assert result["replies"][0] == "Async streaming response"

    def test_inheritance_from_chat_generator(self, mock_watsonx):
        """Test that WatsonxGenerator properly inherits from WatsonxChatGenerator"""

        generator = WatsonxGenerator(project_id=Secret.from_token("test-project"))

        assert isinstance(generator, WatsonxChatGenerator)

        assert hasattr(generator, "_initialize_client")
        assert hasattr(generator, "_prepare_api_call")
        assert hasattr(generator, "_handle_streaming")
        assert hasattr(generator, "_handle_standard")


@pytest.mark.integration
class TestWatsonxGeneratorIntegration:
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run(self):
        generator = WatsonxGenerator(
            model="ibm/granite-3-3-8b-instruct",
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            generation_kwargs={"max_tokens": 50, "temperature": 0.7, "top_p": 0.9},
        )

        result = generator.run(
            prompt="What's the capital of France? Answer concisely.",
            system_prompt="You are a geography expert. Answer concisely.",
        )

        assert isinstance(result, dict)
        assert "replies" in result
        assert "meta" in result
        assert isinstance(result["replies"], list)
        assert isinstance(result["meta"], list)
        assert len(result["replies"]) == 1
        assert len(result["meta"]) == 1
        assert isinstance(result["replies"][0], str)
        assert len(result["replies"][0]) > 0
        assert isinstance(result["meta"][0], dict)

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run_streaming(self):
        generator = WatsonxGenerator(
            model="ibm/granite-3-3-8b-instruct", project_id=Secret.from_env_var("WATSONX_PROJECT_ID")
        )

        collected_chunks = []

        def callback(chunk: StreamingChunk):
            collected_chunks.append(chunk)

        result = generator.run(prompt="What is the capital of France? Answer concisely.", streaming_callback=callback)

        assert isinstance(result, dict)
        assert "replies" in result
        assert "meta" in result
        assert len(result["replies"]) == 1
        assert isinstance(result["replies"][0], str)
        assert len(collected_chunks) > 0
        assert all(isinstance(chunk, StreamingChunk) for chunk in collected_chunks)

        complete_text = "".join(chunk.content for chunk in collected_chunks)
        assert result["replies"][0] == complete_text

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    async def test_live_run_async(self):
        generator = WatsonxGenerator(
            model="ibm/granite-3-3-8b-instruct", project_id=Secret.from_env_var("WATSONX_PROJECT_ID")
        )

        result = await generator.run_async(prompt="What's the capital of Germany? Answer concisely.")

        assert isinstance(result, dict)
        assert "replies" in result
        assert "meta" in result
        assert isinstance(result["replies"], list)
        assert isinstance(result["meta"], list)
        assert len(result["replies"]) == 1
        assert len(result["meta"]) == 1
        assert isinstance(result["replies"][0], str)
        assert len(result["replies"][0]) > 0
        assert isinstance(result["meta"][0], dict)
