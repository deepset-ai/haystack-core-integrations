# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import logging
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

from haystack_integrations.components.generators.watsonx.chat.chat_generator import WatsonxChatGenerator

logger = logging.getLogger(__name__)


class TestWatsonxChatGenerator:
    @pytest.fixture
    def mock_watsonx(self, monkeypatch):
        """Fixture for setting up common mocks"""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-api-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "fake-project-id")

        with (
            patch(
                "haystack_integrations.components.generators.watsonx.chat.chat_generator.ModelInference"
            ) as mock_model,
            patch(
                "haystack_integrations.components.generators.watsonx.chat.chat_generator.select_streaming_callback"
            ) as mock_select_callback,
        ):
            mock_select_callback.side_effect = lambda init_callback, runtime_callback, requires_async: (
                runtime_callback if runtime_callback is not None else init_callback
            )

            mock_model_instance = MagicMock()
            mock_model_instance.chat = MagicMock(
                return_value={
                    "choices": [
                        {
                            "message": {"content": "This is a generated response", "role": "assistant"},
                            "index": 0,
                            "finish_reason": "completed",
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
                            "index": 0,
                            "finish_reason": "completed",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                }
            )
            mock_model_instance.chat_stream = MagicMock(
                return_value=[
                    {"choices": [{"delta": {"content": "Streaming"}, "index": 0, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": " response"}, "index": 0, "finish_reason": "completed"}]},
                ]
            )

            async def mock_achat_stream(messages=None, params=None):
                class MockAsyncGenerator:
                    def __init__(self):
                        self._count = 0

                    def __aiter__(self):
                        return self

                    async def __anext__(self):
                        self._count += 1
                        if self._count == 1:
                            return {
                                "choices": [
                                    {"delta": {"content": "Async streaming"}, "finish_reason": None, "index": 0}
                                ]
                            }
                        elif self._count == 2:
                            return {
                                "choices": [
                                    {"delta": {"content": " response"}, "finish_reason": "completed", "index": 0}
                                ]
                            }
                        else:
                            raise StopAsyncIteration

                return MockAsyncGenerator()

            mock_model_instance.achat_stream = mock_achat_stream
            mock_model.return_value = mock_model_instance

            yield {"model": mock_model, "model_instance": mock_model_instance, "select_callback": mock_select_callback}

    def test_init_default(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct", project_id=Secret.from_token("fake-project-id")
        )

        _, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-3-2b-instruct"
        assert kwargs["project_id"] == "fake-project-id"
        assert kwargs["verify"] is None

        assert generator.model == "ibm/granite-3-2b-instruct"
        assert isinstance(generator.project_id, Secret)
        assert generator.project_id.resolve_value() == "fake-project-id"
        assert generator.api_base_url == "https://us-south.ml.cloud.ibm.com"

    def test_init_with_all_params(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
            api_base_url="https://custom-url.com",
            generation_kwargs={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
            verify=False,
        )

        _, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-3-2b-instruct"
        assert kwargs["project_id"] == "test-project"
        assert kwargs["verify"] is False

        assert isinstance(generator.project_id, Secret)
        assert generator.project_id.resolve_value() == "test-project"

    def test_init_fails_without_project(self, mock_watsonx):
        os.environ.pop("WATSONX_PROJECT_ID", None)

        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
            WatsonxChatGenerator(api_key=Secret.from_token("test-api-key"), model="ibm/granite-3-2b-instruct")

    def test_to_dict(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            generation_kwargs={"max_tokens": 100},
        )

        data = generator.to_dict()

        expected = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-3-2b-instruct",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "api_base_url": "https://us-south.ml.cloud.ibm.com",
                "generation_kwargs": {"max_tokens": 100},
                "verify": None,
                "timeout": None,
                "max_retries": None,
                "streaming_callback": None,
            },
        }
        assert data == expected

    def test_to_dict_with_params(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            generation_kwargs={"max_tokens": 100},
            streaming_callback=print_streaming_chunk,
        )

        data = generator.to_dict()

        expected = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-3-2b-instruct",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "api_base_url": "https://us-south.ml.cloud.ibm.com",
                "generation_kwargs": {"max_tokens": 100},
                "verify": None,
                "timeout": None,
                "max_retries": None,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            },
        }
        assert data == expected

    def test_from_dict(self, mock_watsonx):
        assert mock_watsonx is not None
        data = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-3-2b-instruct",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "generation_kwargs": {"max_tokens": 100},
            },
        }

        generator = WatsonxChatGenerator.from_dict(data)
        assert generator.model == "ibm/granite-3-2b-instruct"
        assert isinstance(generator.project_id, Secret)
        assert generator.project_id.resolve_value() == "fake-project-id"
        assert generator.generation_kwargs == {"max_tokens": 100}

    def test_from_dict_with_callback(self, mock_watsonx):
        callback_str = "haystack.components.generators.utils.print_streaming_chunk"
        data = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-3-2b-instruct",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "streaming_callback": callback_str,
            },
        }

        generator = WatsonxChatGenerator.from_dict(data)
        assert generator.streaming_callback is print_streaming_chunk

    def test_run_single_message(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = generator.run(messages=messages)

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "This is a generated response"
        assert result["replies"][0].meta["finish_reason"] == "completed"

        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}], params={}
        )

    def test_run_with_generation_params(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
            generation_kwargs={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = generator.run(messages=messages)

        assert len(result["replies"]) == 1
        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}],
            params={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
        )

    def test_run_with_streaming(self, mock_watsonx):
        """Test streaming with callback through parent class"""
        generator = WatsonxChatGenerator(
            model="ibm/granite-13b-instruct-v2", project_id=Secret.from_token("test-project")
        )

        mock_callback = MagicMock()
        messages = [ChatMessage.from_user("Test prompt")]
        result = generator.run(messages=messages, streaming_callback=mock_callback)

        mock_watsonx["select_callback"].assert_called_once_with(
            init_callback=None, runtime_callback=mock_callback, requires_async=False
        )

        assert mock_callback.call_count == 2

        first_call_arg = mock_callback.call_args_list[0].args[0]
        assert isinstance(first_call_arg, StreamingChunk)
        assert first_call_arg.content == "Streaming"

        second_call_arg = mock_callback.call_args_list[1].args[0]
        assert isinstance(second_call_arg, StreamingChunk)
        assert second_call_arg.content == " response"

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "Streaming response"
        assert result["replies"][0].meta["finish_reason"] == "completed"

    def test_run_with_empty_messages(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
        )

        result = generator.run(messages=[])
        assert result["replies"] == []

    def test_skips_tool_messages(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
        )

        messages = [ChatMessage.from_user("User message"), ChatMessage.from_tool("Tool result", "test-origin")]

        generator.run(messages=messages)

        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "User message"}], params={}
        )

    def test_init_with_streaming_callback(self, mock_watsonx):
        def custom_callback(chunk: StreamingChunk):
            pass

        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
            streaming_callback=custom_callback,
        )
        assert generator.streaming_callback is custom_callback

    def test_streaming_callback_priority(self, mock_watsonx):
        def init_callback(chunk: StreamingChunk):
            pass

        def run_callback(chunk: StreamingChunk):
            pass

        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
            streaming_callback=init_callback,
        )

        # Run with different callback - should use the runtime callback
        generator.run(messages=[ChatMessage.from_user("test")], streaming_callback=run_callback)

        mock_watsonx["select_callback"].assert_called_once_with(
            init_callback=init_callback, runtime_callback=run_callback, requires_async=False
        )

    @pytest.mark.asyncio
    async def test_run_async_single_message(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = await generator.run_async(messages=messages)

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "Async generated response"
        assert result["replies"][0].meta["finish_reason"] == "completed"

    @pytest.mark.asyncio
    async def test_run_async_streaming(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_token("test-project"),
        )
        received_chunks = []

        async def mock_callback(chunk: StreamingChunk):
            received_chunks.append(chunk)

        messages = [ChatMessage.from_user("Test prompt")]

        result = await generator.run_async(messages=messages, streaming_callback=mock_callback)
        mock_watsonx["select_callback"].assert_called_with(
            init_callback=None, runtime_callback=mock_callback, requires_async=True
        )
        assert len(received_chunks) == 2

        assert received_chunks[0].content == "Async streaming"
        assert received_chunks[1].content == " response"

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "Async streaming response"


@pytest.mark.integration
class TestWatsonxChatGeneratorIntegration:
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run(self):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct",
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            generation_kwargs={"max_tokens": 50, "temperature": 0.7, "top_p": 0.9},
        )
        messages = [ChatMessage.from_user("What's the capital of France? Answer concisely.")]
        results = generator.run(messages=messages)

        assert isinstance(results, dict)
        assert "replies" in results
        assert isinstance(results["replies"], list)
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        assert len(results["replies"][0].text) > 0
        assert isinstance(generator.project_id, Secret)

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run_streaming(self):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct", project_id=Secret.from_env_var("WATSONX_PROJECT_ID")
        )
        collected_chunks = []

        def callback(chunk: StreamingChunk):
            collected_chunks.append(chunk)

        messages = [ChatMessage.from_user("What's the capital of France? Answer concisely.")]
        results = generator.run(messages=messages, streaming_callback=callback)

        assert isinstance(results, dict)
        assert "replies" in results
        assert "chunks" not in results
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        assert len(collected_chunks) > 0
        assert all(isinstance(chunk, StreamingChunk) for chunk in collected_chunks)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    async def test_live_run_async(self):
        generator = WatsonxChatGenerator(
            model="ibm/granite-3-2b-instruct", project_id=Secret.from_env_var("WATSONX_PROJECT_ID")
        )
        messages = [ChatMessage.from_user("What's the capital of Germany? Answer concisely.")]
        results = await generator.run_async(messages=messages)

        assert isinstance(results, dict)
        assert "replies" in results
        assert isinstance(results["replies"], list)
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        assert len(results["replies"][0].text) > 0
