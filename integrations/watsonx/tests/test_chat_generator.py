# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import logging
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, ComponentInfo, ImageContent, StreamingChunk
from haystack.tools import Tool, Toolset
from haystack.utils import Secret

from haystack_integrations.components.generators.watsonx.chat.chat_generator import WatsonxChatGenerator

logger = logging.getLogger(__name__)


def weather(city: str):
    """Get weather information for a city."""
    return f"Weather in {city}: 22°C, sunny"


def population(city: str) -> str:
    return f"The population of {city} is 2.2 million"


@pytest.fixture
def tools():
    return [
        Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=weather,
        )
    ]


class TestWatsonxChatGenerator:
    @pytest.fixture
    def mock_watsonx(self, monkeypatch) -> Generator[dict[str, AsyncMock | MagicMock], None]:
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
                            "finish_reason": "stop",
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
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                }
            )
            mock_model_instance.chat_stream = MagicMock(
                return_value=[
                    {"choices": [{"delta": {"content": "Streaming"}, "index": 0, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": " response"}, "index": 0, "finish_reason": "stop"}]},
                ]
            )

            async def mock_achat_stream(messages=None, params=None, tools=None):
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
                                "choices": [{"delta": {"content": " response"}, "finish_reason": "stop", "index": 0}]
                            }
                        else:
                            raise StopAsyncIteration

                return MockAsyncGenerator()

            mock_model_instance.achat_stream = mock_achat_stream
            mock_model.return_value = mock_model_instance

            yield {"model": mock_model, "model_instance": mock_model_instance, "select_callback": mock_select_callback}

    def test_supported_models(self) -> None:
        """SUPPORTED_MODELS is a non-empty list of strings."""
        models = WatsonxChatGenerator.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, mock_watsonx):
        generator = WatsonxChatGenerator(project_id=Secret.from_token("fake-project-id"))

        _, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-4-h-small"
        assert kwargs["project_id"] == "fake-project-id"
        assert kwargs["verify"] is None

        assert generator.model == "ibm/granite-4-h-small"
        assert isinstance(generator.project_id, Secret)
        assert generator.project_id.resolve_value() == "fake-project-id"
        assert generator.api_base_url == "https://us-south.ml.cloud.ibm.com"
        assert generator.tools is None

    def test_init_with_all_params(self, mock_watsonx: dict[str, AsyncMock | MagicMock]) -> None:
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=weather)

        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
            api_base_url="https://custom-url.com",
            generation_kwargs={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
            verify=False,
            tools=[tool],
        )

        _, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-4-h-small"
        assert kwargs["project_id"] == "test-project"
        assert kwargs["verify"] is False

        assert isinstance(generator.project_id, Secret)
        assert generator.project_id.resolve_value() == "test-project"
        assert generator.tools == [tool]

    def test_init_with_toolset(self, mock_watsonx: dict[str, AsyncMock | MagicMock], tools: list[Tool]) -> None:
        toolset = Toolset(tools)
        generator = WatsonxChatGenerator(project_id=Secret.from_token("fake-project-id"), tools=toolset)
        assert generator.tools == toolset

    def test_init_fails_without_project(self, mock_watsonx):
        os.environ.pop("WATSONX_PROJECT_ID", None)

        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
            WatsonxChatGenerator(api_key=Secret.from_token("test-api-key"))

    def test_to_dict(self, mock_watsonx: dict[str, AsyncMock | MagicMock]) -> None:
        generator = WatsonxChatGenerator(
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"), generation_kwargs={"max_tokens": 100}
        )

        data = generator.to_dict()

        expected = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-4-h-small",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "api_base_url": "https://us-south.ml.cloud.ibm.com",
                "generation_kwargs": {"max_tokens": 100},
                "verify": None,
                "timeout": None,
                "max_retries": None,
                "streaming_callback": None,
                "tools": None,
            },
        }
        assert data == expected

    def test_to_dict_with_params(self, mock_watsonx: dict[str, AsyncMock | MagicMock], tools: list[Tool]) -> None:
        generator = WatsonxChatGenerator(
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            generation_kwargs={"max_tokens": 100},
            streaming_callback=print_streaming_chunk,
            tools=tools,
        )

        data = generator.to_dict()

        expected = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-4-h-small",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "api_base_url": "https://us-south.ml.cloud.ibm.com",
                "generation_kwargs": {"max_tokens": 100},
                "verify": None,
                "timeout": None,
                "max_retries": None,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "tools": [
                    {
                        "data": {
                            "description": "useful to determine the weather in a given location",
                            "function": "tests.test_chat_generator.weather",
                            "inputs_from_state": None,
                            "name": "weather",
                            "outputs_to_state": None,
                            "outputs_to_string": None,
                            "parameters": {
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "type": "object",
                            },
                        },
                        "type": "haystack.tools.tool.Tool",
                    },
                ],
            },
        }
        assert data == expected

    def test_from_dict(self, mock_watsonx):
        assert mock_watsonx is not None
        data = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-4-h-small",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "generation_kwargs": {"max_tokens": 100},
            },
        }

        generator = WatsonxChatGenerator.from_dict(data)
        assert generator.model == "ibm/granite-4-h-small"
        assert isinstance(generator.project_id, Secret)
        assert generator.project_id.resolve_value() == "fake-project-id"
        assert generator.generation_kwargs == {"max_tokens": 100}

    def test_from_dict_with_callback(self, mock_watsonx):
        callback_str = "haystack.components.generators.utils.print_streaming_chunk"
        data = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-4-h-small",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "streaming_callback": callback_str,
            },
        }

        generator = WatsonxChatGenerator.from_dict(data)
        assert generator.streaming_callback is print_streaming_chunk

    def test_from_dict_with_tools(self, mock_watsonx: dict[str, AsyncMock | MagicMock], tools: list[Tool]) -> None:
        data = {
            "type": "haystack_integrations.components.generators.watsonx.chat.chat_generator.WatsonxChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-4-h-small",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "tools": [
                    {
                        "data": {
                            "description": "useful to determine the weather in a given location",
                            "function": "tests.test_chat_generator.weather",
                            "inputs_from_state": None,
                            "name": "weather",
                            "outputs_to_state": None,
                            "outputs_to_string": None,
                            "parameters": {
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "type": "object",
                            },
                        },
                        "type": "haystack.tools.tool.Tool",
                    },
                ],
            },
        }

        generator = WatsonxChatGenerator.from_dict(data)
        assert isinstance(generator.tools, list)
        assert len(generator.tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in generator.tools)

    def test_run_single_message(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = generator.run(messages=messages)

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "This is a generated response"
        assert result["replies"][0].meta["finish_reason"] == "stop"

        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}], params={}, tools=None
        )

    def test_run_with_generation_params(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
            generation_kwargs={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = generator.run(messages=messages)

        assert len(result["replies"]) == 1
        mock_watsonx["model_instance"].chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}],
            params={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
            tools=None,
        )

    def test_run_with_streaming(self, mock_watsonx):
        """Test streaming with callback through parent class"""
        generator = WatsonxChatGenerator(project_id=Secret.from_token("test-project"))

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
        assert result["replies"][0].meta["finish_reason"] == "stop"

    def test_run_with_empty_messages(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            project_id=Secret.from_token("test-project"),
        )

        result = generator.run(messages=[])
        assert result["replies"] == []

    def test_init_with_streaming_callback(self, mock_watsonx):
        def custom_callback(chunk: StreamingChunk):
            pass

        generator = WatsonxChatGenerator(
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
            project_id=Secret.from_token("test-project"),
        )

        messages = [ChatMessage.from_user("Test prompt")]
        result = await generator.run_async(messages=messages)

        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "Async generated response"
        assert result["replies"][0].meta["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_run_async_streaming(self, mock_watsonx):
        generator = WatsonxChatGenerator(
            api_key=Secret.from_token("test-api-key"),
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

    # Multimodal Tests
    def test_prepare_api_call_with_image(self, mock_watsonx):
        """Test that a ChatMessage with ImageContent is converted to WatsonX format correctly."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        generator = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_token("test-project"),
        )

        api_call = generator._prepare_api_call(messages=[message])

        assert "messages" in api_call
        assert len(api_call["messages"]) == 1

        watsonx_message = api_call["messages"][0]
        assert watsonx_message["role"] == "user"
        assert isinstance(watsonx_message["content"], list)
        assert len(watsonx_message["content"]) == 2

        # Check text content
        assert watsonx_message["content"][0]["type"] == "text"
        assert watsonx_message["content"][0]["text"] == "What's in this image?"

        # Check image content
        assert watsonx_message["content"][1]["type"] == "image_url"
        assert "image_url" in watsonx_message["content"][1]
        assert "url" in watsonx_message["content"][1]["image_url"]
        assert watsonx_message["content"][1]["image_url"]["url"] == f"data:image/png;base64,{base64_image}"

    def test_prepare_api_call_with_unsupported_mime_type(self, mock_watsonx):
        """Test that a ChatMessage with unsupported mime type raises ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/bmp")
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        generator = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_token("test-project"),
        )

        with pytest.raises(ValueError, match="Unsupported image format: image/bmp"):
            generator._prepare_api_call(messages=[message])

    def test_prepare_api_call_with_none_mime_type(self, mock_watsonx):
        """Test that a ChatMessage with None mime type raises ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        # Manually set mime_type to None to test the edge case
        image_content.mime_type = None
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        generator = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_token("test-project"),
        )

        with pytest.raises(ValueError, match="Unsupported image format: None"):
            generator._prepare_api_call(messages=[message])

    def test_prepare_api_call_image_in_non_user_message(self, mock_watsonx):
        """Test that images in non-user messages raise ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        # Create assistant message with image (should fail)
        message = ChatMessage.from_assistant(text="Here's an image.")
        message._content = [image_content]  # Manually add image to assistant message

        generator = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_token("test-project"),
        )

        with pytest.raises(ValueError, match="Image content is only supported for user messages"):
            generator._prepare_api_call(messages=[message])

    def test_convert_chunk_to_streaming_chunk_real_example(
        self, mock_watsonx: dict[str, AsyncMock | MagicMock]
    ) -> None:
        component = WatsonxChatGenerator(
            project_id=Secret.from_token("test-project"), model="meta-llama/llama-3-2-11b-vision-instruct"
        )
        component_info = ComponentInfo.from_component(component)

        # Chunk 1: Text only
        chunk1 = {
            "id": "chatcmpl-21e72dd9-ed65-49cc-9ea2-64d971707cda---2dedc26eab5af753744ed4eaa116a197---e0399d75-cd8c-486e-b907-dc211cb70eac",  # noqa: E501
            "object": "chat.completion.chunk",
            "model_id": "meta-llama/llama-3-2-11b-vision-instruct",
            "model": "meta-llama/llama-3-2-11b-vision-instruct",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"content": "I'll get the weather information for Paris and Berlin"},
                }
            ],
            "created": 1773250972,
            "model_version": "3.2.0",
            "created_at": "2026-03-11T17:42:52.921Z",
        }

        streaming_chunk1 = component._convert_chunk_to_streaming_chunk(chunk=chunk1, component_info=component_info)
        assert streaming_chunk1.content == "I'll get the weather information for Paris and Berlin"
        assert streaming_chunk1.tool_calls is None
        assert streaming_chunk1.finish_reason is None
        assert streaming_chunk1.index == 0
        assert "created" in streaming_chunk1.meta
        assert "created_at" in streaming_chunk1.meta
        assert "received_at" in streaming_chunk1.meta
        assert streaming_chunk1.meta["model"] == "meta-llama/llama-3-2-11b-vision-instruct"
        assert streaming_chunk1.meta["model_id"] == "meta-llama/llama-3-2-11b-vision-instruct"
        assert streaming_chunk1.meta["model_version"] == "3.2.0"
        assert streaming_chunk1.component_info == component_info

        # Chunk 2: Text only
        chunk2 = {
            "id": "chatcmpl-21e72dd9-ed65-49cc-9ea2-64d971707cda---2dedc26eab5af753744ed4eaa116a197---e0399d75-cd8c-486e-b907-dc211cb70eac",  # noqa: E501
            "object": "chat.completion.chunk",
            "model_id": "meta-llama/llama-3-2-11b-vision-instruct",
            "model": "meta-llama/llama-3-2-11b-vision-instruct",
            "choices": [
                {"index": 0, "finish_reason": None, "delta": {"content": " and present it in a structured format."}}
            ],
            "created": 1773250972,
            "model_version": "3.2.0",
            "created_at": "2026-03-11T17:42:52.929Z",
        }

        streaming_chunk2 = component._convert_chunk_to_streaming_chunk(chunk=chunk2, component_info=component_info)
        assert streaming_chunk2.content == " and present it in a structured format."
        assert streaming_chunk2.tool_calls is None
        assert streaming_chunk2.finish_reason is None
        assert streaming_chunk2.index == 0
        assert "created" in streaming_chunk2.meta
        assert "created_at" in streaming_chunk2.meta
        assert "received_at" in streaming_chunk2.meta
        assert streaming_chunk2.meta["model"] == "meta-llama/llama-3-2-11b-vision-instruct"
        assert streaming_chunk2.meta["model_id"] == "meta-llama/llama-3-2-11b-vision-instruct"
        assert streaming_chunk2.meta["model_version"] == "3.2.0"
        assert streaming_chunk2.component_info == component_info

        # Chunk 3: Multiple tool calls (6 function calls) for 2 cities with 3 tools each
        chunk3 = {
            "id": "chatcmpl-6b615ca6-4aa7-4f79-832f-bedce4641c2b---87fdc1a1cd2032ff0c6776ecfc20b6a5---34576777-949d-4df1-b95f-56d14b848eca",  # noqa: E501
            "object": "chat.completion.chunk",
            "model_id": "meta-llama/llama-3-2-11b-vision-instruct",
            "model": "meta-llama/llama-3-2-11b-vision-instruct",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "chatcmpl-tool-9646185282a54afc86c3572513b2dafa",
                                "type": "function",
                                "function": {"name": "weather", "arguments": ""},
                            }
                        ]
                    },
                }
            ],
            "created": 1773252289,
            "model_version": "3.2.0",
            "created_at": "2026-03-11T18:04:49.696Z",
        }

        streaming_chunk3 = component._convert_chunk_to_streaming_chunk(chunk=chunk3, component_info=component_info)
        assert streaming_chunk3.content == ""
        assert streaming_chunk3.tool_calls is not None
        assert len(streaming_chunk3.tool_calls) == 1
        assert streaming_chunk3.finish_reason is None
        assert streaming_chunk3.index == 0
        assert "created" in streaming_chunk3.meta
        assert "created_at" in streaming_chunk3.meta
        assert "received_at" in streaming_chunk3.meta
        assert streaming_chunk3.meta["model"] == "meta-llama/llama-3-2-11b-vision-instruct"
        assert streaming_chunk3.meta["model_id"] == "meta-llama/llama-3-2-11b-vision-instruct"
        assert streaming_chunk3.meta["model_version"] == "3.2.0"
        assert streaming_chunk3.component_info == component_info

        assert streaming_chunk3.tool_calls[0].tool_name == "weather"
        assert streaming_chunk3.tool_calls[0].arguments == ""
        assert streaming_chunk3.tool_calls[0].id == "chatcmpl-tool-9646185282a54afc86c3572513b2dafa"
        assert streaming_chunk3.tool_calls[0].index == 0

    def test_multimodal_message_processing(self, mock_watsonx):
        """Test multimodal message processing with mocked model."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

        generator = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_token("test-project"),
        )

        result = generator.run(messages=messages)

        # Verify the multimodal message was processed correctly
        assert "replies" in result
        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "This is a generated response"

        # Verify the model was called with the correct multimodal format
        mock_watsonx["model_instance"].chat.assert_called_once()
        call_args = mock_watsonx["model_instance"].chat.call_args
        messages_arg = call_args[1]["messages"]

        assert len(messages_arg) == 1
        assert messages_arg[0]["role"] == "user"
        assert isinstance(messages_arg[0]["content"], list)
        assert len(messages_arg[0]["content"]) == 2
        assert messages_arg[0]["content"][0]["type"] == "text"
        assert messages_arg[0]["content"][1]["type"] == "image_url"

    def test_supported_image_formats(self, mock_watsonx):
        """Test that all supported image formats work correctly."""
        supported_formats = ["image/jpeg", "image/png"]
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

        generator = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_token("test-project"),
        )

        for mime_type in supported_formats:
            image_content = ImageContent(base64_image=base64_image, mime_type=mime_type)
            message = ChatMessage.from_user(content_parts=["Test image", image_content])

            # Should not raise any exception
            api_call = generator._prepare_api_call(messages=[message])
            assert api_call is not None

    def test_multiple_images_in_single_message(self, mock_watsonx):
        """Test handling multiple images in a single message."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image1 = ImageContent(base64_image=base64_image, mime_type="image/png")
        image2 = ImageContent(base64_image=base64_image, mime_type="image/jpeg")

        message = ChatMessage.from_user(content_parts=["Compare these images:", image1, image2])

        generator = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_token("test-project"),
        )

        api_call = generator._prepare_api_call(messages=[message])

        assert "messages" in api_call
        watsonx_message = api_call["messages"][0]
        assert len(watsonx_message["content"]) == 3  # 1 text + 2 images
        assert watsonx_message["content"][0]["type"] == "text"
        assert watsonx_message["content"][1]["type"] == "image_url"
        assert watsonx_message["content"][2]["type"] == "image_url"


@pytest.mark.integration
class TestWatsonxChatGeneratorIntegration:
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run(self):
        generator = WatsonxChatGenerator(
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
    def test_live_run_with_toolset(self, tools: list[Tool]) -> None:
        """Test that WatsonxChatGenerator can run with a Toolset."""
        toolset = Toolset(tools)
        generator = WatsonxChatGenerator(
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            generation_kwargs={"max_tokens": 50, "temperature": 0.7, "top_p": 0.9},
            tools=toolset,
        )
        messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Check if tool calls were made
        assert message.tool_calls is not None, "Message has no tool calls"
        assert len(message.tool_calls) == 1, "Message has multiple tool calls and it should only have one"
        tool_call = message.tool_calls[0]
        assert message.meta["finish_reason"] == "tool_calls"

        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

        # Test full conversation with tool result
        tool_result_message = ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call)
        follow_up_messages = [*messages, message, tool_result_message]
        final_results = generator.run(messages=follow_up_messages)

        assert len(final_results["replies"]) == 1
        final_message = final_results["replies"][0]
        assert final_message.text
        assert "paris" in final_message.text.lower() or "weather" in final_message.text.lower(), (
            "Response does not contain Paris or weather"
        )

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run_streaming(self):
        generator = WatsonxChatGenerator(project_id=Secret.from_env_var("WATSONX_PROJECT_ID"))
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

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run_with_tools_streaming(self, tools: list[Tool]) -> None:
        """
        Integration test that the WatsonxChatGenerator component can run with tools and streaming.
        """
        component = WatsonxChatGenerator(
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"), tools=tools, streaming_callback=print_streaming_chunk
        )
        results = component.run(messages=[ChatMessage.from_user("What's the weather like in Paris?")])

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert tool_message.tool_calls is not None, "Tool message has no tool calls"
        assert len(tool_message.tool_calls) == 1, "Tool message has multiple tool calls"
        assert tool_message.tool_calls[0].tool_name == "weather"
        assert tool_message.tool_calls[0].arguments == {"city": "Paris"}

        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"
        assert tool_message.meta["finish_reason"] == "tool_calls"

        tool_call = tool_message.tool_calls[0]
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run_with_mixed_tools(self) -> None:
        """
        Integration test that verifies WatsonxChatGenerator works with mixed Tool and Toolset.
        This tests that the LLM can correctly invoke tools from both a standalone Tool and a Toolset.
        """
        weather_tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for, e.g. Paris, London",
                    }
                },
                "required": ["city"],
            },
            function=weather,
        )

        population_tool = Tool(
            name="population",
            description="useful to determine the population of a given city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get population for, e.g. Paris, Berlin",
                    }
                },
                "required": ["city"],
            },
            function=population,
        )

        # Create a toolset with the population tool
        population_toolset = Toolset([population_tool])

        # Mix standalone tool with toolset
        mixed_tools = [weather_tool, population_toolset]

        initial_messages = [
            ChatMessage.from_user("What's the weather like in Paris and what is the population of Berlin?")
        ]
        component = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            tools=mixed_tools,
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        first_reply = results["replies"][0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert first_reply.tool_calls, "First reply has no tool calls"

        tool_calls = first_reply.tool_calls
        assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

        # Verify we got calls to both weather and population tools
        tool_names = {tc.tool_name for tc in tool_calls}
        assert "weather" in tool_names, "Expected 'weather' tool call"
        assert "population" in tool_names, "Expected 'population' tool call"

        # Verify tool call details
        for tool_call in tool_calls:
            assert tool_call.tool_name in ["weather", "population"]
            assert "city" in tool_call.arguments
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert first_reply.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = []
        for tool_call in tool_calls:
            if tool_call.tool_name == "weather":
                result = "The weather in Paris is sunny and 32°C"
            else:  # population
                result = "The population of Berlin is 2.2 million"
            tool_result_messages.append(ChatMessage.from_tool(tool_result=result, origin=tool_call))

        new_messages = [*initial_messages, first_reply, *tool_result_messages]
        results = component.run(messages=new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    async def test_live_run_async(self):
        generator = WatsonxChatGenerator(project_id=Secret.from_env_var("WATSONX_PROJECT_ID"))
        messages = [ChatMessage.from_user("What's the capital of Germany? Answer concisely.")]
        results = await generator.run_async(messages=messages)

        assert isinstance(results, dict)
        assert "replies" in results
        assert isinstance(results["replies"], list)
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        assert len(results["replies"][0].text) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    async def test_live_run_async_with_tools(self, tools: list[Tool]) -> None:
        """Test async version with tools."""
        component = WatsonxChatGenerator(project_id=Secret.from_env_var("WATSONX_PROJECT_ID"), tools=tools)
        results = await component.run_async(messages=[ChatMessage.from_user("What's the weather like in Paris?")])

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert tool_message.tool_calls is not None, "Tool message has no tool calls"
        assert len(tool_message.tool_calls) == 1, "Tool message has multiple tool calls"
        assert tool_message.tool_calls[0].tool_name == "weather"
        assert tool_message.tool_calls[0].arguments == {"city": "Paris"}
        assert tool_message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run_multimodal(self):
        generator = WatsonxChatGenerator(
            model="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
        )

        image_content = ImageContent.from_file_path("tests/test_files/apple.jpg")

        messages = [
            ChatMessage.from_user(
                content_parts=[
                    "What do you see in this image? Be concise.",
                    image_content,
                ]
            )
        ]

        results = generator.run(messages=messages)

        assert isinstance(results, dict)
        assert "replies" in results
        assert isinstance(results["replies"], list)
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        assert len(results["replies"][0].text) > 0
