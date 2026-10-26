# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    ImageContent,
    ReasoningContent,
    StreamingChunk,
    ToolCall,
    ToolCallDelta,
)
from haystack.tools import Tool
from haystack.tools.toolset import Toolset
from haystack.utils.auth import Secret
from huggingface_hub import (
    ChatCompletionInputStreamOptions,
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputFunctionDefinition,
    ChatCompletionOutputMessage,
    ChatCompletionOutputToolCall,
    ChatCompletionOutputUsage,
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputChoice,
    ChatCompletionStreamOutputDelta,
)
from huggingface_hub.errors import RepositoryNotFoundError

from haystack_integrations.common.huggingface_api.utils import HFGenerationAPIType
from haystack_integrations.components.generators.huggingface_api.chat.chat_generator import (
    HuggingFaceAPIChatGenerator,
)


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant speaking A2 level of English"),
        ChatMessage.from_user("Tell me about Berlin"),
    ]


def get_weather(city: str) -> dict[str, Any]:
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(city, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


@pytest.fixture
def tools():
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=get_weather,
    )
    return [weather_tool]


@pytest.fixture
def mock_check_valid_model():
    with (
        patch(
            "haystack_integrations.components.generators.huggingface_api.chat.chat_generator._check_valid_model",
            MagicMock(return_value=None),
        ) as mock,
        patch(
            "haystack_integrations.components.generators.huggingface_api.chat.chat_generator._check_valid_model_async",
            AsyncMock(return_value=None),
        ),
    ):
        yield mock


@pytest.fixture
def chat_completion():
    return ChatCompletionOutput(
        choices=[
            ChatCompletionOutputComplete(
                finish_reason="eos_token",
                index=0,
                message=ChatCompletionOutputMessage(content="The capital of France is Paris.", role="assistant"),
            )
        ],
        id="some_id",
        model="some_model",
        system_fingerprint="some_fingerprint",
        usage=ChatCompletionOutputUsage(completion_tokens=8, prompt_tokens=17, total_tokens=25),
        created=1710498360,
    )


@pytest.fixture
def mock_chat_completion(chat_completion):
    with patch("huggingface_hub.InferenceClient.chat_completion", autospec=True) as mock:
        mock.return_value = chat_completion
        yield mock


@pytest.fixture
def mock_chat_completion_async(chat_completion):
    with patch("huggingface_hub.AsyncInferenceClient.chat_completion", autospec=True) as mock:
        mock.return_value = chat_completion
        yield mock


# used to test serialization of streaming_callback
def streaming_callback_handler(x):
    return x


@pytest.fixture
def hf_stream_chunk():
    def create(delta, finish_reason=None):
        return ChatCompletionStreamOutput.parse_obj_as_instance(
            {
                "id": "completion_id",
                "created": 1710498504,
                "model": "some_model",
                "system_fingerprint": "some_fingerprint",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    return create


@pytest.fixture
def recorded_tool_stream():
    events = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "function": {"name": "weather"},
                                "id": "chatcmpl-tool-88f8a30eee6f69d9",
                                "index": 0,
                                "type": "function",
                            }
                        ]
                    },
                    "index": 0,
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '{"'}, "index": 0}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "city"}, "index": 0}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '":'}, "index": 0}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": ' "'}, "index": 0}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "Paris"}, "index": 0}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '"}'}, "index": 0}]}, "index": 0}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "function": {"name": "weather"},
                                "id": "chatcmpl-tool-82e851da6ffc4a76",
                                "index": 1,
                                "type": "function",
                            }
                        ]
                    },
                    "index": 0,
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '{"'}, "index": 1}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "city"}, "index": 1}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '":'}, "index": 1}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": ' "'}, "index": 1}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "Berlin"}, "index": 1}]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '"}'}, "index": 1}]}, "index": 0}]},
        {"choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}]},
        {
            "choices": [],
            "usage": {"completion_tokens": 38, "prompt_tokens": 162, "total_tokens": 200},
            "system_fingerprint": "vllm-0.21.0-233fc06d",
        },
    ]
    return [
        ChatCompletionStreamOutput.parse_obj_as_instance(
            {
                "id": "chatcmpl-db742092-2040-4eb2-9d3c-7386d2312000",
                "created": 1792419567,
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                **event,
            }
        )
        for event in events
    ]


@pytest.fixture
def expected_recorded_tool_chunks():
    meta = {"model": "Qwen/Qwen3-4B-Instruct-2507", "received_at": ANY, "finish_reason": None}
    return [
        StreamingChunk(content="", meta=meta, component_info=ANY),
        StreamingChunk(
            content="",
            index=0,
            start=True,
            meta=meta,
            component_info=ANY,
            tool_calls=[ToolCallDelta(index=0, id="chatcmpl-tool-88f8a30eee6f69d9", tool_name="weather")],
        ),
        *[
            StreamingChunk(
                content="",
                index=0,
                meta=meta,
                component_info=ANY,
                tool_calls=[ToolCallDelta(index=0, arguments=fragment)],
            )
            for fragment in ['{"', "city", '":', ' "', "Paris", '"}']
        ],
        StreamingChunk(
            content="",
            index=1,
            start=True,
            meta=meta,
            component_info=ANY,
            tool_calls=[ToolCallDelta(index=1, id="chatcmpl-tool-82e851da6ffc4a76", tool_name="weather")],
        ),
        *[
            StreamingChunk(
                content="",
                index=1,
                meta=meta,
                component_info=ANY,
                tool_calls=[ToolCallDelta(index=1, arguments=fragment)],
            )
            for fragment in ['{"', "city", '":', ' "', "Berlin", '"}']
        ],
        StreamingChunk(
            content="", finish_reason="tool_calls", component_info=ANY, meta={**meta, "finish_reason": "tool_calls"}
        ),
        StreamingChunk(
            content="",
            component_info=ANY,
            meta={
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "received_at": ANY,
                "usage": {"prompt_tokens": 162, "completion_tokens": 38},
            },
        ),
    ]


async def _async_stream(chunks):
    for chunk in chunks:
        yield chunk


class TestInitializationAndSerialization:
    def test_init_invalid_api_type(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(api_type="invalid_api_type", api_params={})

    def test_init_serverless(self):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        generation_kwargs = {"temperature": 0.6}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": model},
            token=None,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.api_type == HFGenerationAPIType.SERVERLESS_INFERENCE_API
        assert generator.api_params == {"model": model}
        assert generator.generation_kwargs == {**generation_kwargs, "stop": ["stop"], "max_tokens": 512}
        assert generator.streaming_callback == streaming_callback
        assert generator.tools is None

        assert generator._client is None
        assert generator._async_client is None

    def test_init_serverless_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"param": "irrelevant"}
            )

    def test_init_tgi(self):
        url = "https://some_model.com"
        generation_kwargs = {"temperature": 0.6}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": url},
            token=None,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.api_type == HFGenerationAPIType.TEXT_GENERATION_INFERENCE
        assert generator.api_params == {"url": url}
        assert generator.generation_kwargs == {**generation_kwargs, "stop": ["stop"], "max_tokens": 512}
        assert generator.streaming_callback == streaming_callback
        assert generator.tools is None

        assert generator._client is None
        assert generator._async_client is None

    def test_init_tgi_invalid_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE, api_params={"url": "invalid_url"}
            )

    def test_init_tgi_no_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE, api_params={"param": "irrelevant"}
            )

    def test_init_fail_with_duplicate_tool_names(self, tools):
        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "irrelevant"},
                tools=duplicate_tools,
            )

    def test_to_dict(self):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            tools=[tool],
        )

        result = generator.to_dict()
        init_params = result["init_parameters"]

        assert init_params["api_type"] == "serverless_inference_api"
        assert init_params["api_params"] == {"model": "HuggingFaceH4/zephyr-7b-beta"}
        assert init_params["token"] == {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        assert init_params["generation_kwargs"] == {"temperature": 0.6, "stop": ["stop", "words"], "max_tokens": 512}
        assert init_params["streaming_callback"] is None

        # deserializing the serialized component must reproduce the original tool
        loaded = HuggingFaceAPIChatGenerator.from_dict(result)
        assert loaded.tools == [tool]

    def test_from_dict(self):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            tools=[tool],
        )
        result = generator.to_dict()

        # now deserialize, call from_dict
        generator_2 = HuggingFaceAPIChatGenerator.from_dict(result)
        assert generator_2.api_type == HFGenerationAPIType.SERVERLESS_INFERENCE_API
        assert generator_2.api_params == {"model": "HuggingFaceH4/zephyr-7b-beta"}
        assert generator_2.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert generator_2.generation_kwargs == {"temperature": 0.6, "stop": ["stop", "words"], "max_tokens": 512}
        assert generator_2.streaming_callback is None
        assert generator_2.tools == [tool]

    def test_serde_in_pipeline(self):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()

        # the Tool serialization format is owned by haystack-ai and varies across its versions; the
        # dumps/loads round-trip below covers the tools, so exclude them from the pinned-dict comparison
        tools_entries = pipeline_dict["components"]["generator"]["init_parameters"].pop("tools")
        assert len(tools_entries) == 1
        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack_integrations.components.generators.huggingface_api.chat.chat_generator"
                    ".HuggingFaceAPIChatGenerator",
                    "init_parameters": {
                        "api_type": "serverless_inference_api",
                        "api_params": {"model": "HuggingFaceH4/zephyr-7b-beta"},
                        "token": {"type": "env_var", "env_vars": ["ENV_VAR"], "strict": False},
                        "generation_kwargs": {"temperature": 0.6, "stop": ["stop", "words"], "max_tokens": 512},
                        "streaming_callback": None,
                    },
                }
            },
            "connections": [],
        }

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_HF_TOKEN", raising=False)
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_env_var("MISSING_HF_TOKEN"),
        )

        with pytest.raises(ValueError, match="MISSING_HF_TOKEN"):
            generator.warm_up()

    def test_invalid_model_is_checked_at_warm_up(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id", response=MagicMock())
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "invalid_model_id"}
        )
        with pytest.raises(RepositoryNotFoundError):
            generator.warm_up()

    @patch("haystack_integrations.components.generators.huggingface_api.chat.chat_generator.InferenceClient")
    def test_sync_lifecycle(self, mock_client_cls):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_token("test-token"),
        )
        client = mock_client_cls.return_value

        generator.warm_up()
        assert generator._client is client
        assert generator._async_client is None

        generator.close()
        client.close.assert_called_once_with()
        assert generator._client is None

        generator.warm_up()
        assert mock_client_cls.call_count == 2

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.generators.huggingface_api.chat.chat_generator.AsyncInferenceClient")
    async def test_async_lifecycle(self, mock_client_cls):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_token("test-token"),
        )
        client = MagicMock(close=AsyncMock())
        mock_client_cls.return_value = client

        await generator.warm_up_async()
        assert generator._async_client is client
        assert generator._client is None

        await generator.close_async()
        client.close.assert_awaited_once_with()
        assert generator._async_client is None

        await generator.warm_up_async()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.generators.huggingface_api.chat.chat_generator.InferenceClient")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        generator.warm_up()
        generator.warm_up()
        mock_client_cls.assert_called_once_with(model="https://example.com", token=None)

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.generators.huggingface_api.chat.chat_generator.AsyncInferenceClient")
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        await generator.warm_up_async()
        await generator.warm_up_async()
        mock_client_cls.assert_called_once_with(model="https://example.com", token=None)

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        generator.close()
        await generator.close_async()
        assert generator._client is None
        assert generator._async_client is None

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.generators.huggingface_api.chat.chat_generator.AsyncInferenceClient")
    @patch("haystack_integrations.components.generators.huggingface_api.chat.chat_generator.InferenceClient")
    async def test_close_and_close_async_are_independent(self, mock_sync_cls, mock_async_cls):
        sync_client = mock_sync_cls.return_value
        async_client = MagicMock(close=AsyncMock())
        mock_async_cls.return_value = async_client
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        generator.warm_up()
        await generator.warm_up_async()

        generator.close()
        assert generator._client is None
        assert generator._async_client is async_client
        async_client.close.assert_not_awaited()

        await generator.close_async()
        assert generator._async_client is None
        sync_client.close.assert_called_once_with()


class TestRun:
    def test_run(self, mock_check_valid_model, mock_chat_completion, chat_messages):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            streaming_callback=None,
        )

        response = generator.run(messages=chat_messages)

        # check kwargs passed to chat_completion
        _, kwargs = mock_chat_completion.call_args
        hf_messages = [
            {"role": "system", "content": "You are a helpful assistant speaking A2 level of English"},
            {"role": "user", "content": "Tell me about Berlin"},
        ]
        assert kwargs == {
            "temperature": 0.6,
            "stop": ["stop", "words"],
            "max_tokens": 512,
            "tools": None,
            "messages": hf_messages,
        }

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_generation_kwargs(self, mock_check_valid_model, mock_chat_completion, chat_messages):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            generation_kwargs={"max_tokens": 100, "temperature": 0.5},
        )

        generator.run(messages=chat_messages, generation_kwargs={"temperature": 0.9})

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.9

    def test_run_with_string_input(self, mock_check_valid_model, mock_chat_completion):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
        )
        response = generator.run("What's the capital of France?")

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "What's the capital of France?"}]

        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)

    def test_run_with_streaming_callback_in_run_method(
        self, mock_check_valid_model, mock_chat_completion, chat_messages
    ):
        streaming_call_count = 0

        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
        )

        # Create a fake streamed response
        # self needed here, don't remove
        def mock_iter(self):
            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content="The", role="assistant"),
                        index=0,
                        finish_reason=None,
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content=None, role=None), index=0, finish_reason="length"
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

        mock_response = Mock(__iter__=mock_iter)
        mock_chat_completion.return_value = mock_response

        # Generate text response with streaming callback
        response = generator.run(chat_messages, streaming_callback=streaming_callback_fn)

        # check kwargs passed to text_generation
        _, kwargs = mock_chat_completion.call_args
        assert kwargs == {
            "tools": None,
            "stop": [],
            "stream": True,
            "max_tokens": 512,
            "stream_options": ChatCompletionInputStreamOptions(include_usage=True),
        }

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_tools(self, mock_check_valid_model, tools):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-3.1-70B-Instruct"},
            tools=tools,
        )

        with patch("huggingface_hub.InferenceClient.chat_completion", autospec=True) as mock_chat_completion:
            completion = ChatCompletionOutput(
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ChatCompletionOutputToolCall(
                                    function=ChatCompletionOutputFunctionDefinition(
                                        arguments={"city": "Paris"}, name="weather", description=None
                                    ),
                                    id="0",
                                    type="function",
                                )
                            ],
                        ),
                        logprobs=None,
                    )
                ],
                created=1729074760,
                id="",
                model="meta-llama/Llama-3.1-70B-Instruct",
                system_fingerprint="2.3.2-dev0-sha-28bb7ae",
                usage=ChatCompletionOutputUsage(completion_tokens=30, prompt_tokens=426, total_tokens=456),
            )
            mock_chat_completion.return_value = completion

            messages = [ChatMessage.from_user("What is the weather in Paris?")]
            response = generator.run(messages=messages)

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].tool_calls[0].tool_name == "weather"
        assert response["replies"][0].tool_calls[0].arguments == {"city": "Paris"}
        assert response["replies"][0].tool_calls[0].id == "0"
        assert response["replies"][0].meta == {
            "finish_reason": "tool_calls",
            "index": 0,
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "usage": {"completion_tokens": 30, "prompt_tokens": 426},
        }

    @pytest.mark.parametrize("tools_at_init", [True, False], ids=["init_tools", "runtime_tools"])
    def test_run_with_streaming_tool_calls(
        self,
        mock_check_valid_model,
        mock_chat_completion,
        tools,
        recorded_tool_stream,
        expected_recorded_tool_chunks,
        tools_at_init,
    ):
        callback = Mock()
        generator = HuggingFaceAPIChatGenerator(
            api_type="serverless_inference_api",
            api_params={"model": "Qwen/Qwen3-4B-Instruct-2507"},
            token=None,
            tools=tools if tools_at_init else None,
            streaming_callback=callback,
        )
        mock_chat_completion.return_value = iter(recorded_tool_stream)
        result = generator.run("Weather in Paris and Berlin?", tools=None if tools_at_init else tools)
        assert mock_chat_completion.call_args.kwargs["stream"] is True
        assert [tool.function.name for tool in mock_chat_completion.call_args.kwargs["tools"]] == ["weather"]
        received = [call.args[0] for call in callback.call_args_list]
        assert received == expected_recorded_tool_chunks
        assert all(chunk.component_info == ComponentInfo.from_component(generator) for chunk in received)
        assert result == {
            "replies": [
                ChatMessage.from_assistant(
                    tool_calls=[
                        ToolCall(id="chatcmpl-tool-88f8a30eee6f69d9", tool_name="weather", arguments={"city": "Paris"}),
                        ToolCall(
                            id="chatcmpl-tool-82e851da6ffc4a76", tool_name="weather", arguments={"city": "Berlin"}
                        ),
                    ],
                    meta={
                        "model": "Qwen/Qwen3-4B-Instruct-2507",
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "completion_start_time": received[0].meta["received_at"],
                        "usage": {"prompt_tokens": 162, "completion_tokens": 38},
                    },
                )
            ]
        }

    def test_run_with_streaming_reasoning_and_tool_call(self, mock_check_valid_model, mock_chat_completion, tools):
        events = [
            {"choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": "We"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": " need"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": " to"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": " call"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": " the"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": " weather"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": " function"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": " with"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": " city"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": ' "'}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": "Paris"}, "index": 0}]},
            {"choices": [{"delta": {"reasoning": '".'}, "index": 0}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "function": {"arguments": '{"city":"Paris"}', "name": "weather"},
                                    "id": "fc_633c7513-a52c-4b84-b37a-536ac174837d",
                                    "index": 0,
                                    "type": "function",
                                }
                            ]
                        },
                        "index": 0,
                    }
                ]
            },
            {
                "choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}],
                "usage": {"completion_tokens": 35, "prompt_tokens": 130, "total_tokens": 165},
            },
            {"choices": [], "usage": {"completion_tokens": 35, "prompt_tokens": 130, "total_tokens": 165}},
        ]
        envelope = {
            "created": 1792419651,
            "id": "chatcmpl-b25816e0-8dec-473d-b683-7fea4eee937f",
            "model": "openai/gpt-oss-20b",
            "system_fingerprint": "fp_80501ff3a1",
        }
        streaming_tool_chunks = [
            ChatCompletionStreamOutput.parse_obj_as_instance({**envelope, **event}) for event in events
        ]
        meta = {"model": "openai/gpt-oss-20b", "received_at": ANY, "finish_reason": None}
        usage = {"prompt_tokens": 130, "completion_tokens": 35}
        expected_streaming_tool_chunks = [
            StreamingChunk(content="", meta=meta),
            StreamingChunk(content="", reasoning=ReasoningContent("We"), index=0, start=True, meta=meta),
            *[
                StreamingChunk(content="", reasoning=ReasoningContent(fragment), index=0, meta=meta)
                for fragment in [
                    " need",
                    " to",
                    " call",
                    " the",
                    " weather",
                    " function",
                    " with",
                    " city",
                    ' "',
                    "Paris",
                    '".',
                ]
            ],
            StreamingChunk(
                content="",
                index=1,
                start=True,
                meta=meta,
                tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id="fc_633c7513-a52c-4b84-b37a-536ac174837d",
                        tool_name="weather",
                        arguments='{"city":"Paris"}',
                    )
                ],
            ),
            StreamingChunk(
                content="", finish_reason="tool_calls", meta={**meta, "finish_reason": "tool_calls", "usage": usage}
            ),
            StreamingChunk(content="", meta={"model": "openai/gpt-oss-20b", "received_at": ANY, "usage": usage}),
        ]
        received = []
        generator = HuggingFaceAPIChatGenerator(
            api_type="serverless_inference_api",
            api_params={"model": "openai/gpt-oss-20b"},
            token=None,
            tools=tools,
            streaming_callback=received.append,
        )
        mock_chat_completion.return_value = iter(streaming_tool_chunks)
        result = generator.run("What is the weather in Paris? Use the weather tool.")
        component_info = ComponentInfo.from_component(generator)
        assert received == [replace(chunk, component_info=component_info) for chunk in expected_streaming_tool_chunks]
        assert result == {
            "replies": [
                ChatMessage.from_assistant(
                    reasoning=ReasoningContent('We need to call the weather function with city "Paris".'),
                    tool_calls=[
                        ToolCall(
                            id="fc_633c7513-a52c-4b84-b37a-536ac174837d",
                            tool_name="weather",
                            arguments={"city": "Paris"},
                        )
                    ],
                    meta={
                        "model": "openai/gpt-oss-20b",
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "completion_start_time": received[0].meta["received_at"],
                        "usage": {"prompt_tokens": 130, "completion_tokens": 35},
                    },
                )
            ]
        }


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN", None),
    reason="Export an env var called HF_TOKEN containing the Hugging Face token to run this test.",
)
class TestSyncIntegration:
    def test_live_run_serverless(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "nscale"},
            generation_kwargs={"max_tokens": 20},
        )

        # No need for instruction tokens here since we use the chat_completion endpoint which handles the chat
        # templating for us.
        messages = [
            ChatMessage.from_user("What is the capital of France? Be concise only provide the capital, nothing else.")
        ]
        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].text is not None
        meta = response["replies"][0].meta
        assert "usage" in meta
        assert "prompt_tokens" in meta["usage"]
        assert meta["usage"]["prompt_tokens"] > 0
        assert "completion_tokens" in meta["usage"]
        assert meta["usage"]["completion_tokens"] > 0
        assert meta["model"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert meta["finish_reason"] is not None

    def test_live_run_serverless_streaming(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "nscale"},
            generation_kwargs={"max_tokens": 20},
            streaming_callback=streaming_callback_handler,
        )

        # No need for instruction tokens here since we use the chat_completion endpoint which handles the chat
        # templating for us.
        messages = [
            ChatMessage.from_user("What is the capital of France? Be concise only provide the capital, nothing else.")
        ]
        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].text is not None

        response_meta = response["replies"][0].meta
        assert "completion_start_time" in response_meta
        assert datetime.fromisoformat(response_meta["completion_start_time"]) <= datetime.now(UTC)
        assert "usage" in response_meta
        assert "prompt_tokens" in response_meta["usage"]
        assert response_meta["usage"]["prompt_tokens"] >= 0
        assert "completion_tokens" in response_meta["usage"]
        assert response_meta["usage"]["completion_tokens"] >= 0
        assert response_meta["model"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert response_meta["finish_reason"] is not None

    @pytest.mark.parametrize("streaming", [False, True])
    def test_live_run_with_tools(self, tools, streaming):
        """
        We test the round trip: generate tool call, pass tool message, generate response.
        """

        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "nscale"},
            generation_kwargs={"temperature": 0.5},
            streaming_callback=print_streaming_chunk if streaming else None,
        )

        results = generator.run(chat_messages, tools=tools)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert "city" in tool_call.arguments
        assert "Paris" in tool_call.arguments["city"]
        assert message.meta["finish_reason"] == "tool_calls"

        new_messages = [*chat_messages, message, ChatMessage.from_tool(tool_result="22° C", origin=tool_call)]

        # the model tends to make tool calls if provided with tools, so we don't pass them here
        results = generator.run(new_messages, generation_kwargs={"max_tokens": 50})

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower() and "22" in final_message.text

    def test_live_run_multimodal(self, test_files_path):
        image_path = test_files_path / "apple.jpg"
        # Resize the image to keep this test fast
        image_content = ImageContent.from_file_path(file_path=image_path, size=(100, 100))
        messages = [ChatMessage.from_user(content_parts=["What does this image show? Max 5 words", image_content])]

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "google/gemma-3-4b-it", "provider": "deepinfra"},
            generation_kwargs={"max_tokens": 20},
        )

        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        message = response["replies"][0]
        assert message.text
        assert len(message.text) > 0
        assert any(word in message.text.lower() for word in ["apple", "fruit", "red"])


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_async(self, mock_check_valid_model, mock_chat_completion_async, chat_messages):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            streaming_callback=None,
        )

        response = await generator.run_async(messages=chat_messages)

        # check kwargs passed to chat_completion
        _, kwargs = mock_chat_completion_async.call_args
        hf_messages = [
            {"role": "system", "content": "You are a helpful assistant speaking A2 level of English"},
            {"role": "user", "content": "Tell me about Berlin"},
        ]
        assert kwargs == {
            "temperature": 0.6,
            "stop": ["stop", "words"],
            "max_tokens": 512,
            "tools": None,
            "messages": hf_messages,
        }

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_async_with_generation_kwargs(
        self, mock_check_valid_model, mock_chat_completion_async, chat_messages
    ):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            generation_kwargs={"max_tokens": 100, "temperature": 0.5},
        )

        await generator.run_async(messages=chat_messages, generation_kwargs={"temperature": 0.9})

        _, kwargs = mock_chat_completion_async.call_args
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_run_async_with_string_input(self, mock_check_valid_model, mock_chat_completion_async):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
        )
        response = await generator.run_async("What's the capital of France?")

        _, kwargs = mock_chat_completion_async.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "What's the capital of France?"}]

        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)

    @pytest.mark.asyncio
    async def test_run_async_with_streaming(self, mock_check_valid_model, mock_chat_completion_async, chat_messages):
        streaming_call_count = 0

        async def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        # Create a fake streamed response
        async def mock_aiter(self):
            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content="The", role="assistant"),
                        index=0,
                        finish_reason=None,
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content=None, role=None), index=0, finish_reason="length"
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

        mock_response = Mock(__aiter__=mock_aiter)
        mock_chat_completion_async.return_value = mock_response

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            streaming_callback=streaming_callback_fn,
        )

        response = await generator.run_async(messages=chat_messages)

        # check kwargs passed to chat_completion
        _, kwargs = mock_chat_completion_async.call_args
        assert kwargs == {
            "tools": None,
            "stop": [],
            "stream": True,
            "max_tokens": 512,
            "stream_options": ChatCompletionInputStreamOptions(include_usage=True),
        }

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_async_with_tools(self, tools, mock_check_valid_model):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-3.1-70B-Instruct"},
            tools=tools,
        )

        with patch("huggingface_hub.AsyncInferenceClient.chat_completion", autospec=True) as mock_chat_completion_async:
            completion = ChatCompletionOutput(
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ChatCompletionOutputToolCall(
                                    function=ChatCompletionOutputFunctionDefinition(
                                        arguments={"city": "Paris"}, name="weather", description=None
                                    ),
                                    id="0",
                                    type="function",
                                )
                            ],
                        ),
                        logprobs=None,
                    )
                ],
                created=1729074760,
                id="",
                model="meta-llama/Llama-3.1-70B-Instruct",
                system_fingerprint="2.3.2-dev0-sha-28bb7ae",
                usage=ChatCompletionOutputUsage(completion_tokens=30, prompt_tokens=426, total_tokens=456),
            )
            mock_chat_completion_async.return_value = completion

            messages = [ChatMessage.from_user("What is the weather in Paris?")]
            response = await generator.run_async(messages=messages)

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].tool_calls[0].tool_name == "weather"
        assert response["replies"][0].tool_calls[0].arguments == {"city": "Paris"}
        assert response["replies"][0].tool_calls[0].id == "0"
        assert response["replies"][0].meta == {
            "finish_reason": "tool_calls",
            "index": 0,
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "usage": {"completion_tokens": 30, "prompt_tokens": 426},
        }

    @pytest.mark.asyncio
    async def test_run_async_with_sync_streaming_callback(
        self, mock_check_valid_model, mock_chat_completion_async, hf_stream_chunk
    ):
        callback = Mock()
        generator = HuggingFaceAPIChatGenerator(
            api_type="serverless_inference_api", api_params={"model": "some_model"}, token=None
        )
        chunks = [hf_stream_chunk({"content": "Paris"}), hf_stream_chunk({}, finish_reason="stop")]
        mock_chat_completion_async.return_value = _async_stream(chunks)
        await generator.run_async([ChatMessage.from_user("Capital of France?")], streaming_callback=callback)
        assert callback.call_count == len(chunks)
        assert [call.args[0].content for call in callback.call_args_list] == ["Paris", ""]

    @pytest.mark.parametrize("tools_at_init", [True, False], ids=["init_tools", "runtime_tools"])
    @pytest.mark.asyncio
    async def test_run_async_with_streaming_tool_calls(
        self,
        mock_check_valid_model,
        mock_chat_completion_async,
        tools,
        recorded_tool_stream,
        expected_recorded_tool_chunks,
        tools_at_init,
    ):
        callback = AsyncMock()
        generator = HuggingFaceAPIChatGenerator(
            api_type="serverless_inference_api",
            api_params={"model": "Qwen/Qwen3-4B-Instruct-2507"},
            token=None,
            tools=tools if tools_at_init else None,
            streaming_callback=callback,
        )
        mock_chat_completion_async.return_value = _async_stream(recorded_tool_stream)
        result = await generator.run_async("Weather in Paris and Berlin?", tools=None if tools_at_init else tools)
        assert mock_chat_completion_async.call_args.kwargs["stream"] is True
        assert [tool.function.name for tool in mock_chat_completion_async.call_args.kwargs["tools"]] == ["weather"]
        received = [call.args[0] for call in callback.await_args_list]
        assert received == expected_recorded_tool_chunks
        assert all(chunk.component_info == ComponentInfo.from_component(generator) for chunk in received)
        assert result == {
            "replies": [
                ChatMessage.from_assistant(
                    tool_calls=[
                        ToolCall(id="chatcmpl-tool-88f8a30eee6f69d9", tool_name="weather", arguments={"city": "Paris"}),
                        ToolCall(
                            id="chatcmpl-tool-82e851da6ffc4a76", tool_name="weather", arguments={"city": "Berlin"}
                        ),
                    ],
                    meta={
                        "model": "Qwen/Qwen3-4B-Instruct-2507",
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "completion_start_time": received[0].meta["received_at"],
                        "usage": {"prompt_tokens": 162, "completion_tokens": 38},
                    },
                )
            ]
        }


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN", None),
    reason="Export an env var called HF_TOKEN containing the Hugging Face token to run this test.",
)
class TestAsyncIntegration:
    @pytest.mark.asyncio
    async def test_live_run_async_serverless(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "nscale"},
            generation_kwargs={"max_tokens": 20},
        )

        messages = [
            ChatMessage.from_user("What is the capital of France? Be concise only provide the capital, nothing else.")
        ]
        try:
            response = await generator.run_async(messages=messages)

            assert "replies" in response
            assert isinstance(response["replies"], list)
            assert len(response["replies"]) > 0
            assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
            assert response["replies"][0].text is not None

            meta = response["replies"][0].meta
            assert "usage" in meta
            assert "prompt_tokens" in meta["usage"]
            assert meta["usage"]["prompt_tokens"] > 0
            assert "completion_tokens" in meta["usage"]
            assert meta["usage"]["completion_tokens"] > 0
            assert meta["model"] == "Qwen/Qwen3-4B-Instruct-2507"
            assert meta["finish_reason"] is not None
        finally:
            await generator.close_async()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN", None),
    reason="Export an env var called HF_TOKEN containing the Hugging Face token to run this test.",
)
class TestReasoningIntegration:
    def test_live_run_multi_turn_with_reasoning_model(self):
        """
        Test multi-turn conversation with a reasoning model.

        This test verifies that:
        1. Reasoning content is captured from the model's response
        2. When the assistant message (with reasoning) is sent back in a multi-turn conversation,
           the API call succeeds (reasoning is dropped during conversion since HF API doesn't support it)
        """
        # Note: Using a model that supports reasoning AND a provider that actually follows the spec defined in
        # huggingface-hub. Reasoning content especially seems to be non-standard across providers and is either left
        # in the main response or put in a new field that is not part of the official API.
        # One combo that does respect the spec is groq + openai/gpt-oss-20b.
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "openai/gpt-oss-20b", "provider": "groq"},
            generation_kwargs={"max_tokens": 300},
        )

        # First turn: ask a question
        messages = [ChatMessage.from_user("What is 2 + 2? Answer briefly.")]
        response = generator.run(messages=messages)

        assert "replies" in response
        assert len(response["replies"]) > 0
        first_reply = response["replies"][0]
        assert first_reply.text is not None
        assert first_reply.reasoning is not None

        # Second turn: send a follow-up including the assistant's previous response
        # This tests that convert_message_to_hf_format properly handles messages
        # that may contain ReasoningContent (it should skip it)
        follow_up_messages = [
            ChatMessage.from_user("What is 2 + 2? Answer briefly."),
            first_reply,  # Include the assistant's response with reasoning
            ChatMessage.from_user("Now what is 3 + 3? Answer briefly."),
        ]
        follow_up_response = generator.run(messages=follow_up_messages)

        # Verify the second turn succeeds
        assert "replies" in follow_up_response
        assert len(follow_up_response["replies"]) > 0
        assert follow_up_response["replies"][0].text is not None
        assert follow_up_response["replies"][0].reasoning is not None


class TestTools:
    def test_hugging_face_api_generator_with_toolset_initialization(self, tools):
        """Test that the HuggingFaceAPIChatGenerator can be initialized with a Toolset."""
        toolset = Toolset(tools)
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "irrelevant"}, tools=toolset
        )
        assert generator.tools == toolset

    def test_to_dict_with_toolset(self, tools):
        """Test that the HuggingFaceAPIChatGenerator can be serialized to a dictionary with a Toolset."""
        toolset = Toolset(tools[:1])
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "irrelevant"}, tools=toolset
        )
        data = generator.to_dict()

        # deserializing the serialized component must reproduce the original toolset
        loaded = HuggingFaceAPIChatGenerator.from_dict(data)
        assert isinstance(loaded.tools, Toolset)
        assert list(loaded.tools) == list(toolset)


class TestReasoning:
    def test_run_with_reasoning_non_streaming(self, mock_check_valid_model, chat_messages):
        """Test that reasoning content is correctly extracted from non-streaming responses."""
        with patch("huggingface_hub.InferenceClient.chat_completion", autospec=True) as mock_chat_completion:
            reasoning_text = "Let me think about this. France is a country in Europe. Its capital city is Paris."
            completion = ChatCompletionOutput(
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="eos_token",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            content="The capital of France is Paris.", role="assistant", reasoning=reasoning_text
                        ),
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                usage=ChatCompletionOutputUsage(completion_tokens=20, prompt_tokens=17, total_tokens=37),
                created=1710498360,
            )
            mock_chat_completion.return_value = completion

            generator = HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            )

            response = generator.run(chat_messages)

            assert "replies" in response
            assert len(response["replies"]) == 1
            reply = response["replies"][0]
            assert reply.text == "The capital of France is Paris."
            assert reply.reasoning is not None
            assert isinstance(reply.reasoning, ReasoningContent)
            assert reply.reasoning.reasoning_text == reasoning_text

    def test_run_without_reasoning_non_streaming(self, mock_check_valid_model, mock_chat_completion, chat_messages):
        """Test that responses without reasoning work correctly (backward compatibility)."""
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
        )

        response = generator.run(chat_messages)

        assert "replies" in response
        assert len(response["replies"]) == 1
        reply = response["replies"][0]
        assert reply.text == "The capital of France is Paris."
        assert reply.reasoning is None

    def test_run_with_reasoning_streaming(self, mock_check_valid_model, chat_messages):
        """Test that reasoning content is correctly extracted from streaming responses."""
        streaming_chunks_received = []

        def streaming_callback_fn(chunk: StreamingChunk):
            streaming_chunks_received.append(chunk)

        with patch("huggingface_hub.InferenceClient.chat_completion", autospec=True) as mock_chat_completion:
            # Create a fake streamed response with reasoning
            def mock_iter(self):
                # First chunk with reasoning
                yield ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(
                                role="assistant", content=None, reasoning="Let me think..."
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                    id="some_id",
                    model="some_model",
                    system_fingerprint="some_fingerprint",
                    created=1710498504,
                )
                # Second chunk with more reasoning
                yield ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(
                                role=None, content=None, reasoning=" The capital of France is Paris."
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                    id="some_id",
                    model="some_model",
                    system_fingerprint="some_fingerprint",
                    created=1710498504,
                )
                # Third chunk with actual content
                yield ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(role=None, content="Paris", reasoning=None),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                    id="some_id",
                    model="some_model",
                    system_fingerprint="some_fingerprint",
                    created=1710498504,
                )
                # Final chunk with finish reason
                yield ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(role=None, content=None, reasoning=None),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                    id="some_id",
                    model="some_model",
                    system_fingerprint="some_fingerprint",
                    created=1710498504,
                )

            mock_response = Mock(__iter__=mock_iter)
            mock_chat_completion.return_value = mock_response

            generator = HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
                streaming_callback=streaming_callback_fn,
            )

            response = generator.run(chat_messages)

            # Check streaming chunks received with reasoning
            assert len(streaming_chunks_received) == 4
            assert streaming_chunks_received[0].reasoning is not None
            assert streaming_chunks_received[0].reasoning.reasoning_text == "Let me think..."
            assert streaming_chunks_received[1].reasoning is not None
            assert streaming_chunks_received[1].reasoning.reasoning_text == " The capital of France is Paris."

            # Check final message
            assert "replies" in response
            assert len(response["replies"]) == 1
            reply = response["replies"][0]
            assert reply.text == "Paris"
            assert reply.reasoning is not None
            assert isinstance(reply.reasoning, ReasoningContent)
            assert reply.reasoning.reasoning_text == "Let me think... The capital of France is Paris."

    @pytest.mark.asyncio
    async def test_run_async_with_reasoning_non_streaming(self, mock_check_valid_model, chat_messages):
        """Test that reasoning content is correctly extracted from async non-streaming responses."""
        with patch(
            "huggingface_hub.AsyncInferenceClient.chat_completion", new_callable=AsyncMock
        ) as mock_chat_completion:
            completion = ChatCompletionOutput(
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="eos_token",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            content="The capital of France is Paris.",
                            role="assistant",
                            reasoning="Let me reason about this question step by step.",
                        ),
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                usage=ChatCompletionOutputUsage(completion_tokens=20, prompt_tokens=17, total_tokens=37),
                created=1710498360,
            )
            mock_chat_completion.return_value = completion

            generator = HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            )

            response = await generator.run_async(chat_messages)

            assert "replies" in response
            assert len(response["replies"]) == 1
            reply = response["replies"][0]
            assert reply.text == "The capital of France is Paris."
            assert reply.reasoning is not None
            assert isinstance(reply.reasoning, ReasoningContent)
            assert reply.reasoning.reasoning_text == "Let me reason about this question step by step."

    @pytest.mark.asyncio
    async def test_run_async_with_reasoning_streaming(self, mock_check_valid_model, chat_messages):
        """Test that reasoning content is correctly extracted from async streaming responses."""
        streaming_chunks_received = []

        async def streaming_callback_fn(chunk: StreamingChunk):
            streaming_chunks_received.append(chunk)

        with patch(
            "huggingface_hub.AsyncInferenceClient.chat_completion", new_callable=AsyncMock
        ) as mock_chat_completion:
            # Create async iterable for streaming
            async def mock_aiter():
                # First chunk with reasoning
                yield ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(
                                role="assistant", content=None, reasoning="Thinking..."
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                    id="some_id",
                    model="some_model",
                    system_fingerprint="some_fingerprint",
                    created=1710498504,
                )
                # Second chunk with content
                yield ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(role=None, content="Paris", reasoning=None),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                    id="some_id",
                    model="some_model",
                    system_fingerprint="some_fingerprint",
                    created=1710498504,
                )
                # Final chunk
                yield ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(role=None, content=None, reasoning=None),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                    id="some_id",
                    model="some_model",
                    system_fingerprint="some_fingerprint",
                    created=1710498504,
                )

            mock_chat_completion.return_value = mock_aiter()

            generator = HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
                streaming_callback=streaming_callback_fn,
            )

            response = await generator.run_async(chat_messages)

            # Check streaming chunks
            assert len(streaming_chunks_received) == 3
            assert streaming_chunks_received[0].reasoning is not None
            assert streaming_chunks_received[0].reasoning.reasoning_text == "Thinking..."

            # Check final message
            assert "replies" in response
            assert len(response["replies"]) == 1
            reply = response["replies"][0]
            assert reply.text == "Paris"
            assert reply.reasoning is not None
            assert isinstance(reply.reasoning, ReasoningContent)
            assert reply.reasoning.reasoning_text == "Thinking..."
