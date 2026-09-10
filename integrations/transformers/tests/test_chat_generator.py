# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.dataclasses.streaming_chunk import StreamingChunk
from haystack.tools import Tool
from haystack.tools.toolset import Toolset
from haystack.utils import ComponentDevice
from haystack.utils.auth import Secret
from transformers import PreTrainedTokenizer

from haystack_integrations.components.generators.transformers import TransformersChatGenerator
from haystack_integrations.components.generators.transformers.chat.chat_generator import default_tool_parser


# used to test serialization of streaming_callback
def streaming_callback_handler(x):
    return x


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"Weather data for {city}"


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant speaking A2 level of English"),
        ChatMessage.from_user("Tell me about Berlin"),
    ]


@pytest.fixture
def model_info_mock():
    with patch(
        "haystack_integrations.components.generators.transformers.chat.chat_generator.model_info",
        new=Mock(return_value=Mock(pipeline_tag="text-generation")),
    ) as mock:
        yield mock


@pytest.fixture
def mock_pipeline_with_tokenizer():
    # Mocking the pipeline
    mock_pipeline = Mock(return_value=[{"generated_text": "Berlin is cool"}])

    # Mocking the tokenizer
    mock_tokenizer = Mock(spec=PreTrainedTokenizer)
    mock_tokenizer.encode.return_value = ["Berlin", "is", "cool"]
    mock_tokenizer.apply_chat_template.return_value = "Berlin is cool"
    mock_tokenizer.pad_token_id = 100
    mock_pipeline.tokenizer = mock_tokenizer

    return mock_pipeline


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=get_weather,
    )

    return [tool]


def custom_tool_parser(text: str) -> list[ToolCall] | None:
    """Test implementation of a custom tool parser."""
    return [ToolCall(tool_name="weather", arguments={"city": "Berlin"})]


class TestInitializationAndSerialization:
    def test_initialize_with_valid_model_and_generation_parameters(self):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = TransformersChatGenerator(
            model=model,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.generation_kwargs == {**generation_kwargs, "stop_sequences": ["stop"]}
        assert generator.streaming_callback == streaming_callback

    def test_init_custom_token(self):
        generator = TransformersChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            token=Secret.from_token("test-token"),
            device=ComponentDevice.from_str("cpu"),
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text-generation",
            "device": "cpu",
        }

    def test_init_custom_device(self):
        generator = TransformersChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            device=ComponentDevice.from_str("cpu"),
            token=None,
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text-generation",
            "device": "cpu",
        }

    def test_init_task_parameter(self):
        generator = TransformersChatGenerator(
            task="text-generation", device=ComponentDevice.from_str("cpu"), token=None
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "Qwen/Qwen3-0.6B",
            "task": "text-generation",
            "device": "cpu",
        }

    def test_init_task_in_huggingface_pipeline_kwargs(self):
        generator = TransformersChatGenerator(
            huggingface_pipeline_kwargs={"task": "text-generation"}, device=ComponentDevice.from_str("cpu"), token=None
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "Qwen/Qwen3-0.6B",
            "task": "text-generation",
            "device": "cpu",
        }

    def test_transformers_chat_generator_with_toolset_initialization(self, mock_pipeline_with_tokenizer, tools):
        """Test that the TransformersChatGenerator can be initialized with a Toolset."""
        toolset = Toolset(tools)
        generator = TransformersChatGenerator(model="irrelevant", tools=toolset)
        generator.pipeline = mock_pipeline_with_tokenizer
        assert generator.tools == toolset

    def test_init_image_text_to_text(self):
        llm = TransformersChatGenerator(model="Qwen/Qwen2-VL-2B-Instruct")

        assert llm
        assert isinstance(llm, TransformersChatGenerator)
        assert "model" in llm.huggingface_pipeline_kwargs

    def test_init_image_text_to_text_task(self):
        generator = TransformersChatGenerator(
            model="Qwen/Qwen2-VL-2B-Instruct",
            task="image-text-to-text",
            device=ComponentDevice.from_str("cpu"),
            token=None,
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "task": "image-text-to-text",
            "device": "cpu",
        }

    def test_to_dict(self, tools):
        generator = TransformersChatGenerator(
            model="NousResearch/Llama-2-7b-chat-hf",
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=None,
            chat_template="irrelevant",
            tools=tools,
            enable_thinking=True,
        )

        # Call the to_dict method
        result = generator.to_dict()
        init_params = result["init_parameters"]

        # Assert that the init_params dictionary contains the expected keys and values
        assert init_params["token"] == {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"}
        assert init_params["huggingface_pipeline_kwargs"]["model"] == "NousResearch/Llama-2-7b-chat-hf"
        assert "token" not in init_params["huggingface_pipeline_kwargs"]
        assert init_params["generation_kwargs"] == {
            "max_new_tokens": 512,
            "n": 5,
            "stop_sequences": ["stop", "words"],
        }
        assert init_params["streaming_callback"] is None
        assert init_params["chat_template"] == "irrelevant"
        assert init_params["enable_thinking"] is True

        # deserializing the serialized component must reproduce the original tools
        loaded = TransformersChatGenerator.from_dict(result)
        assert loaded.tools == tools

    def test_from_dict(self, tools):
        generator = TransformersChatGenerator(
            model="NousResearch/Llama-2-7b-chat-hf",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=None,
            chat_template="irrelevant",
            tools=tools,
            enable_thinking=True,
        )
        # Call the to_dict method
        result = generator.to_dict()

        generator_2 = TransformersChatGenerator.from_dict(result)

        assert generator_2.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert generator_2.generation_kwargs == {
            "max_new_tokens": 512,
            "n": 5,
            "stop_sequences": ["stop", "words"],
        }
        assert generator_2.streaming_callback is None
        assert generator_2.chat_template == "irrelevant"
        assert generator_2.enable_thinking is True
        assert len(generator_2.tools) == 1
        assert generator_2.tools[0].name == "weather"
        assert generator_2.tools[0].description == "useful to determine the weather in a given location"
        assert generator_2.tools[0].parameters == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    def test_to_dict_with_toolset(self, mock_pipeline_with_tokenizer, tools):
        """Test that the TransformersChatGenerator can be serialized to a dictionary with a Toolset."""
        toolset = Toolset(tools)
        generator = TransformersChatGenerator(huggingface_pipeline_kwargs={"model": "irrelevant"}, tools=toolset)
        generator.pipeline = mock_pipeline_with_tokenizer
        data = generator.to_dict()

        # deserializing the serialized component must reproduce the original toolset
        loaded = TransformersChatGenerator.from_dict(data)
        assert isinstance(loaded.tools, Toolset)
        assert list(loaded.tools) == list(toolset)

    def test_from_dict_with_toolset(self, tools):
        """Test that the TransformersChatGenerator can be deserialized from a dictionary with a Toolset."""
        toolset = Toolset(tools)
        component = TransformersChatGenerator(model="irrelevant", tools=toolset)
        data = component.to_dict()

        deserialized_component = TransformersChatGenerator.from_dict(data)

        assert isinstance(deserialized_component.tools, Toolset)
        assert len(deserialized_component.tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component.tools)


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_HF_TOKEN", raising=False)
        generator = TransformersChatGenerator(task="text-generation", token=Secret.from_env_var("MISSING_HF_TOKEN"))

        with pytest.raises(ValueError, match="MISSING_HF_TOKEN"):
            generator.warm_up()

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.ThreadPoolExecutor")
    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.pipeline")
    def test_sync_lifecycle(self, pipeline_mock, executor_cls_mock):
        generator = TransformersChatGenerator(task="text-generation", token=None)
        executor = executor_cls_mock.return_value

        generator.warm_up()
        assert generator.pipeline is pipeline_mock.return_value
        assert generator.executor is executor

        generator.close()
        executor.shutdown.assert_called_once_with(wait=True)
        assert generator.executor is None
        assert generator.pipeline is pipeline_mock.return_value

        generator.warm_up()
        assert executor_cls_mock.call_count == 2
        pipeline_mock.assert_called_once()

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.ThreadPoolExecutor")
    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.pipeline")
    def test_warm_up_is_idempotent(self, pipeline_mock, executor_cls_mock):
        generator = TransformersChatGenerator(task="text-generation", token=None)

        generator.warm_up()
        generator.warm_up()

        pipeline_mock.assert_called_once()
        executor_cls_mock.assert_called_once()

    def test_close_is_safe_without_warm_up(self):
        generator = TransformersChatGenerator(task="text-generation", token=None)

        generator.close()

        assert generator.executor is None

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.pipeline")
    def test_close_does_not_shutdown_user_supplied_executor(self, pipeline_mock):
        executor = Mock()
        generator = TransformersChatGenerator(task="text-generation", token=None, async_executor=executor)
        generator.warm_up()

        generator.close()

        executor.shutdown.assert_not_called()
        assert generator.executor is executor

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.pipeline")
    def test_task_inferred_at_warm_up_not_init(self, pipeline_mock, model_info_mock):
        generator = TransformersChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2", device=ComponentDevice.from_str("cpu"), token=None
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "device": "cpu",
        }
        generation_kwargs = generator.generation_kwargs.copy()
        model_info_mock.assert_not_called()

        generator.warm_up()

        assert generator.generation_kwargs == generation_kwargs
        model_info_mock.assert_called_once_with("mistralai/Mistral-7B-Instruct-v0.2", token=None)
        pipeline_mock.assert_called_once_with(
            model="mistralai/Mistral-7B-Instruct-v0.2", device="cpu", token=None, task="text-generation"
        )
        generator.close()

    def test_warm_up_invalid_task(self):
        generator = TransformersChatGenerator(task="text-classification")

        with pytest.raises(ValueError, match=r"is not supported\."):
            generator.warm_up()

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.pipeline")
    def test_warm_up(self, pipeline_mock):
        generator = TransformersChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation", device=ComponentDevice.from_str("cpu")
        )

        pipeline_mock.assert_not_called()

        generator.warm_up()

        pipeline_mock.assert_called_once_with(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            token=generator.token.resolve_value(),
            device="cpu",
        )

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.pipeline")
    def test_warm_up_with_tools(self, pipeline_mock):
        """Test that warm_up() calls warm_up on tools and is idempotent."""

        # Create a mock tool that tracks if warm_up() was called
        class MockTool(Tool):
            warm_up_call_count = 0  # Class variable to track calls

            def __init__(self):
                super().__init__(
                    name="mock_tool",
                    description="A mock tool for testing",
                    parameters={"x": {"type": "string"}},
                    function=lambda x: x,
                )

            def warm_up(self):
                MockTool.warm_up_call_count += 1

        # Reset the class variable before test
        MockTool.warm_up_call_count = 0
        mock_tool = MockTool()

        # Create TransformersChatGenerator with the mock tool
        generator = TransformersChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            device=ComponentDevice.from_str("cpu"),
            tools=[mock_tool],
        )

        # Verify initial state - warm_up not called yet
        assert MockTool.warm_up_call_count == 0
        assert generator.pipeline is None

        # Call warm_up() on the generator
        generator.warm_up()

        # Assert that the tool's warm_up() was called
        assert MockTool.warm_up_call_count == 1
        assert generator.pipeline is not None

        # Verify pipeline was initialized
        pipeline_mock.assert_called_once()

        # Call warm_up() again and verify it's idempotent (only warms up once)
        generator.warm_up()

        # The tool's warm_up should still only have been called once
        assert MockTool.warm_up_call_count == 1
        assert generator.pipeline is not None
        # Pipeline should still only have been called once
        pipeline_mock.assert_called_once()

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.pipeline")
    def test_warm_up_with_no_tools(self, pipeline_mock):
        """Test that warm_up() works when no tools are provided."""

        generator = TransformersChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation", device=ComponentDevice.from_str("cpu")
        )

        # Verify initial state
        assert generator.pipeline is None
        assert generator.tools is None

        # Call warm_up() - should not raise an error
        generator.warm_up()

        # Verify the component is warmed up
        assert generator.pipeline is not None
        pipeline_mock.assert_called_once()

        # Call warm_up() again - should be idempotent
        generator.warm_up()
        assert generator.pipeline is not None
        # Pipeline should still only have been called once
        pipeline_mock.assert_called_once()

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.pipeline")
    def test_warm_up_with_multiple_tools(self, pipeline_mock):
        """Test that warm_up() works with multiple tools."""

        # Track warm_up calls
        warm_up_calls = []

        class MockTool(Tool):
            def __init__(self, tool_name):
                super().__init__(
                    name=tool_name,
                    description=f"Mock tool {tool_name}",
                    parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                    function=lambda x: f"{tool_name} result: {x}",
                )

            def warm_up(self):
                warm_up_calls.append(self.name)

        mock_tool1 = MockTool("tool1")
        mock_tool2 = MockTool("tool2")

        # Use a LIST of tools, not a Toolset
        generator = TransformersChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            device=ComponentDevice.from_str("cpu"),
            tools=[mock_tool1, mock_tool2],
        )

        # Call warm_up()
        generator.warm_up()

        # Assert that both tools' warm_up() were called
        assert "tool1" in warm_up_calls
        assert "tool2" in warm_up_calls
        assert generator.pipeline is not None
        pipeline_mock.assert_called_once()

        # Track count
        call_count = len(warm_up_calls)

        # Verify idempotency
        generator.warm_up()
        assert len(warm_up_calls) == call_count
        # Pipeline should still only have been called once
        pipeline_mock.assert_called_once()


class TestRun:
    def test_run(self, mock_pipeline_with_tokenizer, chat_messages):
        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_with_tokenizer

        results = generator.run(messages=chat_messages)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"

    def test_run_with_string_input(self, mock_pipeline_with_tokenizer):
        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")
        generator.pipeline = mock_pipeline_with_tokenizer

        results = generator.run("Who is the best American actor?")

        assert mock_pipeline_with_tokenizer.tokenizer.apply_chat_template.call_args[0][0] == [
            {"role": "user", "content": "Who is the best American actor?"}
        ]
        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        assert results["replies"][0].is_from(ChatRole.ASSISTANT)

    def test_run_with_custom_generation_parameters(self, mock_pipeline_with_tokenizer, chat_messages):
        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_with_tokenizer

        generation_kwargs = {"temperature": 0.8, "max_new_tokens": 100}

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_with_tokenizer
        results = generator.run(messages=chat_messages, generation_kwargs=generation_kwargs)

        # check kwargs passed pipeline
        _, kwargs = generator.pipeline.call_args
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == 0.8

        # replies are properly parsed and returned
        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"

    def test_run_with_generation_kwargs(self, mock_pipeline_with_tokenizer, chat_messages):
        generator = TransformersChatGenerator(
            model="meta-llama/Llama-2-13b-chat-hf",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5},
        )
        generator.pipeline = mock_pipeline_with_tokenizer

        generator.run(messages=chat_messages, generation_kwargs={"temperature": 0.9})

        _, kwargs = generator.pipeline.call_args
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == 0.9

    def test_run_with_streaming_callback(self, mock_pipeline_with_tokenizer, chat_messages):
        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk): ...

        generator = TransformersChatGenerator(
            model="meta-llama/Llama-2-13b-chat-hf", streaming_callback=streaming_callback_fn
        )

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_with_tokenizer

        results = generator.run(messages=chat_messages)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"
        generator.pipeline.assert_called_once()
        assert generator.pipeline.call_args[1]["streamer"].token_handler == streaming_callback_fn

    def test_run_with_streaming_callback_in_run_method(self, mock_pipeline_with_tokenizer, chat_messages):
        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk): ...

        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_with_tokenizer

        results = generator.run(messages=chat_messages, streaming_callback=streaming_callback_fn)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"
        generator.pipeline.assert_called_once()
        assert generator.pipeline.call_args[1]["streamer"].token_handler == streaming_callback_fn

    @patch("haystack_integrations.components.generators.transformers.chat.chat_generator.convert_message_to_hf_format")
    def test_messages_conversion_is_called(self, mock_convert):
        generator = TransformersChatGenerator(model="fake-model")

        messages = [ChatMessage.from_user("Hello"), ChatMessage.from_assistant("Hi there")]

        with patch.object(generator, "pipeline") as mock_pipeline:
            mock_pipeline.tokenizer.apply_chat_template.return_value = "test prompt"
            mock_pipeline.return_value = [{"generated_text": "test response"}]

            generator.run(messages)

        assert mock_convert.call_count == 2
        mock_convert.assert_any_call(messages[0])
        mock_convert.assert_any_call(messages[1])


class TestValidationAndTools:
    @pytest.mark.parametrize(
        "generated_text",
        [
            "Berlin is the capital of Germany.",  # no tool call
            '{"name": "weather", "arguments": {"city": broken}}',  # malformed JSON arguments
        ],
    )
    def test_default_tool_parser_returns_none_for_invalid_input(self, generated_text):
        assert default_tool_parser(generated_text) is None

    def test_init_fail_with_stop_words_and_stopping_criteria(self):
        with pytest.raises(ValueError, match="Found both the `stop_words` init parameter"):
            TransformersChatGenerator(
                model="irrelevant", stop_words=["stop"], generation_kwargs={"stopping_criteria": "fake-criteria"}
            )

    def test_run_with_stop_words_removed_from_replies(self, mock_pipeline_with_tokenizer):
        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf", stop_words=["unambiguously"])
        mock_pipeline_with_tokenizer.return_value = [{"generated_text": "Berlin is cool unambiguously"}]
        # tokenizer without a pad token: _StopWordsCriteria falls back to the eos token
        mock_pipeline_with_tokenizer.tokenizer.pad_token = None
        mock_pipeline_with_tokenizer.tokenizer.eos_token = "</s>"
        generator.pipeline = mock_pipeline_with_tokenizer

        results = generator.run(messages=[ChatMessage.from_user("Tell me about Berlin")])

        assert results["replies"][0].text == "Berlin is cool"
        assert "stopping_criteria" in generator.pipeline.call_args[1]

    def test_init_fail_with_duplicate_tool_names(self, tools):
        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError, match="Duplicate tool names found"):
            TransformersChatGenerator(model="irrelevant", tools=duplicate_tools)

    def test_init_fail_with_tools_and_streaming(self, tools):
        with pytest.raises(ValueError, match="Using tools and streaming at the same time is not supported"):
            TransformersChatGenerator(model="irrelevant", tools=tools, streaming_callback=streaming_callback_handler)

    def test_run_with_tools(self, tools):
        generator = TransformersChatGenerator(model="Qwen/Qwen3-0.6B", tools=tools)

        # Mock pipeline and tokenizer
        mock_pipeline = Mock(return_value=[{"generated_text": '{"name": "weather", "arguments": {"city": "Paris"}}'}])
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.encode.return_value = ["some", "tokens"]
        mock_tokenizer.pad_token_id = 100
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_pipeline.tokenizer = mock_tokenizer
        generator.pipeline = mock_pipeline

        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.tool_calls
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    def test_run_with_tools_in_run_method(self, tools):
        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Mock pipeline and tokenizer
        mock_pipeline = Mock(return_value=[{"generated_text": '{"name": "weather", "arguments": {"city": "Paris"}}'}])
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.encode.return_value = ["some", "tokens"]
        mock_tokenizer.pad_token_id = 100
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_pipeline.tokenizer = mock_tokenizer
        generator.pipeline = mock_pipeline

        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        results = generator.run(messages=messages, tools=tools)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.tool_calls
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    def test_run_with_tools_and_tool_response(self):
        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Mock pipeline and tokenizer
        mock_pipeline = Mock(return_value=[{"generated_text": "The weather in Paris is 22°C"}])
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.encode.return_value = ["some", "tokens"]
        mock_tokenizer.pad_token_id = 100
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_pipeline.tokenizer = mock_tokenizer
        generator.pipeline = mock_pipeline

        tool_call = ToolCall(tool_name="weather", arguments={"city": "Paris"})
        messages = [
            ChatMessage.from_user("What's the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="22°C", origin=tool_call),
        ]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert not message.tool_calls  # No tool calls in the final response
        assert "22°C" in message.text
        assert message.meta["finish_reason"] == "stop"

    def test_run_with_custom_tool_parser(self, mock_pipeline_with_tokenizer, tools):
        """Test that a custom tool parsing function works correctly."""
        generator = TransformersChatGenerator(
            model="meta-llama/Llama-2-13b-chat-hf", tools=tools, tool_parsing_function=custom_tool_parser
        )
        generator.pipeline = mock_pipeline_with_tokenizer

        messages = [ChatMessage.from_user("What's the weather like in Berlin?")]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        assert len(results["replies"][0].tool_calls) == 1
        assert results["replies"][0].tool_calls[0].tool_name == "weather"
        assert results["replies"][0].tool_calls[0].arguments == {"city": "Berlin"}

    def test_default_tool_parser(self, tools):
        """Test that the default tool parser works correctly with valid tool call format."""
        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf", tools=tools)
        generator.pipeline = Mock(
            return_value=[{"generated_text": '{"name": "weather", "arguments": {"city": "Berlin"}}'}]
        )
        generator.pipeline.tokenizer = Mock()
        generator.pipeline.tokenizer.encode.return_value = [1, 2, 3]
        generator.pipeline.tokenizer.pad_token_id = 1
        generator.pipeline.tokenizer.apply_chat_template.return_value = "Irrelevant"

        messages = [ChatMessage.from_user("What's the weather like in Berlin?")]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        assert len(results["replies"][0].tool_calls) == 1
        assert results["replies"][0].tool_calls[0].tool_name == "weather"
        assert results["replies"][0].tool_calls[0].arguments == {"city": "Berlin"}


class TestRunAsync:
    """Async tests for TransformersChatGenerator"""

    @pytest.mark.asyncio
    async def test_run_async(self, mock_pipeline_with_tokenizer, chat_messages):
        """Test basic async functionality"""
        generator = TransformersChatGenerator(model="mocked-model")
        generator.pipeline = mock_pipeline_with_tokenizer

        results = await generator.run_async(messages=chat_messages)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"
        generator.close()

    @pytest.mark.asyncio
    async def test_run_async_with_generation_kwargs(self, mock_pipeline_with_tokenizer, chat_messages):
        generator = TransformersChatGenerator(
            model="meta-llama/Llama-2-13b-chat-hf",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5},
        )
        generator.pipeline = mock_pipeline_with_tokenizer

        await generator.run_async(messages=chat_messages, generation_kwargs={"temperature": 0.9})

        _, kwargs = generator.pipeline.call_args
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == 0.9
        generator.close()

    @pytest.mark.asyncio
    async def test_run_async_with_string_input(self, mock_pipeline_with_tokenizer):
        generator = TransformersChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")
        generator.pipeline = mock_pipeline_with_tokenizer

        results = await generator.run_async("Who is the best American actor?")

        assert mock_pipeline_with_tokenizer.tokenizer.apply_chat_template.call_args[0][0] == [
            {"role": "user", "content": "Who is the best American actor?"}
        ]
        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        assert results["replies"][0].is_from(ChatRole.ASSISTANT)
        generator.close()

    @pytest.mark.asyncio
    async def test_run_async_with_tools(self, mock_pipeline_with_tokenizer, tools):
        """Test async functionality with tools"""
        generator = TransformersChatGenerator(model="mocked-model", tools=tools)
        # Create a new mock with return_value set in constructor to avoid thread-safety issues
        mock_pipeline = Mock(return_value=[{"generated_text": '{"name": "weather", "arguments": {"city": "Berlin"}}'}])
        # Copy the tokenizer from the fixture to the new mock
        mock_pipeline.tokenizer = mock_pipeline_with_tokenizer.tokenizer
        generator.pipeline = mock_pipeline
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        results = await generator.run_async(messages=messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.tool_calls
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Berlin"}
        generator.close()

    @pytest.mark.asyncio
    async def test_concurrent_async_requests(self, mock_pipeline_with_tokenizer, chat_messages):
        """Test handling of multiple concurrent async requests"""
        generator = TransformersChatGenerator(model="mocked-model")
        generator.pipeline = mock_pipeline_with_tokenizer

        # Create multiple concurrent requests
        tasks = [generator.run_async(messages=chat_messages) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        for result in results:
            assert "replies" in result
            assert isinstance(result["replies"][0], ChatMessage)
            assert result["replies"][0].text == "Berlin is cool"
        generator.close()

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_pipeline_with_tokenizer):
        """Test error handling in async context"""
        generator = TransformersChatGenerator(model="mocked-model")

        # Test with invalid streaming callback
        generator.pipeline = mock_pipeline_with_tokenizer
        with pytest.raises(ValueError, match="Using tools and streaming at the same time is not supported"):
            await generator.run_async(
                messages=[ChatMessage.from_user("test")],
                streaming_callback=lambda _: None,
                tools=[Tool(name="test", description="test", parameters={}, function=lambda: None)],
            )

    @pytest.mark.asyncio
    async def test_run_async_with_streaming_callback(self, mock_pipeline_with_tokenizer):
        streaming_chunks = []

        async def streaming_callback(chunk: StreamingChunk) -> None:
            streaming_chunks.append(chunk)

        # Create a mock that simulates streaming behavior
        def mock_pipeline_call(*args, **kwargs):
            streamer = kwargs.get("streamer")
            if streamer:
                # Simulate streaming chunks
                streamer.on_finalized_text("Berlin", stream_end=False)
                streamer.on_finalized_text(" is cool", stream_end=True)
            return [{"generated_text": "Berlin is cool"}]

        # Setup the mock pipeline with streaming simulation
        mock_pipeline_with_tokenizer.side_effect = mock_pipeline_call

        generator = TransformersChatGenerator(model="test-model", streaming_callback=streaming_callback)
        generator.pipeline = mock_pipeline_with_tokenizer

        messages = [ChatMessage.from_user("Test message")]
        response = await generator.run_async(messages)

        # Verify streaming chunks were collected
        assert len(streaming_chunks) == 2
        assert streaming_chunks[0].content == "Berlin"
        assert streaming_chunks[1].content == " is cool\n"

        # Verify the final response
        assert isinstance(response, dict)
        assert "replies" in response
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)
        assert response["replies"][0].text == "Berlin is cool"
        generator.close()


class TestIntegration:
    @pytest.mark.integration
    def test_live_run(self, del_hf_env_vars_if_empty):
        """Test live run with default behavior (no thinking)."""
        messages = [ChatMessage.from_user("Please create a summary about the following topic: Climate change")]

        llm = TransformersChatGenerator(
            model="Qwen/Qwen3-0.6B",
            generation_kwargs={"max_new_tokens": 50},
            device=ComponentDevice.from_str("cpu"),
        )

        result = llm.run(messages)

        assert "replies" in result
        assert isinstance(result["replies"][0], ChatMessage)
        assert "climate change" in result["replies"][0].text.lower()

    @pytest.mark.integration
    def test_live_run_thinking(self, del_hf_env_vars_if_empty):
        """Test live run with enable_thinking=True."""
        messages = [ChatMessage.from_user("What is 2+2?")]

        llm = TransformersChatGenerator(
            model="Qwen/Qwen3-0.6B",
            generation_kwargs={"max_new_tokens": 450},
            enable_thinking=True,
            device=ComponentDevice.from_str("cpu"),
        )

        result = llm.run(messages)

        assert "replies" in result
        assert isinstance(result["replies"][0], ChatMessage)
        reply_text = result["replies"][0].text
        assert reply_text is not None
        assert "<think>" in reply_text
        assert "</think>" in reply_text
        assert len(reply_text) > 0
        assert "4" in reply_text.lower()


class TestAsyncIntegration:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async_with_streaming(self, del_hf_env_vars_if_empty):
        """Test async streaming with a live model."""
        streaming_chunks = []

        async def streaming_callback(chunk: StreamingChunk) -> None:
            streaming_chunks.append(chunk)

        llm = TransformersChatGenerator(
            model="Qwen/Qwen3-0.6B",
            generation_kwargs={"max_new_tokens": 50},
            streaming_callback=streaming_callback,
            device=ComponentDevice.from_str("cpu"),
        )

        response = await llm.run_async(
            messages=[ChatMessage.from_user("Please create a summary about the following topic: Capital of France")]
        )

        # Verify that the response is not None
        assert len(streaming_chunks) > 0
        assert "replies" in response
        assert isinstance(response["replies"][0], ChatMessage)
        assert response["replies"][0].text is not None

        # Verify that the response contains the word "Paris"
        assert "Paris" in response["replies"][0].text

        # Verify streaming chunks contain actual content
        total_streamed_content = "".join(chunk.content for chunk in streaming_chunks)
        assert len(total_streamed_content.strip()) > 0
        assert "Paris" in total_streamed_content
