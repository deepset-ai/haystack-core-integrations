import os
from unittest.mock import MagicMock

import pytest
from cohere import UserChatMessageV2
from cohere.core import ApiError
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, ImageContent, ReasoningContent, ToolCall
from haystack.dataclasses.streaming_chunk import StreamingChunk
from haystack.tools import Tool
from haystack.utils import Secret

from haystack_integrations.components.generators.cohere import CohereChatGenerator
from haystack_integrations.components.generators.cohere.chat.chat_generator import (
    _extract_reasoning_from_response,
    _format_message,
)


def weather(city: str) -> str:
    return f"The weather in {city} is sunny and 32°C"


def stock_price(ticker: str):
    return f"The current price of {ticker} is $100"


class TestFormatMessage:
    def test_format_message_empty_message_raises_error(self):
        message = ChatMessage.from_user("")

        with pytest.raises(ValueError):
            _format_message(message)

    def test_format_message_tool_call_result_with_none_id_raises_error(self):
        tool_call = ToolCall(id=None, tool_name="test_tool", arguments={})

        message = ChatMessage.from_tool(tool_result="test result", origin=tool_call, error=False)

        with pytest.raises(ValueError):
            _format_message(message)

    def test_format_message_tool_call_with_none_id_raises_error(self):
        tool_call = ToolCall(id=None, tool_name="test_tool", arguments={})

        message = ChatMessage.from_assistant("", tool_calls=[tool_call])

        with pytest.raises(ValueError):
            _format_message(message)

    def test_format_message_with_image(self):
        """Test that a ChatMessage with ImageContent is converted to Cohere format correctly."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        formatted_message = _format_message(message)

        assert isinstance(formatted_message, UserChatMessageV2)
        assert formatted_message.role == "user"
        assert isinstance(formatted_message.content, list)
        assert len(formatted_message.content) == 2

        # Check text content
        assert formatted_message.content[0].type == "text"
        assert formatted_message.content[0].text == "What's in this image?"

        # Check image content
        assert formatted_message.content[1].type == "image_url"
        assert hasattr(formatted_message.content[1], "image_url")
        assert hasattr(formatted_message.content[1].image_url, "url")
        assert formatted_message.content[1].image_url.url == f"data:image/png;base64,{base64_image}"

    def test_format_message_with_unsupported_mime_type(self):
        """Test that a ChatMessage with unsupported mime type raises ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/bmp")
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        with pytest.raises(ValueError, match="Unsupported image format: image/bmp"):
            _format_message(message)

    def test_format_message_with_none_mime_type(self):
        """Test that a ChatMessage with None mime type raises ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        # Manually set mime_type to None to test the edge case
        image_content.mime_type = None
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        with pytest.raises(ValueError, match="Unsupported image format: None"):
            _format_message(message)

    def test_format_message_image_in_non_user_message(self):
        """Test that images in non-user messages raise ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        # Create assistant message with both text and image (should fail because image in assistant message)
        message = ChatMessage.from_assistant(text="Here's an image.")
        message._content.append(image_content)  # Add image to assistant message

        with pytest.raises(ValueError, match=r"`ImageContent` is only supported for user messages\."):
            _format_message(message)

    def test_supported_image_formats(self):
        """Test that all supported image formats work correctly."""
        supported_formats = ["image/png", "image/jpeg", "image/webp", "image/gif"]
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

        for mime_type in supported_formats:
            image_content = ImageContent(base64_image=base64_image, mime_type=mime_type)
            message = ChatMessage.from_user(content_parts=["Test image", image_content])

            # Should not raise any exception
            formatted_message = _format_message(message)
            assert formatted_message is not None
            assert isinstance(formatted_message, UserChatMessageV2)

    def test_multiple_images_in_single_message(self):
        """Test handling multiple images in a single message."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image1 = ImageContent(base64_image=base64_image, mime_type="image/png")
        image2 = ImageContent(base64_image=base64_image, mime_type="image/jpeg")

        message = ChatMessage.from_user(content_parts=["Compare these images:", image1, image2])

        formatted_message = _format_message(message)

        assert isinstance(formatted_message, UserChatMessageV2)
        assert len(formatted_message.content) == 3  # 1 text + 2 images
        assert formatted_message.content[0].type == "text"
        assert formatted_message.content[1].type == "image_url"
        assert formatted_message.content[2].type == "image_url"


class TestCohereChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")

        component = CohereChatGenerator()
        assert component.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert component.model == "command-r-08-2024"
        assert component.streaming_callback is None
        assert component.api_base_url == "https://api.cohere.com"
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        with pytest.raises(ValueError):
            CohereChatGenerator()

    def test_init_with_parameters(self):
        component = CohereChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="command-nightly",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={
                "max_tokens": 10,
                "some_test_param": "test-params",
            },
        )
        assert component.api_key == Secret.from_token("test-api-key")
        assert component.model == "command-nightly"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component = CohereChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-r-08-2024",
                "streaming_callback": None,
                "api_key": {
                    "env_vars": ["COHERE_API_KEY", "CO_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "api_base_url": "https://api.cohere.com",
                "generation_kwargs": {},
                "tools": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        monkeypatch.setenv("CO_API_KEY", "fake-api-key")
        component = CohereChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="command-nightly",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={
                "max_tokens": 10,
                "some_test_param": "test-params",
            },
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-nightly",
                "api_key": {
                    "env_vars": ["ENV_VAR"],
                    "strict": False,
                    "type": "env_var",
                },
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "api_base_url": "test-base-url",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
                "tools": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "fake-api-key")
        monkeypatch.setenv("CO_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-r-08-2024",
                "api_base_url": "test-base-url",
                "api_key": {
                    "env_vars": ["ENV_VAR"],
                    "strict": False,
                    "type": "env_var",
                },
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
            },
        }
        component = CohereChatGenerator.from_dict(data)
        assert component.model == "command-r-08-2024"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-r-08-2024",
                "api_base_url": "test-base-url",
                "api_key": {
                    "env_vars": ["COHERE_API_KEY", "CO_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
            },
        }
        with pytest.raises(ValueError):
            CohereChatGenerator.from_dict(data)

    def test_serde_in_pipeline(self, monkeypatch):
        """
        Test serialization/deserialization of CohereChatGenerator in a Pipeline,
        including detailed dictionary validation
        """
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")

        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        generator = CohereChatGenerator(
            model="command-r-08-2024",
            generation_kwargs={"temperature": 0.7},
            streaming_callback=print_streaming_chunk,
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()

        expected_dict = {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",  # noqa: E501
                    "init_parameters": {
                        "model": "command-r-08-2024",
                        "api_key": {"type": "env_var", "env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True},
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "api_base_url": "https://api.cohere.com",
                        "generation_kwargs": {"temperature": 0.7},
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather",
                                    "description": "useful to determine the weather in a given location",
                                    "parameters": {"city": {"type": "string"}},
                                    "function": "tests.test_chat_generator.weather",
                                    "outputs_to_string": tool.outputs_to_string,
                                    "inputs_from_state": tool.inputs_from_state,
                                    "outputs_to_state": tool.outputs_to_state,
                                },
                            }
                        ],
                    },
                }
            },
            "connections": [],
        }

        assert pipeline_dict == expected_dict

        # Test YAML serialization/deserialization
        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

        # Verify the loaded pipeline's generator has the same configuration
        loaded_generator = new_pipeline.get_component("generator")
        assert loaded_generator.model == generator.model
        assert loaded_generator.generation_kwargs == generator.generation_kwargs
        assert loaded_generator.streaming_callback == generator.streaming_callback
        assert len(loaded_generator.tools) == len(generator.tools)
        assert loaded_generator.tools[0].name == generator.tools[0].name
        assert loaded_generator.tools[0].description == generator.tools[0].description
        assert loaded_generator.tools[0].parameters == generator.tools[0].parameters

    def test_run_image(self):
        """Test multimodal message processing with mocked client."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        messages = [ChatMessage.from_user(content_parts=["What's in this image?", image_content])]

        generator = CohereChatGenerator(api_key=Secret.from_token("test-api-key"))

        # Mock the client's chat method
        mock_response = MagicMock()
        mock_response.message.content = [MagicMock()]
        mock_response.message.content[0].text = "This is a test image response"
        mock_response.message.tool_calls = None
        mock_response.finish_reason = "COMPLETE"
        mock_response.usage = None

        generator.client.chat = MagicMock(return_value=mock_response)

        result = generator.run(messages=messages)

        # Verify the multimodal message was processed correctly
        assert "replies" in result
        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "This is a test image response"

        # Verify the client was called with the correct format
        generator.client.chat.assert_called_once()
        call_args = generator.client.chat.call_args
        formatted_messages = call_args[1]["messages"]

        assert len(formatted_messages) == 1
        # The multimodal message should be passed as a Cohere object
        multimodal_msg = formatted_messages[0]

        assert isinstance(multimodal_msg, UserChatMessageV2)
        assert multimodal_msg.role == "user"
        assert len(multimodal_msg.content) == 2
        assert multimodal_msg.content[0].type == "text"
        assert multimodal_msg.content[1].type == "image_url"


@pytest.mark.skipif(
    not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
    reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
)
@pytest.mark.integration
class TestCohereChatGeneratorInference:
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = CohereChatGenerator(generation_kwargs={"temperature": 0.8})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "usage" in message.meta
        assert "prompt_tokens" in message.meta["usage"]
        assert "completion_tokens" in message.meta["usage"]

    def test_live_run_wrong_model(self):
        component = CohereChatGenerator(model="something-obviously-wrong")
        with pytest.raises(ApiError):
            component.run([ChatMessage.from_assistant("What's the capital of France")])

    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                assert chunk.component_info is not None
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = CohereChatGenerator(streaming_callback=callback, stream=True)
        results = component.run([ChatMessage.from_user("What's the capital of France? answer in a word")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert message.meta["finish_reason"] == "stop"
        assert callback.counter > 1
        assert "Paris" in callback.responses
        assert "usage" in message.meta
        assert "prompt_tokens" in message.meta["usage"]
        assert "completion_tokens" in message.meta["usage"]

    def test_tools_use_old_way(self):
        # See https://docs.cohere.com/docs/structured-outputs-json for more information
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Retrieves the current stock price for a given ticker symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                            }
                        },
                        "required": ["ticker"],
                    },
                },
            }
        ]
        client = CohereChatGenerator(model="command-r-08-2024")
        response = client.run(
            messages=[ChatMessage.from_user("What is the current price of AAPL?")],
            generation_kwargs={"tools": tools_schema},
        )
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply text should be a tool plan"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"

        assert first_reply.tool_calls, "First reply has no tool calls"
        assert len(first_reply.tool_calls) == 1, "First reply has more than one tool call"
        assert first_reply.tool_calls[0].tool_name == "get_stock_price", "First tool call is not get_stock_price"
        assert first_reply.tool_calls[0].arguments == {"ticker": "AAPL"}, "First tool call arguments are not correct"

    def test_tools_use_with_tools(self):
        stock_price_tool = Tool(
            name="get_stock_price",
            description="Retrieves the current stock price for a given ticker symbol.",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                    }
                },
                "required": ["ticker"],
            },
            function=stock_price,
        )
        initial_messages = [ChatMessage.from_user("What is the current price of AAPL?")]
        client = CohereChatGenerator(model="command-r-08-2024")
        response = client.run(
            messages=initial_messages,
            tools=[stock_price_tool],
        )
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply text should be a tool plan"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"

        assert first_reply.tool_calls, "First reply has no tool calls"
        assert len(first_reply.tool_calls) == 1, "First reply has more than one tool call"
        assert first_reply.tool_calls[0].tool_name == "get_stock_price", "First tool call is not get_stock_price"
        assert first_reply.tool_calls[0].arguments == {"ticker": "AAPL"}, "First tool call arguments are not correct"

        # Test with tool result
        new_messages = [
            initial_messages[0],
            first_reply,
            ChatMessage.from_tool(tool_result="150.23", origin=first_reply.tool_calls[0]),
        ]
        results = client.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "150.23" in final_message.text

    def test_live_run_with_tools_streaming(self):
        """
        Test that the CohereChatGenerator can run with tools and streaming callback.
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

        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = CohereChatGenerator(
            model="command-r-08-2024",  # Cohere's model that supports tools
            tools=[weather_tool],
            streaming_callback=print_streaming_chunk,
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"
        first_reply = results["replies"][0]

        assert isinstance(first_reply, ChatMessage), "Reply is not a ChatMessage instance"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "Reply is not from the assistant"
        assert first_reply.tool_calls, "No tool calls in the reply"

        tool_call = first_reply.tool_calls[0]
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

        # Test with tool result
        new_messages = [
            initial_messages[0],
            first_reply,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    def test_pipeline_with_cohere_chat_generator(self):
        """
        Test that the CohereChatGenerator component can be used in a pipeline
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

        pipeline = Pipeline()
        pipeline.add_component("generator", CohereChatGenerator(model="command-r-08-2024", tools=[weather_tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[weather_tool]))

        pipeline.connect("generator", "tool_invoker")

        results = pipeline.run(
            data={"generator": {"messages": [ChatMessage.from_user("What's the weather like in Paris?")]}}
        )

        assert (
            "The weather in Paris is sunny and 32°C"
            == results["tool_invoker"]["tool_messages"][0].tool_call_result.result
        )

    def test_live_run_multimodal(self):
        generator = CohereChatGenerator(
            model="command-a-vision-07-2025",  # Use a vision model
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


class TestReasoningExtraction:
    """Test the reasoning extraction functionality."""

    def test_extract_reasoning_with_thinking_tags(self):
        """Test extraction of reasoning from <thinking> tags."""
        response_text = """<thinking>
I need to calculate the area of a circle.
The formula is π * r².
Given radius is 5, so area = π * 25 = 78.54
</thinking>

The area of a circle with radius 5 is approximately 78.54 square units."""

        reasoning, cleaned = _extract_reasoning_from_response(response_text)

        assert reasoning is not None
        assert isinstance(reasoning, ReasoningContent)
        assert "calculate the area of a circle" in reasoning.reasoning_text
        assert "formula is π * r²" in reasoning.reasoning_text
        assert "area = π * 25 = 78.54" in reasoning.reasoning_text
        assert cleaned.strip() == "The area of a circle with radius 5 is approximately 78.54 square units."

    def test_extract_reasoning_with_reasoning_tags(self):
        """Test extraction of reasoning from <reasoning> tags."""
        response_text = """<reasoning>
Let me think about this step by step:
1. First, I need to understand the problem
2. Then identify the key variables
3. Apply the appropriate formula
</reasoning>

Based on my analysis, here's the solution."""

        reasoning, cleaned = _extract_reasoning_from_response(response_text)

        assert reasoning is not None
        assert isinstance(reasoning, ReasoningContent)
        assert "step by step" in reasoning.reasoning_text
        assert "understand the problem" in reasoning.reasoning_text
        assert "key variables" in reasoning.reasoning_text
        assert cleaned.strip() == "Based on my analysis, here's the solution."

    def test_extract_reasoning_with_step_by_step_headers(self):
        """Test extraction of reasoning from step-by-step format."""
        response_text = """## My reasoning:
Step 1: Analyze the input data
Step 2: Identify patterns
Step 3: Apply the algorithm

## Solution:
The final answer is 42."""

        reasoning, cleaned = _extract_reasoning_from_response(response_text)

        assert reasoning is not None
        assert isinstance(reasoning, ReasoningContent)
        assert "Step 1: Analyze the input data" in reasoning.reasoning_text
        assert "Step 2: Identify patterns" in reasoning.reasoning_text
        assert "Step 3: Apply the algorithm" in reasoning.reasoning_text
        assert cleaned.strip() == "## Solution:\nThe final answer is 42."

    def test_extract_reasoning_no_reasoning_present(self):
        """Test that no reasoning is extracted when none is present."""
        response_text = "This is a simple response without any reasoning content."

        reasoning, cleaned = _extract_reasoning_from_response(response_text)

        assert reasoning is None
        assert cleaned == response_text

    def test_extract_reasoning_short_reasoning_ignored(self):
        """Test that very short reasoning content is ignored."""
        response_text = """<thinking>
OK
</thinking>

The answer is yes."""

        reasoning, cleaned = _extract_reasoning_from_response(response_text)

        assert reasoning is None  # Too short, should be ignored
        assert cleaned.strip() == "The answer is yes."

    def test_extract_reasoning_with_let_me_think(self):
        """Test extraction of reasoning starting with 'Let me think'."""
        response_text = """Let me think through this carefully:

First, I need to consider the constraints of the problem. The user is asking about quantum
mechanics, which requires understanding wave-particle duality.

Second, I should explain the fundamental principles clearly.

Based on this analysis, quantum mechanics describes the behavior of matter and energy at the atomic scale."""

        reasoning, cleaned = _extract_reasoning_from_response(response_text)

        assert reasoning is not None
        assert isinstance(reasoning, ReasoningContent)
        assert "think through this carefully" in reasoning.reasoning_text
        assert "constraints of the problem" in reasoning.reasoning_text
        assert "wave-particle duality" in reasoning.reasoning_text
        assert cleaned.strip() == (
            "Based on this analysis, quantum mechanics describes the behavior of matter and energy at the atomic scale."
        )


class TestCohereChatGeneratorReasoning:
    """Integration tests for reasoning functionality in CohereChatGenerator."""

    @pytest.mark.skipif(not os.environ.get("COHERE_API_KEY"), reason="COHERE_API_KEY not set")
    @pytest.mark.integration
    def test_reasoning_with_command_a_reasoning_model(self):
        """Test reasoning extraction with Command A Reasoning model."""
        generator = CohereChatGenerator(
            model="command-a-reasoning-111b-2024-10-03",
            generation_kwargs={"thinking": True},  # Enable reasoning
        )

        messages = [
            ChatMessage.from_user("Solve this math problem step by step: What is the area of a circle with radius 7?")
        ]

        result = generator.run(messages=messages)

        assert "replies" in result
        assert len(result["replies"]) == 1

        reply = result["replies"][0]
        assert isinstance(reply, ChatMessage)
        assert reply.role == ChatRole.ASSISTANT

        # Check if reasoning was extracted
        if reply.reasoning:
            assert isinstance(reply.reasoning, ReasoningContent)
            assert len(reply.reasoning.reasoning_text) > 50  # Should have substantial reasoning

            # The reasoning should contain mathematical thinking
            reasoning_lower = reply.reasoning.reasoning_text.lower()
            assert any(word in reasoning_lower for word in ["area", "circle", "radius", "formula", "π", "pi"])

        # Check the main response content
        assert len(reply.text) > 0
        response_lower = reply.text.lower()
        assert any(word in response_lower for word in ["area", "153.94", "154", "square"])

    def test_reasoning_with_mock_response(self):
        """Test reasoning extraction with mocked Cohere response."""
        generator = CohereChatGenerator(
            model="command-a-reasoning-111b-2024-10-03", api_key=Secret.from_token("fake-api-key")
        )

        # Mock the Cohere client response
        mock_response = MagicMock()
        mock_response.message.content = [
            MagicMock(
                text="""<thinking>
I need to solve for the area of a circle.
The formula is A = πr²
With radius 7: A = π * 7² = π * 49 ≈ 153.94
</thinking>

The area of a circle with radius 7 is approximately 153.94 square units."""
            )
        ]
        mock_response.message.tool_calls = None
        mock_response.message.citations = None

        generator.client.chat = MagicMock(return_value=mock_response)

        messages = [ChatMessage.from_user("What is the area of a circle with radius 7?")]
        result = generator.run(messages=messages)

        assert "replies" in result
        assert len(result["replies"]) == 1

        reply = result["replies"][0]
        assert isinstance(reply, ChatMessage)
        assert reply.role == ChatRole.ASSISTANT

        # Check reasoning extraction
        assert reply.reasoning is not None
        assert isinstance(reply.reasoning, ReasoningContent)
        assert "formula is A = πr²" in reply.reasoning.reasoning_text
        assert "π * 49 ≈ 153.94" in reply.reasoning.reasoning_text

        # Check cleaned content
        assert reply.text.strip() == "The area of a circle with radius 7 is approximately 153.94 square units."

    def test_reasoning_with_tool_calls_compatibility(self):
        """Test that reasoning works with tool calls."""
        weather_tool = Tool(
            name="weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            function=weather,
        )

        generator = CohereChatGenerator(
            model="command-a-reasoning-111b-2024-10-03", tools=[weather_tool], api_key=Secret.from_token("fake-api-key")
        )

        # Mock response with both reasoning and tool calls
        mock_response = MagicMock()
        mock_response.message.content = [
            MagicMock(
                text="""<thinking>
The user is asking about weather in Paris. I should use the weather tool to get accurate information.
</thinking>

I'll check the weather in Paris for you."""
            )
        ]

        # Mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "weather"
        mock_tool_call.function.arguments = '{"city": "Paris"}'
        mock_tool_call.id = "call_123"
        mock_response.message.tool_calls = [mock_tool_call]
        mock_response.message.tool_plan = "I'll check the weather in Paris for you."
        mock_response.message.citations = None

        generator.client.chat = MagicMock(return_value=mock_response)

        messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        result = generator.run(messages=messages)

        assert "replies" in result
        assert len(result["replies"]) == 1

        reply = result["replies"][0]
        assert isinstance(reply, ChatMessage)

        # Check reasoning extraction
        assert reply.reasoning is not None
        assert isinstance(reply.reasoning, ReasoningContent)
        assert "weather tool" in reply.reasoning.reasoning_text

        # Check tool calls are preserved
        assert reply.tool_calls is not None
        assert len(reply.tool_calls) == 1
        assert reply.tool_calls[0].tool_name == "weather"

        # Check cleaned content
        assert "I'll check the weather in Paris" in reply.text
