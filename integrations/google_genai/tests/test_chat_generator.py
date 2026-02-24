# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FileContent,
    ImageContent,
    ReasoningContent,
    StreamingChunk,
    TextContent,
    ToolCall,
)
from haystack.tools import Tool, Toolset, create_tool_from_function
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.google_genai.chat.chat_generator import (
    GoogleGenAIChatGenerator,
)


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


class TestGoogleGenAIChatGeneratorInitSerDe:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        assert component._model == "gemini-2.5-flash"
        assert component._generation_kwargs == {}
        assert component._safety_settings == []
        assert component._streaming_callback is None
        assert component._tools is None
        assert component._api_key is not None
        assert component._api_key.resolve_value() == "test-api-key"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            GoogleGenAIChatGenerator()

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            GoogleGenAIChatGenerator(tools=duplicate_tools)

    def test_init_with_parameters(self, monkeypatch):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=weather)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-from-env")
        component = GoogleGenAIChatGenerator(
            api_key=Secret.from_token("test-api-key-from-env"),
            model="gemini-2.5-flash",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"temperature": 0.5, "max_output_tokens": 100},
            safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
            tools=[tool],
        )
        assert component._model == "gemini-2.5-flash"
        assert component._streaming_callback is print_streaming_chunk
        assert component._generation_kwargs == {"temperature": 0.5, "max_output_tokens": 100}
        assert component._safety_settings == [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        assert component._tools == [tool]

    def test_init_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = GoogleGenAIChatGenerator(tools=toolset)
        assert generator._tools == toolset

    def test_init_with_mixed_tools_and_toolsets(self, monkeypatch):
        """Test initialization with a mixed list of Tools and Toolsets."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            function=weather,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters={"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]},
            function=weather,
        )
        tool3 = Tool(
            name="tool3",
            description="Third tool",
            parameters={"type": "object", "properties": {"z": {"type": "string"}}, "required": ["z"]},
            function=weather,
        )

        toolset1 = Toolset([tool2])

        # Initialize with mixed list: Tool, Toolset, Tool
        generator = GoogleGenAIChatGenerator(tools=[tool1, toolset1, tool3])

        assert generator._tools == [tool1, toolset1, tool3]
        assert isinstance(generator._tools, list)
        assert len(generator._tools) == 3
        assert isinstance(generator._tools[0], Tool)
        assert isinstance(generator._tools[1], Toolset)
        assert isinstance(generator._tools[2], Tool)

    def test_to_dict_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = GoogleGenAIChatGenerator(tools=toolset)
        data = generator.to_dict()

        assert data["init_parameters"]["tools"]["type"] == "haystack.tools.toolset.Toolset"
        assert "tools" in data["init_parameters"]["tools"]["data"]
        assert len(data["init_parameters"]["tools"]["data"]["tools"]) == len(tools)

    def test_from_dict_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        component = GoogleGenAIChatGenerator(tools=toolset)
        data = component.to_dict()

        deserialized_component = GoogleGenAIChatGenerator.from_dict(data)

        assert isinstance(deserialized_component._tools, Toolset)
        assert len(deserialized_component._tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component._tools)

    def test_serde_with_mixed_tools_and_toolsets(self, monkeypatch):
        """Test serialization/deserialization with mixed Tools and Toolsets."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            function=weather,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters={"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]},
            function=weather,
        )

        toolset1 = Toolset([tool2])

        generator = GoogleGenAIChatGenerator(tools=[tool1, toolset1])
        data = generator.to_dict()

        # Verify serialization preserves structure
        assert isinstance(data["init_parameters"]["tools"], list)
        assert len(data["init_parameters"]["tools"]) == 2
        assert data["init_parameters"]["tools"][0]["type"].endswith("Tool")
        assert data["init_parameters"]["tools"][1]["type"].endswith("Toolset")

        # Verify deserialization
        restored = GoogleGenAIChatGenerator.from_dict(data)
        assert isinstance(restored._tools, list)
        assert len(restored._tools) == 2
        assert isinstance(restored._tools[0], Tool)
        assert isinstance(restored._tools[1], Toolset)
        assert restored._tools[0].name == "tool1"
        assert len(restored._tools[1]) == 1


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY", None),
    reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
)
@pytest.mark.integration
class TestGoogleGenAIChatGeneratorInference:
    def test_live_run(self) -> None:
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = GoogleGenAIChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert "gemini-2.5-flash" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    def test_run_with_multiple_images_mixed_content(self, test_files_path):
        """Test that multiple images with interleaved text maintain proper ordering."""
        client = GoogleGenAIChatGenerator()

        # Load both test images
        apple_path = test_files_path / "apple.jpg"
        banana_path = test_files_path / "banana.png"

        apple_content = ImageContent.from_file_path(apple_path, size=(100, 100))
        banana_content = ImageContent.from_file_path(banana_path, size=(100, 100))

        # Create message with interleaved text and images to test ordering preservation
        chat_message = ChatMessage.from_user(
            content_parts=[
                "Here are two fruits. First image:",
                apple_content,
                "Second image:",
                banana_content,
                "What fruits do you see? List them in order.",
            ]
        )

        response = client.run([chat_message])

        first_reply = response["replies"][0]
        assert isinstance(first_reply, ChatMessage)
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT)
        assert first_reply.text

        # Verify both fruits are mentioned in the response
        response_text = first_reply.text.lower()
        assert "apple" in response_text, "Apple should be mentioned in the response"
        assert "banana" in response_text, "Banana should be mentioned in the response"

        # Verify that apple is mentioned before banana (preserving our input order)
        apple_pos = response_text.find("apple")
        banana_pos = response_text.find("banana")
        assert apple_pos < banana_pos, (
            f"Apple should be mentioned before banana in the response. Got: {first_reply.text}"
        )

    def test_live_run_with_file_content(self, test_files_path):
        pdf_path = test_files_path / "sample_pdf_3.pdf"

        file_content = FileContent.from_file_path(file_path=pdf_path)

        chat_messages = [
            ChatMessage.from_user(
                content_parts=[file_content, "Is this document a paper about LLMs? Respond with 'yes' or 'no' only."]
            )
        ]

        generator = GoogleGenAIChatGenerator()
        results = generator.run(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]

        assert message.is_from(ChatRole.ASSISTANT)

        assert message.text
        indicates_no = any(
            phrase in message.text.lower()
            for phrase in (
                "no",
                "nope",
                "not about",
                "not a paper about",
                "it is not",
                "it's not",
                "the answer is no",
                "does not",
                "doesn't",
                "negative",
            )
        )

        assert indicates_no is True

    def test_live_run_streaming(self):
        component = GoogleGenAIChatGenerator()
        component_info = ComponentInfo.from_component(component)

        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""
                assert chunk.component_info == component_info

        callback = Callback()

        results = component.run(
            messages=[ChatMessage.from_user("What's the capital of France?")], streaming_callback=callback
        )

        assert len(results["replies"]) == 1
        assert callback.counter > 0, "No streaming chunks received"
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert message.meta["finish_reason"] == "stop"

    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the GoogleGenAIChatGenerator component can run with tools and streaming.
        """
        component = GoogleGenAIChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
        results = component.run([ChatMessage.from_user("What's the weather like in Paris?")])

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
        # Google Gen AI (gemini-2.5-flash and gemini-2.5-pro-preview-05-06) does not provide ids for tool calls although
        # it is in the response schema, revisit in future to see if there are changes and id is provided
        # assert tool_message.tool_calls[0].id is not None, "Tool call has no id"
        assert tool_message.tool_calls[0].tool_name == "weather"
        assert tool_message.tool_calls[0].arguments == {"city": "Paris"}

        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"
        assert tool_message.meta["finish_reason"] == "stop"

        tool_call = tool_message.tool_calls[0]
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

    def test_live_run_with_toolset(self, tools):
        """Test that GoogleGenAIChatGenerator can run with a Toolset."""
        toolset = Toolset(tools)
        component = GoogleGenAIChatGenerator(tools=toolset)

        messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        results = component.run(messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Check if tool calls were made
        assert message.tool_calls is not None, "Message has no tool calls"
        assert len(message.tool_calls) == 1, "Message has multiple tool calls and it should only have one"
        tool_call = message.tool_calls[0]
        # Google Gen AI (gemini-2.5-flash and gemini-2.5-pro-preview-05-06) does not provide ids for tool calls although
        # it is in the response schema, revisit in future to see if there are changes and id is provided
        # assert tool_call.id is not None, "Tool call has no id"
        assert message.meta["finish_reason"] == "stop"

        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

        # Test full conversation with tool result
        tool_result_message = ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call)
        follow_up_messages = [*messages, message, tool_result_message]
        final_results = component.run(follow_up_messages)

        assert len(final_results["replies"]) == 1
        final_message = final_results["replies"][0]
        assert final_message.text
        assert "paris" in final_message.text.lower() or "weather" in final_message.text.lower(), (
            "Response does not contain Paris or weather"
        )

    def test_live_run_with_parallel_tools(self, tools):
        """
        Integration test that the GoogleGenAIChatGenerator component can run with parallel tools.
        """
        initial_messages = [
            ChatMessage.from_user(
                "What's the weather like in Paris and Berlin? Produce a separate tool call for each city."
            )
        ]
        component = GoogleGenAIChatGenerator(model="gemini-3-flash-preview", tools=tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.meta["finish_reason"] == "stop"

        # Google GenAI should make tool calls for both cities
        assert len(message.tool_calls) == 2
        paris_index = 0 if message.tool_calls[0].arguments["city"].lower() == "paris" else 1
        tool_call_paris = message.tool_calls[paris_index]
        assert isinstance(tool_call_paris, ToolCall)
        assert tool_call_paris.tool_name == "weather"
        assert tool_call_paris.arguments["city"].lower() == "paris"

        tool_call_berlin = message.tool_calls[1 - paris_index]
        assert isinstance(tool_call_berlin, ToolCall)
        assert tool_call_berlin.tool_name == "weather"
        assert tool_call_berlin.arguments["city"].lower() == "berlin"

        # Google GenAI expects results from both tools in separate messages
        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call_paris),
            ChatMessage.from_tool(tool_result="15°C, cloudy", origin=tool_call_berlin),
        ]

        # Response from the model contains results from both tools
        results = component.run(new_messages)
        final_message = results["replies"][0]

        assert not final_message.tool_calls, "Message has tool calls and it should not have any"
        assert len(final_message.text) > 0, "Message has no text"
        assert final_message.text and "paris" in final_message.text.lower() and "berlin" in final_message.text.lower()
        # Check that the response mentions both temperature readings
        assert "22" in final_message.text and "15" in final_message.text

    def test_live_run_with_mixed_tools(self):
        """
        Integration test that verifies GoogleGenAIChatGenerator works with mixed Tool and Toolset.
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
        component = GoogleGenAIChatGenerator(tools=mixed_tools)
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
            # Google GenAI may not provide IDs for tool calls
            assert tool_call.tool_name in ["weather", "population"]
            assert "city" in tool_call.arguments
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert first_reply.meta["finish_reason"] == "stop"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = []
        for tool_call in tool_calls:
            if tool_call.tool_name == "weather":
                result = "The weather in Paris is sunny and 32°C"
            else:  # population
                result = "The population of Berlin is 2.2 million"
            tool_result_messages.append(ChatMessage.from_tool(tool_result=result, origin=tool_call))

        new_messages = [*initial_messages, first_reply, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    def test_live_run_with_thinking(self):
        """
        Integration test for the thinking feature with a model that supports it.
        """
        # We use a model that supports the thinking feature
        chat_messages = [ChatMessage.from_user("Why is the sky blue? Explain in one sentence.")]
        component = GoogleGenAIChatGenerator(
            model="gemini-3-flash-preview", generation_kwargs={"thinking_level": "low"}
        )
        results = component.run(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text

        # Check for reasoning content, which should be present when thinking is enabled
        assert message.reasonings
        assert len(message.reasonings) > 0
        assert all(isinstance(r, ReasoningContent) for r in message.reasonings)

        # Check for thinking token usage
        assert "usage" in message.meta
        assert "thoughts_token_count" in message.meta["usage"]
        assert message.meta["usage"]["thoughts_token_count"] is not None
        assert message.meta["usage"]["thoughts_token_count"] > 0

    def test_live_run_with_thinking_and_tools_multi_turn(self, tools):
        """
        Integration test for thought signatures preservation in multi-turn conversations with tools.
        This verifies that thought context is maintained across turns when using tools with thinking.
        """
        # Use a model that supports thinking with tools
        component = GoogleGenAIChatGenerator(
            model="gemini-3-flash-preview",
            tools=tools,
            generation_kwargs={"thinking_level": "low"},  # Dynamic allocation
        )

        # First turn: Ask about the weather
        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        result = component.run(messages)

        assert len(result["replies"]) == 1
        first_response = result["replies"][0]

        # Should have tool calls
        assert first_response.tool_calls
        assert len(first_response.tool_calls) == 1
        assert first_response.tool_calls[0].tool_name == "weather"

        # Check for thought signatures in meta (only present with tools)
        assert "thought_signatures" in first_response.meta
        assert len(first_response.meta["thought_signatures"]) > 0

        # Second turn: Provide tool result and continue conversation
        tool_call = first_response.tool_calls[0]
        messages.extend(
            [
                first_response,  # Include the assistant's response with thought signatures
                ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call),
                ChatMessage.from_user("Is that good weather for a picnic?"),
            ]
        )

        # The thought signatures from first_response should be preserved automatically
        result2 = component.run(messages)

        assert len(result2["replies"]) == 1
        second_response = result2["replies"][0]

        # check that the thought signatures are there
        assert "thought_signatures" in second_response.meta
        assert len(second_response.meta["thought_signatures"]) > 0

        # Should have a text response about picnic weather
        assert second_response.text
        assert "picnic" in second_response.text.lower() or "yes" in second_response.text.lower()

        # The model should maintain context from previous turns
        assert "22" in second_response.text or "sunny" in second_response.text.lower()

    def test_live_run_with_thinking_unsupported_model_fails_fast(self):
        """
        Integration test to verify that thinking configuration fails fast with unsupported models.
        """
        # gemini-2.0-flash does not support thinking
        chat_messages = [ChatMessage.from_user("Why is the sky blue?")]
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash", generation_kwargs={"thinking_budget": 1024})

        # The call should raise a RuntimeError with a helpful message
        with pytest.raises(RuntimeError) as exc_info:
            component.run(chat_messages)

        # Verify the error message is helpful and mentions thinking configuration
        error_message = str(exc_info.value)
        assert "Thinking configuration error" in error_message
        assert "gemini-2.0" in error_message
        assert "thinking_budget" in error_message or "thinking features" in error_message
        assert "Try removing" in error_message or "use a different model" in error_message

    def test_live_run_agent_with_images_in_tool_result(self, test_files_path):
        def retrieve_image():
            return [
                TextContent("Here is the retrieved image."),
                ImageContent.from_file_path(test_files_path / "apple.jpg", size=(100, 100)),
            ]

        image_retriever_tool = create_tool_from_function(
            name="retrieve_image", description="Tool to retrieve an image", function=retrieve_image
        )
        image_retriever_tool.outputs_to_string = {"raw_result": True}

        agent = Agent(
            chat_generator=GoogleGenAIChatGenerator(model="gemini-3-flash-preview"),
            system_prompt="You are an Agent that can retrieve images and describe them.",
            tools=[image_retriever_tool],
        )

        user_message = ChatMessage.from_user("Retrieve the image and describe it in max 5 words.")
        result = agent.run(messages=[user_message])

        assert "apple" in result["last_message"].text.lower()


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY", None),
    reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncGoogleGenAIChatGeneratorInference:
    """Test class for async functionality of GoogleGenAIChatGenerator."""

    async def test_live_run_async(self) -> None:
        """Test async version of the run method."""
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = GoogleGenAIChatGenerator()
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert "gemini-2.5-flash" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    async def test_live_run_async_streaming(self):
        """Test async version with streaming."""
        responses = ""
        counter = 0

        async def async_callback(chunk: StreamingChunk) -> None:
            nonlocal counter, responses
            counter += 1
            responses += chunk.content if chunk.content else ""

        component = GoogleGenAIChatGenerator()
        results = await component.run_async(
            [ChatMessage.from_user("What's the capital of France?")], streaming_callback=async_callback
        )

        assert len(results["replies"]) == 1
        assert counter > 0, "No streaming chunks received"
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert message.meta["finish_reason"] == "stop"

    async def test_live_run_async_with_tools(self, tools):
        """Test async version with tools."""
        component = GoogleGenAIChatGenerator(tools=tools)
        results = await component.run_async([ChatMessage.from_user("What's the weather like in Paris?")])

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
        assert tool_message.meta["finish_reason"] == "stop"

    async def test_live_run_async_with_thinking(self):
        """
        Async integration test for the thinking feature.
        """
        chat_messages = [ChatMessage.from_user("Why is the sky blue? Explain in one sentence.")]
        component = GoogleGenAIChatGenerator(
            model="gemini-3-flash-preview", generation_kwargs={"thinking_level": "low"}
        )
        results = await component.run_async(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text

        # Check for reasoning content
        assert message.reasonings
        assert len(message.reasonings) > 0
        assert all(isinstance(r, ReasoningContent) for r in message.reasonings)

        # Check for thinking token usage
        assert "usage" in message.meta
        assert "thoughts_token_count" in message.meta["usage"]
        assert message.meta["usage"]["thoughts_token_count"] is not None
        assert message.meta["usage"]["thoughts_token_count"] > 0

    async def test_live_run_async_with_thinking_unsupported_model_fails_fast(self):
        """
        Async integration test to verify that thinking configuration fails fast with unsupported models.
        This tests the fail-fast principle - no silent fallbacks.
        """
        # Use a model that does NOT support thinking features (gemini-2.0-flash)
        chat_messages = [ChatMessage.from_user("Why is the sky blue?")]
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash", generation_kwargs={"thinking_budget": 1024})

        # The call should raise a RuntimeError with a helpful message
        with pytest.raises(RuntimeError) as exc_info:
            await component.run_async(chat_messages)

        # Verify the error message is helpful and mentions thinking configuration
        error_message = str(exc_info.value)
        assert "Thinking configuration error" in error_message
        assert "gemini-2.0" in error_message
        assert "thinking_budget" in error_message or "thinking features" in error_message
        assert "Try removing" in error_message or "use a different model" in error_message

    async def test_concurrent_async_calls(self):
        """Test multiple concurrent async calls."""
        component = GoogleGenAIChatGenerator()

        # Create multiple tasks
        tasks = []
        for i in range(3):
            messages = [ChatMessage.from_user(f"What's the capital of country number {i + 1}? Just say the city name.")]
            task = component.run_async(messages)
            tasks.append(task)

        # Run concurrently
        results = await asyncio.gather(*tasks)

        # Verify all calls completed successfully
        assert len(results) == 3
        for result in results:
            assert len(result["replies"]) == 1
            assert result["replies"][0].text
            assert result["replies"][0].meta["model"]
            assert result["replies"][0].meta["finish_reason"] == "stop"
