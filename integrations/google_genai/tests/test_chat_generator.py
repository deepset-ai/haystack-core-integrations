#!/usr/bin/env python3

import os

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


def weather(city: str):
    """Get weather information for a city."""
    return f"Weather in {city}: 22°C, sunny"


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


class TestGoogleGenAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        assert component._model == "gemini-2.0-flash"
        assert component._generation_config == {}
        assert component._safety_settings == []
        assert component._streaming_callback is None
        assert component._tools is None

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
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gemini-2.0-flash",
            streaming_callback=print_streaming_chunk,
            generation_config={"temperature": 0.5, "max_output_tokens": 100},
            safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
            tools=[tool],
        )
        assert component._model == "gemini-2.0-flash"
        assert component._streaming_callback is print_streaming_chunk
        assert component._generation_config == {"temperature": 0.5, "max_output_tokens": 100}
        assert component._safety_settings == [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        assert component._tools == [tool]

    def test_init_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash", tools=toolset)
        assert generator._tools == toolset

    def test_to_dict_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash", tools=toolset)
        data = generator.to_dict()

        assert data["init_parameters"]["tools"]["type"] == "haystack.tools.toolset.Toolset"
        assert "tools" in data["init_parameters"]["tools"]["data"]
        assert len(data["init_parameters"]["tools"]["data"]["tools"]) == len(tools)

    def test_from_dict_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash", tools=toolset)
        data = component.to_dict()

        deserialized_component = GoogleGenAIChatGenerator.from_dict(data)

        assert isinstance(deserialized_component._tools, Toolset)
        assert len(deserialized_component._tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component._tools)

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self) -> None:
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash-001")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gemini-2.0-flash" in message.meta["model"]
        assert message.meta["finish_reason"] is not None
        # Google Gen AI may have different usage metadata structure
        assert hasattr(message, "_meta")

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_with_wrong_model(self, chat_messages):
        component = GoogleGenAIChatGenerator(model="something-obviously-wrong")
        with pytest.raises(RuntimeError):  # Google Gen AI raises RuntimeError for invalid models
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
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
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash-001", streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
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
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_calls[0]
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_google_genai_chat_generator_with_toolset_initialization(self, tools):
        """Test that GoogleGenAIChatGenerator can be initialized with a Toolset."""
        toolset = Toolset(tools)
        component = GoogleGenAIChatGenerator(tools=toolset)

        # Test that it works with a simple query
        messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        results = component.run(messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Should either have tool calls or text response
        assert message.tool_calls or message.text

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_toolset(self, tools):
        """Test that GoogleGenAIChatGenerator can run with a Toolset."""
        toolset = Toolset(tools)
        component = GoogleGenAIChatGenerator(tools=toolset)

        messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        results = component.run(messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Check if tool calls were made
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            assert tool_call.tool_name == "weather"
            assert tool_call.arguments == {"city": "Paris"}

            # Test full conversation with tool result
            tool_result_message = ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call)
            follow_up_messages = [*messages, message, tool_result_message]
            final_results = component.run(follow_up_messages)

            assert len(final_results["replies"]) == 1
            final_message = final_results["replies"][0]
            assert final_message.text
            assert "paris" in final_message.text.lower() or "weather" in final_message.text.lower()
