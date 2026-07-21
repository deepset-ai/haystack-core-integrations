# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack.dataclasses import (
    ChatMessage,
    StreamingChunk,
    ToolCall,
)
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.anthropic.chat.chat_generator import (
    AnthropicChatGenerator,
)


class TestUnit:
    async def test_run_async(self, chat_messages, mock_anthropic_completion_async, monkeypatch):
        """
        Test that the async run method of AnthropicChatGenerator works correctly.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        response = await component.run_async(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    async def test_run_async_with_string_input(self, mock_anthropic_completion_async):
        """
        Test that the async run method of AnthropicChatGenerator works with string input.
        """
        component = AnthropicChatGenerator(api_key=Secret.from_token("test-api-key"))
        result = await component.run_async("What's the capital of France?")

        # Check that the backend received exactly one user message
        _, kwargs = mock_anthropic_completion_async.call_args
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][0]["content"][0]["type"] == "text"
        assert kwargs["messages"][0]["content"][0]["text"] == "What's the capital of France?"

        # Check that the result contains exactly one ChatMessage
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) == 1
        assert isinstance(result["replies"][0], ChatMessage)

    async def test_run_async_with_params(self, chat_messages, mock_anthropic_completion_async):
        """
        Test that the async run method of AnthropicChatGenerator works with parameters.
        """
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = await component.run_async(chat_messages)

        # Check that the component calls the Anthropic API with the correct parameters
        _, kwargs = mock_anthropic_completion_async.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)
        assert "Hello! I'm Claude." in response["replies"][0].text
        assert response["replies"][0].meta["model"] == "claude-sonnet-4-5"
        assert response["replies"][0].meta["finish_reason"] == "stop"
        assert "completion_tokens" in response["replies"][0].meta["usage"]


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestIntegration:
    async def test_live_run_async(self):
        """
        Integration test that the async run method of AnthropicChatGenerator works with default parameters.
        """
        component = AnthropicChatGenerator()
        results = await component.run_async(messages=[ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "claude-sonnet-4-5" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        assert "completion_tokens" in message.meta["usage"]

    async def test_live_run_async_with_streaming(self):
        """
        Test that the async run method of AnthropicChatGenerator works with streaming.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator()  # No streaming callback during initialization

        counter = 0
        responses = ""

        # Create a callback that's compatible with async operations
        async def callback(chunk: StreamingChunk) -> None:
            nonlocal counter
            nonlocal responses
            counter += 1
            responses += chunk.content if chunk.content else ""
            assert chunk.component_info is not None
            assert chunk.component_info.type.endswith("chat_generator.AnthropicChatGenerator")

        # Run the async streaming test
        results = await component.run_async(messages=initial_messages, streaming_callback=callback)

        # Verify the results
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert "paris" in message.text.lower()
        assert "claude-sonnet-4-5" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        assert "input_tokens" in message.meta["usage"]
        assert "output_tokens" in message.meta["usage"]

        # Verify streaming behavior
        assert counter > 1  # Should have received multiple chunks
        assert "paris" in responses.lower()  # Should have received the response in chunks

    async def test_live_run_async_with_tools(self, tools):
        """
        Integration test that the async run method works with tools.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator(tools=tools)
        results = await component.run_async(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        results = await component.run_async(new_messages)
        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "completion_tokens" in final_message.meta["usage"]
