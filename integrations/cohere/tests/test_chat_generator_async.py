import os

import pytest
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.dataclasses.streaming_chunk import StreamingChunk
from haystack.tools import Tool

from haystack_integrations.components.generators.cohere.chat.chat_generator import CohereChatGenerator


def weather(city: str) -> str:
    return f"The weather in {city} is sunny and 32°C"


def stock_price(ticker: str):
    return f"The current price of {ticker} is $100"


@pytest.mark.skipif(
    not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
    reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestCohereChatGeneratorAsyncInference:
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = CohereChatGenerator(generation_kwargs={"temperature": 0.8})
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "usage" in message.meta
        assert "prompt_tokens" in message.meta["usage"]
        assert "completion_tokens" in message.meta["usage"]

    async def test_live_run_streaming_async(self):
        counter = 0
        responses = ""

        async def callback(chunk: StreamingChunk):
            nonlocal counter, responses
            assert chunk.component_info is not None
            counter += 1
            responses += chunk.content if chunk.content else ""

        component = CohereChatGenerator(streaming_callback=callback)
        results = await component.run_async([ChatMessage.from_user("What's the capital of France? answer in a word")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert message.meta["finish_reason"] == "stop"
        assert counter > 1
        assert "Paris" in responses
        assert "usage" in message.meta
        assert "prompt_tokens" in message.meta["usage"]
        assert "completion_tokens" in message.meta["usage"]

    async def test_tools_use_with_tools_async(self):
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
        response = await client.run_async(
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

    async def test_live_run_with_tools_streaming_async(self):
        """
        Test that the CohereChatGenerator can run with tools and streaming callback.
        """

        async def print_streaming_chunk_async(chunk: StreamingChunk) -> None:
            """
            Async callback function for streaming responses.
            Prints the tokens of the first completion to stdout as soon as they are received
            """
            print(chunk.content)  # noqa: T201

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
            streaming_callback=print_streaming_chunk_async,
        )
        results = await component.run_async(messages=initial_messages)

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
        results = await component.run_async(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
