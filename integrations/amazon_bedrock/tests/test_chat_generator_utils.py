import pytest
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, ToolCall
from haystack.tools import Tool

from haystack_integrations.components.generators.amazon_bedrock.chat.utils import (
    _format_messages,
    _format_tools,
    _parse_completion_response,
    _parse_streaming_response,
)


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


def addition(a: int, b: int):
    """Add two numbers."""
    return a + b


@pytest.fixture
def tools():
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=weather,
    )
    addition_tool = Tool(
        name="addition",
        description="useful to add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=addition,
    )
    return [weather_tool, addition_tool]


class TestAmazonBedrockChatGeneratorUtils:
    def test_format_tools(self, tools):
        formatted_tool = _format_tools(tools)
        assert formatted_tool == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "weather",
                        "description": "useful to determine the weather in a given location",
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
                        },
                    }
                },
                {
                    "toolSpec": {
                        "name": "addition",
                        "description": "useful to add two numbers",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                "required": ["a", "b"],
                            }
                        },
                    }
                },
            ]
        }

    def test_format_messages(self):
        messages = [
            ChatMessage.from_system("\\nYou are a helpful assistant, be super brief in your responses."),
            ChatMessage.from_user("What's the capital of France?"),
            ChatMessage.from_assistant("The capital of France is Paris."),
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})]
            ),
            ChatMessage.from_tool(
                tool_result="Sunny and 25°C",
                origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"}),
            ),
            ChatMessage.from_assistant("The weather in Paris is sunny and 25°C."),
        ]
        formatted_system_prompts, formatted_messages = _format_messages(messages)
        assert formatted_system_prompts == [
            {"text": "\\nYou are a helpful assistant, be super brief in your responses."}
        ]
        assert formatted_messages == [
            {"role": "user", "content": [{"text": "What's the capital of France?"}]},
            {"role": "assistant", "content": [{"text": "The capital of France is Paris."}]},
            {"role": "user", "content": [{"text": "What is the weather in Paris?"}]},
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "123", "name": "weather", "input": {"city": "Paris"}}}],
            },
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "123", "content": [{"text": "Sunny and 25°C"}]}}],
            },
            {"role": "assistant", "content": [{"text": "The weather in Paris is sunny and 25°C."}]},
        ]

    def test_extract_replies_from_text_response(self, mock_boto3_session):
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        text_response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "This is a test response"}]}},
            "stopReason": "complete",
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
        }

        replies = _parse_completion_response(text_response, model)
        assert len(replies) == 1
        assert replies[0].text == "This is a test response"
        assert replies[0].role == ChatRole.ASSISTANT
        assert replies[0].meta == {
            "model": model,
            "finish_reason": "complete",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "index": 0,
        }

    def test_extract_replies_from_tool_response(self, mock_boto3_session):
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        tool_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {"key": "value"}}}],
                }
            },
            "stopReason": "tool_call",
            "usage": {"inputTokens": 15, "outputTokens": 25, "totalTokens": 40},
        }

        replies = _parse_completion_response(tool_response, model)
        assert len(replies) == 1
        tool_content = replies[0].tool_call
        assert tool_content.id == "123"
        assert tool_content.tool_name == "test_tool"
        assert tool_content.arguments == {"key": "value"}
        assert replies[0].role == ChatRole.ASSISTANT
        assert replies[0].meta == {
            "model": model,
            "finish_reason": "tool_call",
            "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
            "index": 0,
        }

    def test_extract_replies_from_text_mixed_response(self, mock_boto3_session):
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        mixed_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me help you with that. I'll use the search tool to find the answer."},
                        {"toolUse": {"toolUseId": "456", "name": "search_tool", "input": {"query": "test"}}},
                    ],
                }
            },
            "stopReason": "complete",
            "usage": {"inputTokens": 25, "outputTokens": 35, "totalTokens": 60},
        }

        replies = _parse_completion_response(mixed_response, model)
        assert len(replies) == 1
        assert replies[0].text == "Let me help you with that. I'll use the search tool to find the answer."
        tool_content = replies[0].tool_call
        assert tool_content.id == "456"
        assert tool_content.tool_name == "search_tool"
        assert tool_content.arguments == {"query": "test"}
        assert replies[0].role == ChatRole.ASSISTANT
        assert replies[0].meta == {
            "model": model,
            "finish_reason": "complete",
            "usage": {"prompt_tokens": 25, "completion_tokens": 35, "total_tokens": 60},
            "index": 0,
        }

    def test_process_streaming_response(self, mock_boto3_session):
        """
        Test that process_streaming_response correctly handles streaming events and accumulates responses
        """
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        streaming_chunks = []

        def test_callback(chunk: StreamingChunk):
            streaming_chunks.append(chunk)

        # Simulate a stream of events for both text and tool use
        events = [
            {"contentBlockStart": {"start": {"text": ""}}},
            {"contentBlockDelta": {"delta": {"text": "Let me "}}},
            {"contentBlockDelta": {"delta": {"text": "help you."}}},
            {"contentBlockStop": {}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "search_tool"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"query":'}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '"test"}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "complete"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}}},
        ]

        replies = _parse_streaming_response(events, test_callback, model)

        # Verify streaming chunks were received for text content
        assert len(streaming_chunks) == 2
        assert streaming_chunks[0].content == "Let me "
        assert streaming_chunks[1].content == "help you."

        # Verify final replies
        assert len(replies) == 2
        # Check text reply
        assert replies[0].text == "Let me help you."
        assert replies[0].meta["model"] == model
        assert replies[0].meta["finish_reason"] == "complete"
        assert replies[0].meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        # Check tool use reply
        tool_content = replies[1].tool_call
        assert tool_content.id == "123"
        assert tool_content.tool_name == "search_tool"
        assert tool_content.arguments == {"query": "test"}
