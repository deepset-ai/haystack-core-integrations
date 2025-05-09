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

    def test_formate_messages_multi_tool(self):
        messages = [
            ChatMessage.from_user("What is the weather in Berlin and Paris?"),
            ChatMessage.from_assistant(
                text="To provide you with the weather information for both Berlin and Paris, I'll need to use the "
                "weather tool for each city. I'll make two separate calls to the weather_tool function to get "
                "this information for you.",
                tool_calls=[
                    ToolCall(
                        tool_name="weather_tool", arguments={"location": "Berlin"}, id="tooluse_evFtOFYeSiG_TQ0cAAgy4Q"
                    ),
                    ToolCall(
                        tool_name="weather_tool", arguments={"location": "Paris"}, id="tooluse_Oc0n2we2RvquHwuPEflaQA"
                    ),
                ],
                name=None,
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "index": 0,
                    "finish_reason": "tool_use",
                    "usage": {"prompt_tokens": 366, "completion_tokens": 134, "total_tokens": 500},
                },
            ),
            ChatMessage.from_tool(
                tool_result="Mostly sunny",
                origin=ToolCall(
                    tool_name="weather_tool", arguments={"location": "Berlin"}, id="tooluse_evFtOFYeSiG_TQ0cAAgy4Q"
                ),
            ),
            ChatMessage.from_tool(
                tool_result="Mostly cloudy",
                origin=ToolCall(
                    tool_name="weather_tool", arguments={"location": "Paris"}, id="tooluse_Oc0n2we2RvquHwuPEflaQA"
                ),
            ),
        ]
        formatted_system_prompts, formatted_messages = _format_messages(messages)
        assert formatted_system_prompts == []
        assert formatted_messages == [
            {"role": "user", "content": [{"text": "What is the weather in Berlin and Paris?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "To provide you with the weather information for both Berlin and Paris, I'll need to "
                        "use the weather tool for each city. I'll make two separate calls to the weather_tool "
                        "function to get this information for you."
                    },
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_evFtOFYeSiG_TQ0cAAgy4Q",
                            "name": "weather_tool",
                            "input": {"location": "Berlin"},
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_Oc0n2we2RvquHwuPEflaQA",
                            "name": "weather_tool",
                            "input": {"location": "Paris"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_evFtOFYeSiG_TQ0cAAgy4Q",
                            "content": [{"text": "Mostly sunny"}],
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_Oc0n2we2RvquHwuPEflaQA",
                            "content": [{"text": "Mostly cloudy"}],
                        }
                    },
                ],
            },
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

    def test_extract_replies_from_multi_tool_response(self, mock_boto3_session):
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        response_body = {
            "ResponseMetadata": {
                "RequestId": "0ba58797-2194-4779-9a53-597c24ce337a",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": "Tue, 06 May 2025 20:47:24 GMT",
                    "content-type": "application/json",
                    "content-length": "616",
                    "connection": "keep-alive",
                    "x-amzn-requestid": "0ba58797-2194-4779-9a53-597c24ce337a",
                },
                "RetryAttempts": 0,
            },
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "To provide you with the weather information for both Berlin and Paris, I'll need "
                            "to use the weather tool for each city. I'll make two separate calls to the "
                            "weather_tool function to get this information for you."
                        },
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_evFtOFYeSiG_TQ0cAAgy4Q",
                                "name": "weather_tool",
                                "input": {"location": "Berlin"},
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_Oc0n2we2RvquHwuPEflaQA",
                                "name": "weather_tool",
                                "input": {"location": "Paris"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 366, "outputTokens": 134, "totalTokens": 500},
            "metrics": {"latencyMs": 3726},
        }
        replies = _parse_completion_response(response_body, model)

        expected_message = ChatMessage.from_assistant(
            text="To provide you with the weather information for both Berlin and Paris, I'll need to use the weather "
            "tool for each city. I'll make two separate calls to the weather_tool function to get this "
            "information for you.",
            tool_calls=[
                ToolCall(
                    tool_name="weather_tool", arguments={"location": "Berlin"}, id="tooluse_evFtOFYeSiG_TQ0cAAgy4Q"
                ),
                ToolCall(
                    tool_name="weather_tool", arguments={"location": "Paris"}, id="tooluse_Oc0n2we2RvquHwuPEflaQA"
                ),
            ],
            name=None,
            meta={
                "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "index": 0,
                "finish_reason": "tool_use",
                "usage": {"prompt_tokens": 366, "completion_tokens": 134, "total_tokens": 500},
            },
        )
        assert replies[0] == expected_message

    def test_process_streaming_response_one_tool_call(self, mock_boto3_session):
        """
        Test that process_streaming_response correctly handles streaming events and accumulates responses
        """
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        streaming_chunks = []

        def test_callback(chunk: StreamingChunk):
            streaming_chunks.append(chunk)

        # Simulate a stream of events for both text and tool use
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Certainly! I can"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " help you find out"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " the weather"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " in Berlin. To"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " get this information, I'll"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " use the weather tool available"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " to me."}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " Let me fetch"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " that data for"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " you."}, "contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "tooluse_pLGRAmK7TNKoZQ_rntVN_Q", "name": "weather_tool"}},
                    "contentBlockIndex": 1,
                }
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": ""}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"'}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": 'location": '}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '"B'}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": 'erlin"}'}}, "contentBlockIndex": 1}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 364, "outputTokens": 71, "totalTokens": 435},
                    "metrics": {"latencyMs": 2449},
                }
            },
        ]

        replies = _parse_streaming_response(events, test_callback, model)
        # Pop completion_start_time since it will always change
        replies[0].meta.pop("completion_start_time")
        expected_messages = [
            ChatMessage.from_assistant(
                text="Certainly! I can help you find out the weather in Berlin. To get this information, I'll use the "
                "weather tool available to me. Let me fetch that data for you.",
                name=None,
                tool_calls=[
                    ToolCall(
                        tool_name="weather_tool", arguments={"location": "Berlin"}, id="tooluse_pLGRAmK7TNKoZQ_rntVN_Q"
                    )
                ],
                meta={
                    "model": model,
                    "index": 0,
                    "finish_reason": "tool_use",
                    "usage": {"prompt_tokens": 364, "completion_tokens": 71, "total_tokens": 435},
                },
            )
        ]

        # Verify streaming chunks were received for all content
        assert len(streaming_chunks) == 21
        assert streaming_chunks[1].content == "Certainly! I can"
        assert streaming_chunks[2].content == " help you find out"
        assert streaming_chunks[12].meta["tool_calls"] == [
            {
                "index": 1,
                "id": "tooluse_pLGRAmK7TNKoZQ_rntVN_Q",
                "function": {"arguments": "", "name": "weather_tool"},
                "type": "function",
            }
        ]

        # Verify final replies
        assert len(replies) == 1
        assert replies == expected_messages

    def test_parse_streaming_response_with_two_tool_calls(self, mock_boto3_session):
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        streaming_chunks = []

        def test_callback(chunk: StreamingChunk):
            streaming_chunks.append(chunk)

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "To"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " answer your question about the"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " weather in Berlin and Paris, I'll"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " need to use the weather_tool"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " for each city. Let"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " me fetch that information for"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": " you."}, "contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "tooluse_A0jTtaiQTFmqD_cIq8I1BA", "name": "weather_tool"}},
                    "contentBlockIndex": 1,
                }
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": ""}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"location":'}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": ' "Be'}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": 'rlin"}'}}, "contentBlockIndex": 1}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "tooluse_LTc2TUMgTRiobK5Z5CCNSw", "name": "weather_tool"}},
                    "contentBlockIndex": 2,
                }
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": ""}}, "contentBlockIndex": 2}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"l'}}, "contentBlockIndex": 2}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "ocati"}}, "contentBlockIndex": 2}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": 'on": "P'}}, "contentBlockIndex": 2}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "ari"}}, "contentBlockIndex": 2}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": 's"}'}}, "contentBlockIndex": 2}},
            {"contentBlockStop": {"contentBlockIndex": 2}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 366, "outputTokens": 83, "totalTokens": 449},
                    "metrics": {"latencyMs": 3194},
                }
            },
        ]

        replies = _parse_streaming_response(events, test_callback, model)
        # Pop completion_start_time since it will always change
        replies[0].meta.pop("completion_start_time")
        expected_messages = [
            ChatMessage.from_assistant(
                text="To answer your question about the weather in Berlin and Paris, I'll need to use the "
                "weather_tool for each city. Let me fetch that information for you.",
                name=None,
                tool_calls=[
                    ToolCall(
                        tool_name="weather_tool", arguments={"location": "Berlin"}, id="tooluse_A0jTtaiQTFmqD_cIq8I1BA"
                    ),
                    ToolCall(
                        tool_name="weather_tool", arguments={"location": "Paris"}, id="tooluse_LTc2TUMgTRiobK5Z5CCNSw"
                    ),
                ],
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "index": 0,
                    "finish_reason": "tool_use",
                    "usage": {"prompt_tokens": 366, "completion_tokens": 83, "total_tokens": 449},
                },
            ),
        ]
        assert replies == expected_messages
