import base64
from unittest.mock import ANY

import pytest
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    ImageContent,
    ReasoningContent,
    StreamingChunk,
    ToolCall,
    ToolCallDelta,
)
from haystack.tools import Tool

from haystack_integrations.components.generators.amazon_bedrock.chat.utils import (
    _convert_streaming_chunks_to_chat_message,
    _format_messages,
    _format_text_image_message,
    _format_tools,
    _parse_completion_response,
    _parse_streaming_response,
    _validate_guardrail_config,
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

    def test_format_message_thinking(self):
        assistant_message = ChatMessage.from_assistant(
            "This is a test message.",
            reasoning=ReasoningContent(
                reasoning_text="This is the reasoning behind the message.",
                extra={
                    "reasoning_contents": [
                        {
                            "reasoning_content": {
                                "reasoning_text": {
                                    "text": "This is the reasoning behind the message.",
                                    "signature": "reasoning_signature",
                                }
                            }
                        }
                    ]
                },
            ),
        )
        formatted_message = _format_messages([assistant_message])[1][0]
        assert formatted_message == {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "This is the reasoning behind the message.",
                            "signature": "reasoning_signature",
                        }
                    }
                },
                {"text": "This is a test message."},
            ],
        }

        tool_call_message = ChatMessage.from_assistant(
            "This is a test message with a tool call.",
            tool_calls=[ToolCall(id="123", tool_name="test_tool", arguments={"key": "value"})],
            reasoning=ReasoningContent(
                reasoning_text="This is the reasoning behind the tool call.",
                extra={
                    "reasoning_contents": [
                        {
                            "reasoning_content": {
                                "reasoning_text": {
                                    "text": "This is the reasoning behind the tool call.",
                                    "signature": "reasoning_signature",
                                }
                            }
                        }
                    ]
                },
            ),
        )
        formatted_message = _format_messages([tool_call_message])[1][0]
        assert formatted_message == {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "This is the reasoning behind the tool call.",
                            "signature": "reasoning_signature",
                        }
                    }
                },
                {"text": "This is a test message with a tool call."},
                {"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {"key": "value"}}},
            ],
        }

        tool_call_message_with_redacted = ChatMessage.from_assistant(
            "This is a test message with a tool call.",
            tool_calls=[ToolCall(id="123", tool_name="test_tool", arguments={"key": "value"})],
            reasoning=ReasoningContent(
                reasoning_text="[REDACTED]",
                extra={
                    "reasoning_contents": [{"reasoning_content": {"redacted_content": b"Some encrypted byte string"}}]
                },
            ),
        )
        formatted_message = _format_messages([tool_call_message_with_redacted])[1][0]
        assert formatted_message == {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"redactedContent": b"Some encrypted byte string"}},
                {"text": "This is a test message with a tool call."},
                {"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {"key": "value"}}},
            ],
        }

        tool_call_message_with_redacted_and_normal_thinking = ChatMessage.from_assistant(
            "This is a test message with a tool call.",
            tool_calls=[ToolCall(id="123", tool_name="test_tool", arguments={"key": "value"})],
            reasoning=ReasoningContent(
                reasoning_text="[REDACTED]This is the reasoning behind the tool call.",
                extra={
                    "reasoning_contents": [
                        {"reasoning_content": {"redacted_content": b"Some encrypted byte string"}},
                        {
                            "reasoning_content": {
                                "reasoningText": {
                                    "text": "This is the reasoning behind the tool call.",
                                    "signature": "reasoning_signature",
                                }
                            }
                        },
                    ]
                },
            ),
        )
        formatted_message = _format_messages([tool_call_message_with_redacted_and_normal_thinking])[1][0]
        assert formatted_message == {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"redactedContent": b"Some encrypted byte string"}},
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "This is the reasoning behind the tool call.",
                            "signature": "reasoning_signature",
                        }
                    }
                },
                {"text": "This is a test message with a tool call."},
                {"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {"key": "value"}}},
            ],
        }

    def test_format_text_image_message(self):
        plain_assistant_message = ChatMessage.from_assistant("This is a test message.")
        formatted_message = _format_text_image_message(plain_assistant_message)
        assert formatted_message == {"role": "assistant", "content": [{"text": "This is a test message."}]}

        plain_user_message = ChatMessage.from_user("This is a test message.")
        formatted_message = _format_text_image_message(plain_user_message)
        assert formatted_message == {"role": "user", "content": [{"text": "This is a test message."}]}

        base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        image_content = ImageContent(base64_image)
        image_message = ChatMessage.from_user(content_parts=["This is a test message.", image_content])
        formatted_message = _format_text_image_message(image_message)
        assert formatted_message == {
            "role": "user",
            "content": [
                {"text": "This is a test message."},
                {"image": {"format": "png", "source": {"bytes": base64.b64decode(base64_image)}}},
            ],
        }

    def test_format_text_image_message_errors(self):
        base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        image_content = ImageContent(base64_image)
        assistant_message_with_image = ChatMessage.from_user(content_parts=["This is a test message.", image_content])
        assistant_message_with_image._role = ChatRole.ASSISTANT
        with pytest.raises(ValueError):
            _format_text_image_message(assistant_message_with_image)

        image_content_unsupported_format = ImageContent(base64_image, mime_type="image/tiff")
        image_message = ChatMessage.from_user(
            content_parts=["This is a test message.", image_content_unsupported_format]
        )
        with pytest.raises(ValueError):
            _format_text_image_message(image_message)

    def test_format_messages_multi_tool(self):
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
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
        }

        replies = _parse_completion_response(text_response, model)
        assert len(replies) == 1
        assert replies[0].text == "This is a test response"
        assert replies[0].role == ChatRole.ASSISTANT
        assert replies[0].meta == {
            "model": model,
            "finish_reason": "stop",
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
            "stopReason": "tool_use",
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
            "finish_reason": "tool_calls",
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
            "stopReason": "end_turn",
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
            "finish_reason": "stop",
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
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 366, "completion_tokens": 134, "total_tokens": 500},
            },
        )
        assert replies[0] == expected_message

    def test_extract_replies_from_one_tool_response_with_thinking(self, mock_boto3_session):
        model = "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        response_body = {
            "ResponseMetadata": {
                "RequestId": "d7be81a1-5d37-40fe-936a-7c96e850cdda",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": "Tue, 15 Jul 2025 12:49:56 GMT",
                    "content-type": "application/json",
                    "content-length": "1107",
                    "connection": "keep-alive",
                    "x-amzn-requestid": "d7be81a1-5d37-40fe-936a-7c96e850cdda",
                },
                "RetryAttempts": 0,
            },
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": "The user wants to know the weather in Paris. I have a `weather` function "
                                    "available that can provide this information. \n\nRequired parameters for "
                                    "the weather function:\n- city: The city to get the weather for\n\nIn this "
                                    'case, the user has clearly specified "Paris" as the city, so I have all '
                                    "the required information to make the function call.",
                                    "signature": "...",
                                }
                            }
                        },
                        {"text": "I'll check the current weather in Paris for you."},
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_iUqy8-ypSByLK5zFkka8uA",
                                "name": "weather",
                                "input": {"city": "Paris"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 412,
                "outputTokens": 146,
                "totalTokens": 558,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
            },
            "metrics": {"latencyMs": 4811},
        }
        replies = _parse_completion_response(response_body, model)

        expected_message = ChatMessage.from_assistant(
            text="I'll check the current weather in Paris for you.",
            tool_calls=[
                ToolCall(tool_name="weather", arguments={"city": "Paris"}, id="tooluse_iUqy8-ypSByLK5zFkka8uA")
            ],
            reasoning=ReasoningContent(
                reasoning_text="The user wants to know the weather in Paris. I have a `weather` function available "
                "that can provide this information. \n\nRequired parameters for the weather function:\n- city: The "
                'city to get the weather for\n\nIn this case, the user has clearly specified "Paris" as the city, so '
                "I have all the required information to make the function call.",
                extra={
                    "reasoning_contents": [
                        {
                            "reasoning_content": {
                                "reasoning_text": {
                                    "text": "The user wants to know the weather in Paris. I have a `weather` function "
                                    "available that can provide this information. \n\nRequired parameters for "
                                    "the weather function:\n- city: The city to get the weather for\n\nIn this "
                                    'case, the user has clearly specified "Paris" as the city, so I have all '
                                    "the required information to make the function call.",
                                    "signature": "...",
                                }
                            }
                        }
                    ]
                },
            ),
            meta={
                "model": "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "index": 0,
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 412, "completion_tokens": 146, "total_tokens": 558},
            },
        )
        assert replies[0] == expected_message

    def test_extract_replies_with_guardrail(self, mock_boto3_session):
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"

        trace = {
            "guardrail": {
                "inputAssessment": {
                    "test_guardrail_id": {
                        "topicPolicy": {
                            "topics": [
                                {"name": "Investments topic", "type": "DENY", "action": "BLOCKED", "detected": True}
                            ]
                        },
                        "invocationMetrics": {
                            "guardrailProcessingLatency": 273,
                            "usage": {
                                "topicPolicyUnits": 1,
                                "contentPolicyUnits": 0,
                                "wordPolicyUnits": 0,
                                "sensitiveInformationPolicyUnits": 0,
                                "sensitiveInformationPolicyFreeUnits": 0,
                                "contextualGroundingPolicyUnits": 0,
                                "contentPolicyImageUnits": 0,
                            },
                            "guardrailCoverage": {"textCharacters": {"guarded": 48, "total": 48}},
                        },
                    }
                },
                "actionReason": "Guardrail blocked.",
            }
        }

        response_body = {
            "ResponseMetadata": {
                "RequestId": "7f2b43ef-fb52-40e4-ab14-8cc1edaf5013",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": "Thu, 18 Sep 2025 09:14:48 GMT",
                    "content-type": "application/json",
                    "content-length": "835",
                    "connection": "keep-alive",
                    "x-amzn-requestid": "7f2b43ef-fb52-40e4-ab14-8cc1edaf5013",
                },
                "RetryAttempts": 0,
            },
            "output": {
                "message": {"role": "assistant", "content": [{"text": "Sorry, the model cannot answer this question."}]}
            },
            "stopReason": "guardrail_intervened",
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 316},
            "trace": trace,
        }

        replies = _parse_completion_response(response_body, model)
        assert len(replies) == 1
        assert replies[0].text == "Sorry, the model cannot answer this question."
        assert replies[0].role == ChatRole.ASSISTANT
        assert replies[0].meta == {
            "model": model,
            "finish_reason": "content_filter",
            "index": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "trace": trace,
        }

    def test_process_streaming_response_one_tool_call(self, mock_boto3_session):
        """
        Test that process_streaming_response correctly handles streaming events and accumulates responses
        """
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        type_ = (
            "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
        )
        base_meta = {"model": model, "received_at": ANY}
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

        c_info = ComponentInfo(type=type_)

        replies = _parse_streaming_response(events, test_callback, model, c_info)
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
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 364, "completion_tokens": 71, "total_tokens": 435},
                },
            )
        ]

        expected_chunks = [
            StreamingChunk(content="", meta=base_meta, component_info=c_info),
            StreamingChunk(content="Certainly! I can", meta=base_meta, component_info=c_info, index=0, start=True),
            StreamingChunk(content=" help you find out", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content=" the weather", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content=" in Berlin. To", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content=" get this information, I'll", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content=" use the weather tool available", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content=" to me.", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content=" Let me fetch", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content=" that data for", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content=" you.", meta=base_meta, component_info=c_info, index=0),
            StreamingChunk(content="", meta=base_meta, component_info=c_info),
            StreamingChunk(
                content="",
                meta=base_meta,
                component_info=c_info,
                index=1,
                tool_calls=[ToolCallDelta(index=1, tool_name="weather_tool", id="tooluse_pLGRAmK7TNKoZQ_rntVN_Q")],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta=base_meta,
                component_info=c_info,
                index=1,
                tool_calls=[ToolCallDelta(index=1, arguments="")],
            ),
            StreamingChunk(
                content="",
                meta=base_meta,
                component_info=c_info,
                index=1,
                tool_calls=[ToolCallDelta(index=1, arguments='{"')],
            ),
            StreamingChunk(
                content="",
                meta=base_meta,
                component_info=c_info,
                index=1,
                tool_calls=[ToolCallDelta(index=1, arguments='location": ')],
            ),
            StreamingChunk(
                content="",
                meta=base_meta,
                component_info=c_info,
                index=1,
                tool_calls=[ToolCallDelta(index=1, arguments='"B')],
            ),
            StreamingChunk(
                content="",
                meta=base_meta,
                component_info=c_info,
                index=1,
                tool_calls=[ToolCallDelta(index=1, arguments='erlin"}')],
            ),
            StreamingChunk(content="", meta=base_meta, component_info=c_info),
            StreamingChunk(content="", meta=base_meta, component_info=c_info, finish_reason="tool_calls"),
            StreamingChunk(
                content="",
                meta={
                    "model": model,
                    "received_at": ANY,
                    "usage": {"prompt_tokens": 364, "completion_tokens": 71, "total_tokens": 435},
                },
                component_info=c_info,
            ),
        ]
        # Verify streaming chunks were received for all content
        assert len(streaming_chunks) == 21
        for idx, chunk in enumerate(streaming_chunks):
            assert chunk == expected_chunks[idx]

        # Verify final replies
        assert len(replies) == 1
        assert replies == expected_messages

    def test_process_streaming_response_one_tool_call_with_thinking(self, mock_boto3_session):
        model = "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
        type_ = (
            "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
        )
        streaming_chunks = []

        def test_callback(chunk: StreamingChunk):
            streaming_chunks.append(chunk)

        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "The user is asking about the weather"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " in Paris. I have"}},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": " access to a"}}, "contentBlockIndex": 0}},
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " weather function that takes"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " a city parameter. Paris"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " is clearly specifie"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "d as the city, so I have all"}},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": " the required parameters to make the"}},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": " function call."}}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "..."}}, "contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "tooluse_1gPhO4A1RNWgzKbt1PXWLg", "name": "weather"}},
                    "contentBlockIndex": 1,
                }
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": ""}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"ci'}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "ty"}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '": "P'}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": "aris"}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '"}'}}, "contentBlockIndex": 1}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 412, "outputTokens": 104, "totalTokens": 516},
                    "metrics": {"latencyMs": 2134},
                }
            },
        ]

        replies = _parse_streaming_response(events, test_callback, model, ComponentInfo(type=type_))

        expected_messages = [
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(tool_name="weather", arguments={"city": "Paris"}, id="tooluse_1gPhO4A1RNWgzKbt1PXWLg")
                ],
                reasoning=ReasoningContent(
                    reasoning_text="The user is asking about the weather in Paris. I have access to a weather function "
                    "that takes a city parameter. Paris is clearly specified as the city, so I have all the required "
                    "parameters to make the function call.",
                    extra={
                        "reasoning_contents": [
                            {
                                "reasoning_content": {
                                    "reasoning_text": {
                                        "text": "The user is asking about the weather in Paris. I have access to a "
                                        "weather function that takes a city parameter. Paris is clearly specified "
                                        "as the city, so I have all the required parameters to make the function call.",
                                        "signature": "...",
                                    }
                                }
                            }
                        ]
                    },
                ),
                meta={
                    "model": "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 412, "completion_tokens": 104, "total_tokens": 516},
                    "completion_start_time": ANY,
                },
            )
        ]
        assert replies == expected_messages

    def test_process_streaming_response_with_one_tool_call_with_redacted_thinking(self, mock_boto3_session):
        model = "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-sonnet-3-7-20250219-v1:0"
        type_ = (
            "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
        )
        streaming_chunks = []

        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"redactedContent": b"Some encrypted byte string"}},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {
                "contentBlockDelta": {
                    "delta": {"text": "I notice your message contains a special"},
                    "contentBlockIndex": 1,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": " string that doesn't include an actual question"},
                    "contentBlockIndex": 1,
                }
            },
            {"contentBlockDelta": {"delta": {"text": " or request. \n\nIs there something"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " specific you'd like to know"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": "? If you're intereste"}, "contentBlockIndex": 1}},
            {
                "contentBlockDelta": {
                    "delta": {"text": "d in weather information for a particular location"},
                    "contentBlockIndex": 1,
                }
            },
            {"contentBlockDelta": {"delta": {"text": ", I can help"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " you with that using the weather"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " tool. Just let me know which"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " city you'd like"}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"text": " to check the weather for."}, "contentBlockIndex": 1}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 461, "outputTokens": 138, "totalTokens": 599},
                    "metrics": {"latencyMs": 8493},
                }
            },
        ]

        def test_callback(chunk: StreamingChunk):
            streaming_chunks.append(chunk)

        replies = _parse_streaming_response(events, test_callback, model, ComponentInfo(type=type_))

        expected_messages = [
            ChatMessage.from_assistant(
                text="I notice your message contains a special string that doesn't include an actual question or "
                "request. \n\nIs there something specific you'd like to know? If you're interested in weather "
                "information for a particular location, I can help you with that using the weather tool. "
                "Just let me know which city you'd like to check the weather for.",
                reasoning=ReasoningContent(
                    reasoning_text="[REDACTED]",
                    extra={
                        "reasoning_contents": [
                            {
                                "reasoning_content": {
                                    "redacted_content": b"Some encrypted byte string",
                                }
                            }
                        ]
                    },
                ),
                meta={
                    "model": model,
                    "index": 0,
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 461, "completion_tokens": 138, "total_tokens": 599},
                    "completion_start_time": ANY,
                },
            )
        ]
        assert replies == expected_messages

    def test_parse_streaming_response_with_two_tool_calls(self, mock_boto3_session):
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        type_ = (
            "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
        )
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

        replies = _parse_streaming_response(events, test_callback, model, ComponentInfo(type=type_))
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
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 366, "completion_tokens": 83, "total_tokens": 449},
                    "completion_start_time": ANY,
                },
            )
        ]
        assert replies == expected_messages

    def test_parse_streaming_response_with_guardrail(self, mock_boto3_session):
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        type_ = (
            "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
        )
        streaming_chunks = []

        trace = {
            "guardrail": {
                "inputAssessment": {
                    "vodp82dpe5xv": {
                        "test_guardrail_id": {
                            "topicPolicy": {
                                "topics": [
                                    {"name": "Investments topic", "type": "DENY", "action": "BLOCKED", "detected": True}
                                ]
                            },
                            "invocationMetrics": {
                                "guardrailProcessingLatency": 299,
                                "usage": {
                                    "topicPolicyUnits": 1,
                                    "contentPolicyUnits": 0,
                                    "wordPolicyUnits": 0,
                                    "sensitiveInformationPolicyUnits": 0,
                                    "sensitiveInformationPolicyFreeUnits": 0,
                                    "contextualGroundingPolicyUnits": 0,
                                    "contentPolicyImageUnits": 0,
                                },
                                "guardrailCoverage": {"textCharacters": {"guarded": 48, "total": 48}},
                            },
                        }
                    },
                    "actionReason": "Guardrail blocked.",
                }
            }
        }

        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "delta": {"text": "Sorry, the model cannot answer this question."},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "guardrail_intervened"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                    "metrics": {"latencyMs": 334},
                    "trace": trace,
                }
            },
        ]

        def test_callback(chunk: StreamingChunk):
            streaming_chunks.append(chunk)

        replies = _parse_streaming_response(events, test_callback, model, ComponentInfo(type=type_))

        expected_messages = [
            ChatMessage.from_assistant(
                text="Sorry, the model cannot answer this question.",
                meta={
                    "completion_start_time": ANY,
                    "model": model,
                    "index": 0,
                    "finish_reason": "content_filter",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "trace": trace,
                },
            )
        ]
        assert replies == expected_messages

    def test_convert_streaming_chunks_to_chat_message_tool_call_with_empty_arguments(self):
        chunks = [
            StreamingChunk(
                content="Certainly! I",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.072764+00:00",
                },
                index=0,
                start=True,
            ),
            StreamingChunk(
                content=" can help",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.111264+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content=" you print",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.162575+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content=' "Hello World"',
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.215535+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content=" using the available",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.270642+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content=' "',
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.349415+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content='hello_world" tool',
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.426891+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content=". This tool is",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.495910+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content=' designed to print "',
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.527426+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content='Hello World" an',
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.590629+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content="d doesn't require any parameters",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.682261+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content=". Let",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.790526+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content="'s go ahead an",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.845332+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content="d use",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.990588+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content=" it.",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:07.994309+00:00",
                },
                index=0,
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:08.359127+00:00",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:08.359912+00:00",
                },
                index=1,
                tool_calls=[
                    ToolCallDelta(index=1, id="tooluse_QZlUqTveTwyUaCQGQbWP6g", tool_name="hello_world", arguments="")
                ],
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:08.361612+00:00",
                },
                index=1,
                tool_calls=[ToolCallDelta(index=1, arguments="")],
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:08.592175+00:00",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:08.592175+00:00",
                },
                finish_reason="tool_calls",
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "received_at": "2025-07-31T08:46:08.596700+00:00",
                    "usage": {"prompt_tokens": 349, "completion_tokens": 84, "total_tokens": 433},
                },
            ),
        ]

        message = _convert_streaming_chunks_to_chat_message(chunks)

        # Verify the message content
        assert message.text == (
            'Certainly! I can help you print "Hello World" using the available "hello_world" tool. This tool is '
            "designed to print \"Hello World\" and doesn't require any parameters. Let's go ahead and use it."
        )

        # Verify tool calls
        assert len(message.tool_calls) == 1
        tool_call = message.tool_calls[0]
        assert tool_call.id == "tooluse_QZlUqTveTwyUaCQGQbWP6g"
        assert tool_call.tool_name == "hello_world"
        assert tool_call.arguments == {}

        # Verify meta information
        assert message._meta["model"] == "anthropic.claude-3-5-sonnet-20240620-v1:0"
        assert message._meta["index"] == 0
        assert message._meta["finish_reason"] == "tool_calls"
        assert message._meta["usage"] == {"completion_tokens": 84, "prompt_tokens": 349, "total_tokens": 433}

    def test_validate_guardrail_config_with_valid_configs(self):
        _validate_guardrail_config(guardrail_config=None, streaming=False)
        _validate_guardrail_config(
            guardrail_config={"guardrailIdentifier": "test", "guardrailVersion": "test"}, streaming=False
        )
        _validate_guardrail_config(
            guardrail_config={"guardrailIdentifier": "test", "guardrailVersion": "test"}, streaming=True
        )
        _validate_guardrail_config(
            guardrail_config={
                "guardrailIdentifier": "test",
                "guardrailVersion": "test",
                "streamProcessingMode": "enabled",
            },
            streaming=True,
        )

    def test_validate_guardrail_config_with_invalid_configs(self):
        with pytest.raises(ValueError, match="`guardrailIdentifier` and `guardrailVersion` fields are required"):
            _validate_guardrail_config(guardrail_config={"guardrailIdentifier": "test"}, streaming=False)
        with pytest.raises(ValueError, match="`guardrailIdentifier` and `guardrailVersion` fields are required"):
            _validate_guardrail_config(guardrail_config={"guardrailVersion": "test"}, streaming=False)
        with pytest.raises(ValueError, match="`streamProcessingMode` field is only supported for streaming"):
            _validate_guardrail_config(
                guardrail_config={
                    "guardrailIdentifier": "test",
                    "guardrailVersion": "test",
                    "streamProcessingMode": "test",
                },
                streaming=False,
            )
