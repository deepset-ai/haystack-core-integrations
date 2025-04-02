from typing import Any, Dict, Optional

import pytest
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.tools import Tool

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator
from haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator import (
    _parse_completion_response,
    _parse_streaming_response,
)

KLASS = "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
MODELS_TO_TEST = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "cohere.command-r-plus-v1:0",
    "mistral.mistral-large-2402-v1:0",
]
MODELS_TO_TEST_WITH_TOOLS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    # "cohere.command-r-plus-v1:0",
    # "mistral.mistral-large-2402-v1:0",
]

# so far we've discovered these models support streaming and tool use
STREAMING_TOOL_MODELS = ["anthropic.claude-3-5-sonnet-20240620-v1:0", "cohere.command-r-plus-v1:0"]


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


@pytest.fixture
def chat_messages():
    messages = [
        ChatMessage.from_system("\\nYou are a helpful assistant, be super brief in your responses."),
        ChatMessage.from_user("What's the capital of France?"),
    ]
    return messages


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=weather,
    )
    return [tool]


@pytest.mark.parametrize(
    "boto3_config",
    [
        None,
        {
            "read_timeout": 1000,
        },
    ],
)
def test_to_dict(mock_boto3_session, boto3_config):
    """
    Test that the to_dict method returns the correct dictionary without aws credentials
    """
    generator = AmazonBedrockChatGenerator(
        model="cohere.command-r-plus-v1:0",
        generation_kwargs={"temperature": 0.7},
        streaming_callback=print_streaming_chunk,
        boto3_config=boto3_config,
    )
    expected_dict = {
        "type": KLASS,
        "init_parameters": {
            "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
            "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
            "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
            "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
            "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
            "model": "cohere.command-r-plus-v1:0",
            "generation_kwargs": {"temperature": 0.7},
            "stop_words": [],
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "boto3_config": boto3_config,
            "tools": None,
        },
    }

    assert generator.to_dict() == expected_dict


@pytest.mark.parametrize(
    "boto3_config",
    [
        None,
        {
            "read_timeout": 1000,
        },
    ],
)
def test_from_dict(mock_boto3_session: Any, boto3_config: Optional[Dict[str, Any]]):
    """
    Test that the from_dict method returns the correct object
    """
    generator = AmazonBedrockChatGenerator.from_dict(
        {
            "type": KLASS,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "generation_kwargs": {"temperature": 0.7},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "boto3_config": boto3_config,
                "tools": None,
            },
        }
    )
    assert generator.model == "anthropic.claude-3-5-sonnet-20240620-v1:0"
    assert generator.streaming_callback == print_streaming_chunk
    assert generator.boto3_config == boto3_config


def test_default_constructor(mock_boto3_session, mock_aioboto3_session, set_env_variables):
    """
    Test that the default constructor sets the correct values
    """

    layer = AmazonBedrockChatGenerator(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    assert layer.model == "anthropic.claude-3-5-sonnet-20240620-v1:0"

    # assert mocked boto3 client called exactly once
    mock_boto3_session.assert_called_once()

    # assert mocked boto3 client was called with the correct parameters
    mock_boto3_session.assert_called_with(
        aws_access_key_id="some_fake_id",
        aws_secret_access_key="some_fake_key",
        aws_session_token="some_fake_token",
        region_name="fake_region",
        profile_name="some_fake_profile",
    )


def test_constructor_with_generation_kwargs(mock_boto3_session):
    """
    Test that model_kwargs are correctly set in the constructor
    """
    generation_kwargs = {"temperature": 0.7}

    layer = AmazonBedrockChatGenerator(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0", generation_kwargs=generation_kwargs
    )
    assert layer.generation_kwargs == generation_kwargs


def test_constructor_with_empty_model():
    """
    Test that the constructor raises an error when the model is empty
    """
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        AmazonBedrockChatGenerator(model="")


class TestAmazonBedrockChatGeneratorInference:

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    @pytest.mark.integration
    def test_default_inference_params(self, model_name, chat_messages):
        client = AmazonBedrockChatGenerator(model=model_name)
        response = client.run(chat_messages)

        assert "replies" in response, "Response does not contain 'replies' key"
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert "paris" in first_reply.text.lower(), "First reply does not contain 'paris'"
        assert first_reply.meta, "First reply has no metadata"

        if first_reply.meta and "usage" in first_reply.meta:
            assert "prompt_tokens" in first_reply.meta["usage"]
            assert "completion_tokens" in first_reply.meta["usage"]

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    @pytest.mark.integration
    def test_default_inference_with_streaming(self, model_name, chat_messages):
        streaming_callback_called = False
        paris_found_in_response = False

        def streaming_callback(chunk: StreamingChunk):
            nonlocal streaming_callback_called, paris_found_in_response
            streaming_callback_called = True
            assert isinstance(chunk, StreamingChunk)
            assert chunk.content is not None
            if not paris_found_in_response:
                paris_found_in_response = "paris" in chunk.content.lower()

        client = AmazonBedrockChatGenerator(model=model_name, streaming_callback=streaming_callback)
        response = client.run(chat_messages)

        assert streaming_callback_called, "Streaming callback was not called"
        assert paris_found_in_response, "The streaming callback response did not contain 'paris'"
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert "paris" in first_reply.text.lower(), "First reply does not contain 'paris'"
        assert first_reply.meta, "First reply has no metadata"

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_TOOLS)
    @pytest.mark.integration
    def test_tools_use(self, model_name):
        """
        Test tools use with passing the generation_kwargs={"toolConfig": tool_config}
        and not the tools parameter. We support this because some users might want to use the toolConfig
        parameter to pass the tool configuration to the model.
        """
        # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolConfiguration.html
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "top_song",
                        "description": "Get the most popular song played on a radio station.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station "
                                        "for which you want the most popular song. "
                                        "Example calls signs are WZPZ and WKRP.",
                                    }
                                },
                                "required": ["sign"],
                            }
                        },
                    }
                }
            ],
            # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            "toolChoice": {"auto": {}},
        }

        messages = [ChatMessage.from_user("What is the most popular song on WZPZ?")]
        client = AmazonBedrockChatGenerator(model=model_name)
        response = client.run(messages=messages, generation_kwargs={"toolConfig": tool_config})
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        # Find the message with tool calls as in some models it is the first message, in some second
        tool_message = None
        for message in replies:
            if message.tool_call:  # Using tool_call instead of tool_calls to match existing code
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "top_song", f"{tool_call} does not contain the correct 'tool_name' value"
        assert tool_call.arguments, f"Tool call {tool_call} does not contain 'arguments' value"
        assert (
            tool_call.arguments["sign"] == "WZPZ"
        ), f"Tool call {tool_call} does not contain the correct 'arguments' value"

    @pytest.mark.parametrize("model_name", STREAMING_TOOL_MODELS)
    @pytest.mark.integration
    def test_tools_use_with_streaming(self, model_name):
        """
        Test tools use with streaming but with passing the generation_kwargs={"toolConfig": tool_config}
        and not the tools parameter. We support this because some users might want to use the toolConfig
        parameter to pass the tool configuration to the model.
        """
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "top_song",
                        "description": "Get the most popular song played on a radio station.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station "
                                        "for which you want the most popular song. Example "
                                        "calls signs are WZPZ and WKRP.",
                                    }
                                },
                                "required": ["sign"],
                            }
                        },
                    }
                }
            ],
            "toolChoice": {"auto": {}},
        }

        messages = [ChatMessage.from_user("What is the most popular song on WZPZ?")]
        client = AmazonBedrockChatGenerator(model=model_name, streaming_callback=print_streaming_chunk)
        response = client.run(messages=messages, generation_kwargs={"toolConfig": tool_config})
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert first_reply.meta, "First reply has no metadata"

        # Find the message containing the tool call
        tool_message = None
        for message in replies:
            if message.tool_call:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "top_song", f"{tool_call} does not contain the correct 'tool_name' value"
        assert tool_call.arguments, f"{tool_call} does not contain 'arguments' value"
        assert tool_call.arguments["sign"] == "WZPZ", f"{tool_call} does not contain the correct 'input' value"

    def test_extract_replies_from_response(self, mock_boto3_session):
        """
        Test that extract_replies_from_response correctly processes both text and tool use responses
        """
        model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        # Test case 1: Simple text response
        text_response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "This is a test response"}]}},
            "stopReason": "complete",
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
        }

        replies = _parse_completion_response(text_response, model)
        assert len(replies) == 1
        assert replies[0].text == "This is a test response"
        assert replies[0].role == ChatRole.ASSISTANT
        assert replies[0].meta["model"] == model
        assert replies[0].meta["finish_reason"] == "complete"
        assert replies[0].meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        # Test case 2: Tool use response
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
        assert replies[0].meta["finish_reason"] == "tool_call"
        assert replies[0].meta["usage"] == {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}

        # Test case 3: Mixed content response
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

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_TOOLS)
    @pytest.mark.integration
    def test_live_run_with_tools(self, model_name, tools):
        """
        Integration test that the AmazonBedrockChatGenerator component can run with tools. Here we are using the
        Haystack tools parameter to pass the tool configuration to the model.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AmazonBedrockChatGenerator(model=model_name, tools=tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = None
        for message in results["replies"]:
            if message.tool_call:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert tool_message.meta["finish_reason"] == "tool_use"

        new_messages = [
            initial_messages[0],
            tool_message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        # Pass the tool result to the model to get the final response
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.parametrize("model_name", STREAMING_TOOL_MODELS)
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, model_name, tools):
        """
        Integration test that the AmazonBedrockChatGenerator component can run with the Haystack tools parameter.
        and the streaming_callback parameter to get the streaming response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AmazonBedrockChatGenerator(model=model_name, tools=tools, streaming_callback=print_streaming_chunk)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = None
        for message in results["replies"]:
            if message.tool_call:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert tool_message.meta["finish_reason"] == "tool_use"

        new_messages = [
            initial_messages[0],
            tool_message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        # Pass the tool result to the model to get the final response
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.parametrize("model_name", [MODELS_TO_TEST_WITH_TOOLS[0]])  # just one model is enough
    @pytest.mark.integration
    def test_pipeline_with_amazon_bedrock_chat_generator(self, model_name, tools):
        """
        Test that the AmazonBedrockChatGenerator component can be used in a pipeline
        """

        pipeline = Pipeline()
        pipeline.add_component("generator", AmazonBedrockChatGenerator(model=model_name, tools=tools))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=tools))

        pipeline.connect("generator", "tool_invoker")

        results = pipeline.run(
            data={"generator": {"messages": [ChatMessage.from_user("What's the weather like in Paris?")]}}
        )

        assert (
            "The weather in Paris is sunny and 32°C"
            == results["tool_invoker"]["tool_messages"][0].tool_call_result.result
        )

    def test_serde_in_pipeline(self, mock_boto3_session, monkeypatch):
        """
        Test serialization/deserialization of AmazonBedrockChatGenerator in a Pipeline,
        including YAML conversion and detailed dictionary validation
        """
        # Set mock AWS credentials
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        # Create a test tool
        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        # Create generator with specific configuration
        generator = AmazonBedrockChatGenerator(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            generation_kwargs={"temperature": 0.7},
            stop_words=["eviscerate"],
            streaming_callback=print_streaming_chunk,
            tools=[tool],
        )

        # Create and configure pipeline
        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        # Get pipeline dictionary and verify its structure
        pipeline_dict = pipeline.to_dict()

        expected_dict = {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": KLASS,
                    "init_parameters": {
                        "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                        "aws_secret_access_key": {
                            "type": "env_var",
                            "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                            "strict": False,
                        },
                        "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                        "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                        "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                        "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                        "generation_kwargs": {"temperature": 0.7},
                        "stop_words": ["eviscerate"],
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "boto3_config": None,
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather",
                                    "description": "useful to determine the weather in a given location",
                                    "parameters": {"city": {"type": "string"}},
                                    "function": "tests.test_chat_generator.weather",
                                },
                            }
                        ],
                    },
                }
            },
            "connections": [],
        }

        if not hasattr(pipeline, "_connection_type_validation"):
            expected_dict.pop("connection_type_validation")

        # add outputs_to_string, inputs_from_state and outputs_to_state tool parameters for compatibility with
        # haystack-ai>=2.12.0
        if hasattr(tool, "outputs_to_string"):
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"][
                "outputs_to_string"
            ] = tool.outputs_to_string
        if hasattr(tool, "inputs_from_state"):
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"][
                "inputs_from_state"
            ] = tool.inputs_from_state
        if hasattr(tool, "outputs_to_state"):
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"][
                "outputs_to_state"
            ] = tool.outputs_to_state

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


class TestAmazonBedrockChatGeneratorAsyncInference:
    """
    Test class for async inference functionality of AmazonBedrockChatGenerator
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    @pytest.mark.integration
    async def test_async_default_inference_params(self, model_name, chat_messages):
        """
        Test basic async chat completion without streaming
        """
        client = AmazonBedrockChatGenerator(model=model_name)
        response = await client.run_async(chat_messages)

        assert "replies" in response, "Response does not contain 'replies' key"
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert "paris" in first_reply.text.lower(), "First reply does not contain 'paris'"
        assert first_reply.meta, "First reply has no metadata"

        if first_reply.meta and "usage" in first_reply.meta:
            assert "prompt_tokens" in first_reply.meta["usage"]
            assert "completion_tokens" in first_reply.meta["usage"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_TOOLS)
    @pytest.mark.integration
    async def test_async_tools_use(self, model_name):
        """
        Test async tools use with passing the generation_kwargs={"toolConfig": tool_config}
        """
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "top_song",
                        "description": "Get the most popular song played on a radio station.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station "
                                        "for which you want the most popular song. "
                                        "Example calls signs are WZPZ and WKRP.",
                                    }
                                },
                                "required": ["sign"],
                            }
                        },
                    }
                }
            ],
            "toolChoice": {"auto": {}},
        }

        messages = [ChatMessage.from_user("What is the most popular song on WZPZ?")]
        client = AmazonBedrockChatGenerator(model=model_name)
        response = await client.run_async(messages=messages, generation_kwargs={"toolConfig": tool_config})
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = next((msg for msg in replies if msg.tool_call), None)
        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "top_song", f"{tool_call} does not contain the correct 'tool_name' value"
        assert tool_call.arguments, f"Tool call {tool_call} does not contain 'arguments' value"
        assert (
            tool_call.arguments["sign"] == "WZPZ"
        ), f"Tool call {tool_call} does not contain the correct 'arguments' value"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_TOOLS)
    @pytest.mark.integration
    async def test_async_live_run_with_tools(self, model_name, tools):
        """
        Integration test that the AmazonBedrockChatGenerator component can run asynchronously with tools
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AmazonBedrockChatGenerator(model=model_name, tools=tools)
        results = await component.run_async(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = next((msg for msg in results["replies"] if msg.tool_call), None)
        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert tool_message.meta["finish_reason"] == "tool_use"

        new_messages = [
            initial_messages[0],
            tool_message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        # Pass the tool result to the model to get the final response
        results = await component.run_async(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", STREAMING_TOOL_MODELS)
    @pytest.mark.integration
    async def test_async_inference_with_streaming(self, model_name, chat_messages):
        """
        Test async chat completion with streaming
        """
        streaming_callback_called = False
        paris_found_in_response = False

        async def streaming_callback(chunk: StreamingChunk):
            nonlocal streaming_callback_called, paris_found_in_response
            streaming_callback_called = True
            assert isinstance(chunk, StreamingChunk)
            assert chunk.content is not None
            if not paris_found_in_response:
                paris_found_in_response = "paris" in chunk.content.lower()

        client = AmazonBedrockChatGenerator(model=model_name)
        response = await client.run_async(chat_messages, streaming_callback=streaming_callback)

        assert streaming_callback_called, "Streaming callback was not called"
        assert paris_found_in_response, "The streaming callback response did not contain 'paris'"
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply has no content"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert "paris" in first_reply.text.lower(), "First reply does not contain 'paris'"
        assert first_reply.meta, "First reply has no metadata"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", STREAMING_TOOL_MODELS)
    @pytest.mark.integration
    async def test_async_tools_use_with_streaming(self, model_name):
        """
        Test async tools use with streaming
        """
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "top_song",
                        "description": "Get the most popular song played on a radio station.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station "
                                        "for which you want the most popular song. Example "
                                        "calls signs are WZPZ and WKRP.",
                                    }
                                },
                                "required": ["sign"],
                            }
                        },
                    }
                }
            ],
            "toolChoice": {"auto": {}},
        }

        async def streaming_callback(chunk: StreamingChunk):
            print(chunk, flush=True, end="")  # noqa: T201

        messages = [ChatMessage.from_user("What is the most popular song on WZPZ?")]
        client = AmazonBedrockChatGenerator(model=model_name, streaming_callback=streaming_callback)
        response = await client.run_async(messages=messages, generation_kwargs={"toolConfig": tool_config})
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        # Find the message containing the tool call
        tool_message = next((msg for msg in replies if msg.tool_call), None)
        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "top_song", f"{tool_call} does not contain the correct 'tool_name' value"
        assert tool_call.arguments, f"{tool_call} does not contain 'arguments' value"
        assert tool_call.arguments["sign"] == "WZPZ", f"{tool_call} does not contain the correct 'input' value"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", STREAMING_TOOL_MODELS)
    @pytest.mark.integration
    async def test_async_live_run_with_tools_streaming(self, model_name, tools):
        """
        Integration test that the AmazonBedrockChatGenerator component can run asynchronously with tools and streaming
        """

        async def streaming_callback(chunk: StreamingChunk):
            print(chunk, flush=True, end="")  # noqa: T201

        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AmazonBedrockChatGenerator(model=model_name, tools=tools, streaming_callback=streaming_callback)
        results = await component.run_async(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = next((msg for msg in results["replies"] if msg.tool_call), None)
        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert tool_message.meta["finish_reason"] == "tool_use"

        new_messages = [
            initial_messages[0],
            tool_message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]

        # Pass the tool result to the model to get the final response
        results = await component.run_async(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
