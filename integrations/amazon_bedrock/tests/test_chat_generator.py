import os
from typing import Any, Dict, Optional

import pytest
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, ImageContent, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

CLASS_TYPE = "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
MODELS_TO_TEST = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "cohere.command-r-plus-v1:0",
    "mistral.mistral-large-2402-v1:0",
]
MODELS_TO_TEST_WITH_TOOLS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "cohere.command-r-plus-v1:0",
    "mistral.mistral-large-2402-v1:0",
]

# so far we've discovered these models support streaming and tool use
STREAMING_TOOL_MODELS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "cohere.command-r-plus-v1:0",
]

MODELS_TO_TEST_WITH_IMAGE_INPUT = [
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
]

MODELS_TO_TEST_WITH_THINKING = [
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
]


def hello_world():
    return "Hello, World!"


@pytest.fixture
def tool_with_no_parameters():
    tool = Tool(
        name="hello_world",
        description="This prints hello world",
        parameters={"properties": {}, "type": "object"},
        function=hello_world,
    )
    return tool


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


def population(city: str):
    """Get population for a given city."""
    return f"The population of {city} is 2.2 million"


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


@pytest.fixture
def mixed_tools():
    """Fixture that returns a mixed list of Tool and Toolset."""
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=weather,
    )
    population_tool = Tool(
        name="population",
        description="useful to determine the population of a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=population,
    )
    toolset = Toolset([population_tool])
    return [weather_tool, toolset]


@pytest.fixture
def chat_messages():
    messages = [
        ChatMessage.from_system("\\nYou are a helpful assistant, be super brief in your responses."),
        ChatMessage.from_user("What's the capital of France?"),
    ]
    return messages


# See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolConfiguration.html
@pytest.fixture
def top_song_tool_config():
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
        # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
        "toolChoice": {"auto": {}},
    }
    return tool_config


class TestAmazonBedrockChatGenerator:
    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 1000}])
    def test_to_dict(self, mock_boto3_session, boto3_config):
        """
        Test that the to_dict method returns the correct dictionary without aws credentials
        """
        generator = AmazonBedrockChatGenerator(
            model="cohere.command-r-plus-v1:0",
            generation_kwargs={"temperature": 0.7},
            streaming_callback=print_streaming_chunk,
            boto3_config=boto3_config,
            guardrail_config={"guardrailIdentifier": "test", "guardrailVersion": "test"},
        )
        expected_dict = {
            "type": CLASS_TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "model": "cohere.command-r-plus-v1:0",
                "generation_kwargs": {"temperature": 0.7},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "boto3_config": boto3_config,
                "tools": None,
                "guardrail_config": {"guardrailIdentifier": "test", "guardrailVersion": "test"},
            },
        }

        assert generator.to_dict() == expected_dict

    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 1000}])
    def test_from_dict(self, mock_boto3_session: Any, boto3_config: Optional[Dict[str, Any]]):
        """
        Test that the from_dict method returns the correct object
        """
        generator = AmazonBedrockChatGenerator.from_dict(
            {
                "type": CLASS_TYPE,
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
                    "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                    "boto3_config": boto3_config,
                    "tools": None,
                    "stop_words": ["stop"],  # this parameter will be ignored
                    "guardrail_config": None,
                },
            }
        )
        assert generator.model == "anthropic.claude-3-5-sonnet-20240620-v1:0"
        assert generator.streaming_callback == print_streaming_chunk
        assert generator.boto3_config == boto3_config

    def test_default_constructor(self, mock_boto3_session, mock_aioboto3_session, set_env_variables):
        """
        Test that the default constructor sets the correct values
        """
        layer = AmazonBedrockChatGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
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

    def test_constructor_with_generation_kwargs(self, mock_boto3_session):
        """
        Test that model_kwargs are correctly set in the constructor
        """
        generation_kwargs = {"temperature": 0.7}
        layer = AmazonBedrockChatGenerator(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0", generation_kwargs=generation_kwargs
        )
        assert layer.generation_kwargs == generation_kwargs

    def test_constructor_with_empty_model(self):
        """
        Test that the constructor raises an error when the model is empty
        """
        with pytest.raises(ValueError, match="cannot be None or empty string"):
            AmazonBedrockChatGenerator(model="")

    def test_serde_in_pipeline(self, mock_boto3_session, monkeypatch):
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

        generator = AmazonBedrockChatGenerator(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            generation_kwargs={"temperature": 0.7, "stopSequences": ["eviscerate"]},
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
                    "type": CLASS_TYPE,
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
                        "generation_kwargs": {"temperature": 0.7, "stopSequences": ["eviscerate"]},
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
                                    "outputs_to_string": None,
                                    "inputs_from_state": None,
                                    "outputs_to_state": None,
                                },
                            }
                        ],
                        "guardrail_config": None,
                    },
                }
            },
            "connections": [],
        }

        if not hasattr(pipeline, "_connection_type_validation"):
            expected_dict.pop("connection_type_validation")

        assert pipeline_dict == expected_dict

    def test_prepare_request_params_tool_config(self, top_song_tool_config, mock_boto3_session, set_env_variables):
        generator = AmazonBedrockChatGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
        request_params, _ = generator._prepare_request_params(
            messages=[ChatMessage.from_user("What's the capital of France?")],
            generation_kwargs={"toolConfig": top_song_tool_config},
            tools=None,
        )
        assert request_params["messages"] == [{"content": [{"text": "What's the capital of France?"}], "role": "user"}]
        assert request_params["toolConfig"] == top_song_tool_config

    def test_prepare_request_params_guardrail_config(self, mock_boto3_session, set_env_variables):
        generator = AmazonBedrockChatGenerator(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            guardrail_config={"guardrailIdentifier": "test", "guardrailVersion": "test"},
        )
        request_params, _ = generator._prepare_request_params(
            messages=[ChatMessage.from_user("What's the capital of France?")],
        )
        assert request_params["messages"] == [{"content": [{"text": "What's the capital of France?"}], "role": "user"}]
        assert request_params["guardrailConfig"] == {"guardrailIdentifier": "test", "guardrailVersion": "test"}

    def test_init_with_mixed_tools(self, mock_boto3_session, set_env_variables):
        def tool_fn(city: str) -> str:
            return city

        weather_tool = Tool(
            name="weather",
            description="Weather lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=tool_fn,
        )
        population_tool = Tool(
            name="population",
            description="Population lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=tool_fn,
        )
        toolset = Toolset([population_tool])

        generator = AmazonBedrockChatGenerator(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            tools=[weather_tool, toolset],
        )

        assert generator.tools == [weather_tool, toolset]

    def test_prepare_request_params_with_mixed_tools(self, mock_boto3_session, set_env_variables):
        def tool_fn(city: str) -> str:
            return city

        weather_tool = Tool(
            name="weather",
            description="Weather lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=tool_fn,
        )
        population_tool = Tool(
            name="population",
            description="Population lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=tool_fn,
        )
        toolset = Toolset([population_tool])

        generator = AmazonBedrockChatGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
        request_params, _ = generator._prepare_request_params(
            messages=[ChatMessage.from_user("What's the capital of France?")],
            tools=[weather_tool, toolset],
        )

        assert request_params["toolConfig"] == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "weather",
                        "description": "Weather lookup",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            }
                        },
                    }
                },
                {
                    "toolSpec": {
                        "name": "population",
                        "description": "Population lookup",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            }
                        },
                    }
                },
            ]
        }


# In the CI, those tests are skipped if AWS Authentication fails
@pytest.mark.integration
class TestAmazonBedrockChatGeneratorInference:
    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
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

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_IMAGE_INPUT)
    def test_run_with_image_input(self, model_name, test_files_path):
        client = AmazonBedrockChatGenerator(model=model_name)

        image_path = test_files_path / "apple.jpg"
        image_content = ImageContent.from_file_path(image_path, size=(100, 100))

        chat_message = ChatMessage.from_user(content_parts=["What's in the image? Max 5 words.", image_content])

        response = client.run([chat_message])

        first_reply = response["replies"][0]
        assert isinstance(first_reply, ChatMessage)
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT)
        assert first_reply.text
        assert "apple" in first_reply.text.lower()

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
    def test_default_inference_with_streaming(self, model_name, chat_messages):
        streaming_callback_called = False
        paris_found_in_response = False

        def streaming_callback(chunk: StreamingChunk):
            nonlocal streaming_callback_called, paris_found_in_response
            streaming_callback_called = True
            assert isinstance(chunk, StreamingChunk)
            assert chunk.content is not None
            assert chunk.component_info is not None
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
    def test_live_run_with_multi_tool_calls(self, model_name, tools):
        """
        Integration test that the AmazonBedrockChatGenerator component can run with tools. Here we are using the
        Haystack tools parameter to pass the tool configuration to the model.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = AmazonBedrockChatGenerator(model=model_name, tools=tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_call_message = message
                break

        assert tool_call_message is not None, "No message with tool call found"
        assert isinstance(tool_call_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_calls = tool_call_message.tool_calls
        assert len(tool_calls) == 2
        for tool_call in tool_calls:
            assert tool_call.id, "Tool call does not contain value for 'id' key"
            assert tool_call.tool_name == "weather"
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert tool_call_message.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = [
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call) for tool_call in tool_calls
        ]

        new_messages = [*initial_messages, tool_call_message, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_TOOLS)
    def test_live_run_with_mixed_tools(self, model_name, mixed_tools):
        """
        Integration test that verifies AmazonBedrockChatGenerator works with mixed Tool and Toolset.
        This tests that the LLM can correctly invoke tools from both a standalone Tool and a Toolset.
        """
        initial_messages = [
            ChatMessage.from_user("What's the weather like in Paris and what is the population of Berlin?")
        ]
        component = AmazonBedrockChatGenerator(model=model_name, tools=mixed_tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_call_message = message
                break

        assert tool_call_message is not None, "No message with tool call found"
        assert isinstance(tool_call_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_calls = tool_call_message.tool_calls
        assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

        # Verify we got calls to both weather and population tools
        tool_names = {tc.tool_name for tc in tool_calls}
        assert "weather" in tool_names, "Expected 'weather' tool call"
        assert "population" in tool_names, "Expected 'population' tool call"

        # Verify tool call details
        for tool_call in tool_calls:
            assert tool_call.id, "Tool call does not contain value for 'id' key"
            assert tool_call.tool_name in ["weather", "population"]
            assert "city" in tool_call.arguments
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert tool_call_message.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = []
        for tool_call in tool_calls:
            if tool_call.tool_name == "weather":
                result = "The weather in Paris is sunny and 32°C"
            else:  # population
                result = "The population of Berlin is 2.2 million"
            tool_result_messages.append(ChatMessage.from_tool(tool_result=result, origin=tool_call))

        new_messages = [*initial_messages, tool_call_message, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_THINKING)
    def test_live_run_with_tool_call_and_thinking(self, model_name, tools):
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AmazonBedrockChatGenerator(
            model=model_name,
            tools=tools,
            generation_kwargs={
                "maxTokens": 8192,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024,
                },
            },
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_call_message = message
                break

        assert tool_call_message is not None, "No message with tool call found"
        assert isinstance(tool_call_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        assert tool_call_message.reasoning is not None, "Tool message does not contain reasoning"

        tool_calls = tool_call_message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id, "Tool call does not contain value for 'id' key"
        assert tool_calls[0].tool_name == "weather"
        assert tool_calls[0].arguments["city"] == "Paris"
        assert tool_call_message.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = [
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call) for tool_call in tool_calls
        ]

        new_messages = [*initial_messages, tool_call_message, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.parametrize("model_name", MODELS_TO_TEST_WITH_THINKING)
    def test_live_run_with_tool_call_and_thinking_streaming(self, model_name, tools):
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AmazonBedrockChatGenerator(
            model=model_name,
            tools=tools,
            generation_kwargs={
                "maxTokens": 8192,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024,
                },
            },
            streaming_callback=print_streaming_chunk,
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_call_message = message
                break

        assert tool_call_message is not None, "No message with tool call found"
        assert isinstance(tool_call_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        assert tool_call_message.reasoning is not None, "Tool message does not contain reasoning"

        tool_calls = tool_call_message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id, "Tool call does not contain value for 'id' key"
        assert tool_calls[0].tool_name == "weather"
        assert tool_calls[0].arguments["city"] == "Paris"
        assert tool_call_message.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = [
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call) for tool_call in tool_calls
        ]

        new_messages = [*initial_messages, tool_call_message, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    def test_live_run_with_redacted_thinking(self, tools):
        # Special prompt to induce redacted thinking
        initial_messages = [
            ChatMessage.from_user(
                "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
            )
        ]
        component = AmazonBedrockChatGenerator(
            # Redacted thinking only happens with Claude 3.7 https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-thinking-encryption.html
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            tools=tools,
            generation_kwargs={
                "maxTokens": 8192,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024,
                },
            },
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"
        assert isinstance(
            results["replies"][0].reasoning.extra["reasoning_contents"][0]["reasoning_content"]["redacted_content"],
            bytes,
        )

    def test_live_run_with_redacted_thinking_streaming(self, tools):
        # Special prompt to induce redacted thinking
        initial_messages = [
            ChatMessage.from_user(
                "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
            )
        ]
        component = AmazonBedrockChatGenerator(
            # Redacted thinking only happens with Claude 3.7 https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-thinking-encryption.html
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            tools=tools,
            generation_kwargs={
                "maxTokens": 8192,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024,
                },
            },
            streaming_callback=print_streaming_chunk,
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"
        assert isinstance(
            results["replies"][0].reasoning.extra["reasoning_contents"][0]["reasoning_content"]["redacted_content"],
            bytes,
        )

    @pytest.mark.parametrize("model_name", STREAMING_TOOL_MODELS)
    def test_live_run_with_multi_tool_calls_streaming(self, model_name, tools):
        """
        Integration test that the AmazonBedrockChatGenerator component can run with the Haystack tools parameter.
        and the streaming_callback parameter to get the streaming response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = AmazonBedrockChatGenerator(model=model_name, tools=tools, streaming_callback=print_streaming_chunk)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_call_message = message
                break

        assert tool_call_message is not None, "No message with tool call found"
        assert isinstance(tool_call_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_calls = tool_call_message.tool_calls
        for tool_call in tool_calls:
            assert tool_call.id, "Tool call does not contain value for 'id' key"
            assert tool_call.tool_name == "weather"
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert tool_call_message.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = [
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call) for tool_call in tool_calls
        ]

        new_messages = [*initial_messages, tool_call_message, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    @pytest.mark.parametrize("model_name", [STREAMING_TOOL_MODELS[1]])  # just one model is enough
    def test_live_run_with_tool_with_no_args_streaming(self, tool_with_no_parameters, model_name):
        """
        Integration test that the AmazonBedrockChatGenerator component can run with a tool that has no arguments and
        streaming.
        """
        initial_messages = [ChatMessage.from_user("Print Hello World using the print hello world tool.")]
        component = AmazonBedrockChatGenerator(
            model=model_name, tools=[tool_with_no_parameters], streaming_callback=print_streaming_chunk
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # # this is Claude thinking message prior to tool call
        # assert message.text is not None

        # now we have the tool call
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "hello_world"
        assert tool_call.arguments == {}
        assert message.meta["finish_reason"] == "tool_calls"

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="Hello World!", origin=tool_call),
        ]
        results = component.run(new_messages)
        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "hello" in final_message.text.lower()

    @pytest.mark.skipif(
        not os.getenv("AWS_BEDROCK_GUARDRAIL_ID") or not os.getenv("AWS_BEDROCK_GUARDRAIL_VERSION"),
        reason=(
            "Export AWS_BEDROCK_GUARDRAIL_ID and AWS_BEDROCK_GUARDRAIL_VERSION environment variables corresponding"
            "to a Bedrock Guardrail to run this test."
        ),
    )
    @pytest.mark.parametrize("streaming_callback", [None, print_streaming_chunk])
    def test_live_run_with_guardrail(self, streaming_callback):
        messages = [ChatMessage.from_user("Should I invest in Tesla or Apple?")]
        component = AmazonBedrockChatGenerator(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            guardrail_config={
                "guardrailIdentifier": os.getenv("AWS_BEDROCK_GUARDRAIL_ID"),
                "guardrailVersion": os.getenv("AWS_BEDROCK_GUARDRAIL_VERSION"),
                "trace": "enabled",
            },
            streaming_callback=streaming_callback,
        )
        results = component.run(messages=messages)

        assert results["replies"][0].meta["finish_reason"] == "content_filter"
        assert results["replies"][0].text == "Sorry, the model cannot answer this question."
        assert "trace" in results["replies"][0].meta
        assert "guardrail" in results["replies"][0].meta["trace"]

    @pytest.mark.parametrize("model_name", [MODELS_TO_TEST_WITH_TOOLS[0]])  # just one model is enough
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


# In the CI, those tests are skipped if AWS Authentication fails
@pytest.mark.integration
class TestAmazonBedrockChatGeneratorAsyncInference:
    """
    Test class for async inference functionality of AmazonBedrockChatGenerator
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", MODELS_TO_TEST)
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
    async def test_async_live_run_with_multi_tool_calls(self, model_name, tools):
        """
        Integration test that the AmazonBedrockChatGenerator component can run asynchronously with tools
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = AmazonBedrockChatGenerator(model=model_name, tools=tools)
        results = await component.run_async(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = next((msg for msg in results["replies"] if msg.tool_calls), None)
        assert tool_call_message is not None, "No message with tool call found"
        assert isinstance(tool_call_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_calls = tool_call_message.tool_calls
        for tool_call in tool_calls:
            assert tool_call.id, "Tool call does not contain value for 'id' key"
            assert tool_call.tool_name == "weather"
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert tool_call_message.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = [
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call) for tool_call in tool_calls
        ]
        new_messages = [*initial_messages, tool_call_message, *tool_result_messages]
        results = await component.run_async(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", STREAMING_TOOL_MODELS)
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
    async def test_async_live_run_with_multi_tool_calls_streaming(self, model_name, tools):
        """
        Integration test that the AmazonBedrockChatGenerator component can run asynchronously with tools and streaming
        """

        async def streaming_callback(chunk: StreamingChunk):
            print(chunk, flush=True, end="")  # noqa: T201

        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = AmazonBedrockChatGenerator(model=model_name, tools=tools, streaming_callback=streaming_callback)
        results = await component.run_async(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = next((msg for msg in results["replies"] if msg.tool_calls), None)
        assert tool_call_message is not None, "No message with tool call found"
        assert isinstance(tool_call_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_calls = tool_call_message.tool_calls
        for tool_call in tool_calls:
            assert tool_call.id, "Tool call does not contain value for 'id' key"
            assert tool_call.tool_name == "weather"
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert tool_call_message.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = [
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call) for tool_call in tool_calls
        ]
        new_messages = [*initial_messages, tool_call_message, *tool_result_messages]
        results = await component.run_async(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()
