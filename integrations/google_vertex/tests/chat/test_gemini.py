import json
from typing import Annotated, Literal
from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Pipeline
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, TextContent, ToolCall
from haystack.tools import Tool, create_tool_from_function
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    ToolConfig,
)

from haystack_integrations.components.generators.google_vertex.chat.gemini import (
    VertexAIGeminiChatGenerator,
    _convert_chatmessage_to_google_content,
)


def get_current_weather(
    city: Annotated[str, "the city for which to get the weather, e.g. 'San Francisco'"] = "Munich",
    unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius",
):
    """A simple function to get the current weather for a location."""
    return f"Weather report for {city}: 20 {unit}, sunny"


@pytest.fixture
def tools():
    tool = create_tool_from_function(get_current_weather)
    return [tool]


def test_convert_chatmessage_to_google_content():
    chat_message = ChatMessage.from_assistant("Hello, how are you?")
    google_content = _convert_chatmessage_to_google_content(chat_message)

    assert google_content.parts[0].text == "Hello, how are you?"
    assert google_content.role == "model"

    message = ChatMessage.from_user("I have a question")
    google_content = _convert_chatmessage_to_google_content(message)
    assert google_content.parts[0].text == "I have a question"
    assert google_content.role == "user"

    message = ChatMessage.from_assistant(
        tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})]
    )
    google_content = _convert_chatmessage_to_google_content(message)
    assert google_content.parts[0].function_call.name == "weather"
    assert google_content.parts[0].function_call.args == {"city": "Paris"}
    assert google_content.role == "model"

    tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
    message = ChatMessage.from_tool(
        tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
    )
    google_content = _convert_chatmessage_to_google_content(message)
    assert google_content.parts[0].function_response.name == "weather"
    assert google_content.parts[0].function_response.response == {"result": tool_result}
    assert google_content.role == "user"


def test_convert_chatmessage_to_google_content_invalid():
    message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
    with pytest.raises(ValueError):
        _convert_chatmessage_to_google_content(message)

    message = ChatMessage(
        _role=ChatRole.ASSISTANT,
        _content=[TextContent(text="I have an answer"), TextContent(text="I have another answer")],
    )
    with pytest.raises(ValueError):
        _convert_chatmessage_to_google_content(message)

    message = ChatMessage.from_system("You are a helpful assistant.")
    with pytest.raises(ValueError):
        _convert_chatmessage_to_google_content(message)


class TestVertexAIGeminiChatGenerator:

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_init(self, mock_vertexai_init, _mock_generative_model, tools):

        generation_config = GenerationConfig(
            candidate_count=1,
            stop_sequences=["stop"],
            max_output_tokens=10,
            temperature=0.5,
            top_p=0.5,
            top_k=0.5,
        )
        safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                allowed_function_names=["get_current_weather_func"],
            )
        )

        gemini = VertexAIGeminiChatGenerator(
            project_id="TestID123",
            location="TestLocation",
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
        )
        mock_vertexai_init.assert_called()
        assert gemini._model_name == "gemini-1.5-flash"
        assert gemini._generation_config == generation_config
        assert gemini._safety_settings == safety_settings
        assert gemini._tools == tools
        assert gemini._tool_config == tool_config

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_to_dict(self, _mock_vertexai_init, _mock_generative_model):

        gemini = VertexAIGeminiChatGenerator()
        assert gemini.to_dict() == {
            "type": "haystack_integrations.components.generators.google_vertex.chat.gemini.VertexAIGeminiChatGenerator",
            "init_parameters": {
                "model": "gemini-1.5-flash",
                "project_id": None,
                "location": None,
                "generation_config": None,
                "safety_settings": None,
                "streaming_callback": None,
                "tools": None,
                "tool_config": None,
            },
        }

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_to_dict_with_params(self, _mock_vertexai_init, _mock_generative_model):
        tools = [Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)]

        generation_config = GenerationConfig(
            candidate_count=1,
            stop_sequences=["stop"],
            max_output_tokens=10,
            temperature=0.5,
            top_p=0.5,
            top_k=2,
        )
        safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                allowed_function_names=["get_current_weather_func"],
            )
        )

        gemini = VertexAIGeminiChatGenerator(
            project_id="TestID123",
            location="TestLocation",
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
        )

        assert gemini.to_dict() == {
            "type": "haystack_integrations.components.generators.google_vertex.chat.gemini.VertexAIGeminiChatGenerator",
            "init_parameters": {
                "model": "gemini-1.5-flash",
                "project_id": "TestID123",
                "location": "TestLocation",
                "generation_config": {
                    "temperature": 0.5,
                    "top_p": 0.5,
                    "top_k": 2.0,
                    "candidate_count": 1,
                    "max_output_tokens": 10,
                    "stop_sequences": ["stop"],
                },
                "safety_settings": {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH},
                "streaming_callback": None,
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "name": "name",
                            "parameters": {"x": {"type": "string"}},
                        },
                    }
                ],
                "tool_config": {
                    "function_calling_config": {
                        "mode": ToolConfig.FunctionCallingConfig.Mode.ANY,
                        "allowed_function_names": ["get_current_weather_func"],
                    }
                },
            },
        }

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_from_dict(self, _mock_vertexai_init, _mock_generative_model):
        gemini = VertexAIGeminiChatGenerator.from_dict(
            {
                "type": (
                    "haystack_integrations.components.generators.google_vertex.chat.gemini."
                    "VertexAIGeminiChatGenerator"
                ),
                "init_parameters": {
                    "project_id": None,
                    "model": "gemini-1.5-flash",
                    "generation_config": None,
                    "safety_settings": None,
                    "tools": None,
                    "streaming_callback": None,
                },
            }
        )

        assert gemini._model_name == "gemini-1.5-flash"
        assert gemini._project_id is None
        assert gemini._safety_settings is None
        assert gemini._tools is None
        assert gemini._tool_config is None
        assert gemini._generation_config is None

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_from_dict_with_param(self, _mock_vertexai_init, _mock_generative_model):
        tools = [Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)]

        gemini = VertexAIGeminiChatGenerator.from_dict(
            {
                "type": (
                    "haystack_integrations.components.generators.google_vertex.chat.gemini."
                    "VertexAIGeminiChatGenerator"
                ),
                "init_parameters": {
                    "project_id": "TestID123",
                    "location": "TestLocation",
                    "model": "gemini-1.5-flash",
                    "generation_config": {
                        "temperature": 0.5,
                        "top_p": 0.5,
                        "top_k": 0.5,
                        "candidate_count": 1,
                        "max_output_tokens": 10,
                        "stop_sequences": ["stop"],
                    },
                    "safety_settings": {
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
                    },
                    "tools": [
                        {
                            "type": "haystack.tools.tool.Tool",
                            "data": {
                                "description": "description",
                                "function": "builtins.print",
                                "name": "name",
                                "parameters": {"x": {"type": "string"}},
                            },
                        }
                    ],
                    "tool_config": {
                        "function_calling_config": {
                            "mode": ToolConfig.FunctionCallingConfig.Mode.ANY,
                            "allowed_function_names": ["get_current_weather_func"],
                        }
                    },
                    "streaming_callback": None,
                },
            }
        )

        assert gemini._model_name == "gemini-1.5-flash"
        assert gemini._project_id == "TestID123"
        assert gemini._location == "TestLocation"
        assert gemini._safety_settings == {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
        }
        assert gemini._tools == tools
        assert isinstance(gemini._tool_config, ToolConfig)
        assert isinstance(gemini._generation_config, GenerationConfig)
        assert (
            gemini._tool_config._gapic_tool_config.function_calling_config.mode
            == ToolConfig.FunctionCallingConfig.Mode.ANY
        )

    def test_convert_to_vertex_tools(self, tools):
        vertex_tools = VertexAIGeminiChatGenerator._convert_to_vertex_tools(tools)

        function_declaration = vertex_tools[0]._raw_tool.function_declarations[0]
        assert function_declaration.name == tools[0].name
        assert function_declaration.description == tools[0].description

        assert function_declaration.parameters

        # check if default values are removed. This type is not easily inspectable
        assert "default" not in str(function_declaration.parameters)

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_run(self, mock_generative_model):
        mock_model = Mock()
        mock_candidate = MagicMock(
            content=Content(parts=[Part.from_text("This is a generated response.")], role="model")
        )
        mock_response = MagicMock(spec=GenerationResponse, candidates=[mock_candidate])

        mock_model.send_message.return_value = mock_response
        mock_model.start_chat.return_value = mock_model
        mock_generative_model.return_value = mock_model

        messages = [
            ChatMessage.from_system("You are a helpful assistant"),
            ChatMessage.from_user("What's the capital of France?"),
        ]
        gemini = VertexAIGeminiChatGenerator()
        response = gemini.run(messages=messages)

        mock_model.send_message.assert_called_once()
        assert "replies" in response
        reply = response["replies"][0]
        assert reply.role == ChatRole.ASSISTANT
        assert reply.text == "This is a generated response."

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_run_with_streaming_callback(self, mock_generative_model):
        mock_model = Mock()

        mock_responses = [
            MagicMock(
                spec=GenerationResponse,
                to_dict=lambda: {
                    "candidates": [{"content": {"parts": [{"text": "First part "}]}}],
                },
            ),
            MagicMock(
                spec=GenerationResponse,
                to_dict=lambda: {
                    "candidates": [{"content": {"parts": [{"text": "Second part"}]}}],
                    "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
                },
            ),
        ]

        mock_model.send_message.return_value = iter(mock_responses)
        mock_model.start_chat.return_value = mock_model
        mock_generative_model.return_value = mock_model

        received_chunks = []

        def streaming_callback(chunk: StreamingChunk) -> None:
            received_chunks.append(chunk)

        gemini = VertexAIGeminiChatGenerator(streaming_callback=streaming_callback)
        messages = [
            ChatMessage.from_system("You are a helpful assistant"),
            ChatMessage.from_user("What's the capital of France?"),
        ]

        response = gemini.run(messages=messages)

        assert len(received_chunks) == 2
        assert received_chunks[0].content == "First part "
        assert received_chunks[1].content == "Second part"

        assert "replies" in response
        reply = response["replies"][0]
        assert reply.role == ChatRole.ASSISTANT
        assert reply.text == "First part Second part"

        assert reply.meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_run_with_tools(self, mock_generative_model, tools):
        mock_model = Mock()
        mock_candidate = MagicMock(
            content=Content(
                parts=[
                    Part.from_dict(
                        {"function_call": {"name": "get_current_weather", "args": {"city": "Paris", "unit": "Celsius"}}}
                    ),
                ],
                role="model",
            )
        )
        mock_response = MagicMock(spec=GenerationResponse, candidates=[mock_candidate])

        mock_model.send_message.return_value = mock_response
        mock_model.start_chat.return_value = mock_model
        mock_generative_model.return_value = mock_model

        messages = [
            ChatMessage.from_user("What's the weather in Paris?"),
        ]

        gemini = VertexAIGeminiChatGenerator(tools=tools)
        response = gemini.run(messages=messages)

        mock_model.send_message.assert_called_once()
        call_kwargs = mock_model.send_message.call_args.kwargs
        assert "tools" in call_kwargs

        assert "replies" in response
        reply = response["replies"][0]
        assert reply.role == ChatRole.ASSISTANT
        assert not reply.text
        assert len(reply.tool_calls) == 1
        assert reply.tool_calls[0].tool_name == "get_current_weather"
        assert reply.tool_calls[0].arguments == {"city": "Paris", "unit": "Celsius"}

    @patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
    def test_run_with_muliple_tools_and_streaming(self, mock_generative_model, tools):
        """
        Test that the generator can handle multiple tools and streaming.
        Note: this test case is made up because in practice I have always seen multiple function calls in a single
        streaming chunk.
        """

        def population(city: Annotated[str, "the city for which to get the population, e.g. 'Munich'"] = "Munich"):
            """A simple function to get the population for a location."""
            return f"Population of {city}: 1,000,000"

        multiple_tools = [tools[0], create_tool_from_function(population)]

        mock_model = Mock()

        mock_responses = [
            MagicMock(
                spec=GenerationResponse,
                to_dict=lambda: {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "function_call": {
                                            "name": "get_current_weather",
                                            "args": {"city": "Munich", "unit": "Farenheit"},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            ),
            MagicMock(
                spec=GenerationResponse,
                to_dict=lambda: {
                    "candidates": [
                        {"content": {"parts": [{"function_call": {"name": "population", "args": {"city": "Munich"}}}]}}
                    ],
                    "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
                },
            ),
        ]

        mock_model.send_message.return_value = iter(mock_responses)
        mock_model.start_chat.return_value = mock_model
        mock_generative_model.return_value = mock_model

        received_chunks = []

        def streaming_callback(chunk: StreamingChunk) -> None:
            received_chunks.append(chunk)

        messages = [
            ChatMessage.from_user("What's the weather in Munich (in Farenheit) and how many people live there?"),
        ]

        gemini = VertexAIGeminiChatGenerator(tools=multiple_tools, streaming_callback=streaming_callback)
        response = gemini.run(messages=messages)

        assert len(received_chunks) == 2
        assert json.loads(received_chunks[0].content) == {
            "name": "get_current_weather",
            "args": {"city": "Munich", "unit": "Farenheit"},
        }
        assert json.loads(received_chunks[1].content) == {"name": "population", "args": {"city": "Munich"}}

        assert "replies" in response
        reply = response["replies"][0]
        assert reply.role == ChatRole.ASSISTANT
        assert not reply.text
        assert len(reply.tool_calls) == 2
        assert reply.tool_calls[0].tool_name == "get_current_weather"
        assert reply.tool_calls[0].arguments == {"city": "Munich", "unit": "Farenheit"}
        assert reply.tool_calls[1].tool_name == "population"
        assert reply.tool_calls[1].arguments == {"city": "Munich"}
        assert reply.meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_serde_in_pipeline(self):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = VertexAIGeminiChatGenerator(
            project_id="TestID123",
            model="gemini-1.5-flash",
            generation_config=GenerationConfig(
                temperature=0.6,
                stop_sequences=["stop", "words"],
            ),
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "generator": {
                    "type": (
                        "haystack_integrations.components.generators.google_vertex.chat.gemini."
                        "VertexAIGeminiChatGenerator"
                    ),
                    "init_parameters": {
                        "project_id": "TestID123",
                        "model": "gemini-1.5-flash",
                        "generation_config": {
                            "temperature": 0.6,
                            "stop_sequences": ["stop", "words"],
                        },
                        "location": None,
                        "safety_settings": None,
                        "streaming_callback": None,
                        "tool_config": None,
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "name",
                                    "description": "description",
                                    "parameters": {"x": {"type": "string"}},
                                    "function": "builtins.print",
                                },
                            }
                        ],
                    },
                }
            },
            "connections": [],
        }

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline
