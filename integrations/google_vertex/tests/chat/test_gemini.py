from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Tool,
    ToolConfig,
)

from haystack_integrations.components.generators.google_vertex import VertexAIGeminiChatGenerator

GET_CURRENT_WEATHER_FUNC = FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
            "unit": {
                "type": "string",
                "enum": [
                    "celsius",
                    "fahrenheit",
                ],
            },
        },
        "required": ["location"],
    },
)


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_init(mock_vertexai_init, _mock_generative_model):

    generation_config = GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

    tool = Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])
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
        tools=[tool],
        tool_config=tool_config,
        system_instruction="Please provide brief answers.",
    )
    mock_vertexai_init.assert_called()
    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config == generation_config
    assert gemini._safety_settings == safety_settings
    assert gemini._tools == [tool]
    assert gemini._tool_config == tool_config
    assert gemini._system_instruction == "Please provide brief answers."


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_to_dict(_mock_vertexai_init, _mock_generative_model):

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
            "system_instruction": None,
        },
    }


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_to_dict_with_params(_mock_vertexai_init, _mock_generative_model):
    generation_config = GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=2,
    )
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

    tool = Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])
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
        tools=[tool],
        tool_config=tool_config,
        system_instruction="Please provide brief answers.",
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
                    "function_declarations": [
                        {
                            "name": "get_current_weather",
                            "description": "Get the current weather in a given location",
                            "parameters": {
                                "type_": "OBJECT",
                                "properties": {
                                    "location": {
                                        "type_": "STRING",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {"type_": "STRING", "enum": ["celsius", "fahrenheit"]},
                                },
                                "required": ["location"],
                            },
                        }
                    ]
                }
            ],
            "tool_config": {
                "function_calling_config": {
                    "mode": ToolConfig.FunctionCallingConfig.Mode.ANY,
                    "allowed_function_names": ["get_current_weather_func"],
                }
            },
            "system_instruction": "Please provide brief answers.",
        },
    }


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_from_dict(_mock_vertexai_init, _mock_generative_model):
    gemini = VertexAIGeminiChatGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.chat.gemini.VertexAIGeminiChatGenerator",
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
    assert gemini._system_instruction is None
    assert gemini._generation_config is None


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_from_dict_with_param(_mock_vertexai_init, _mock_generative_model):
    gemini = VertexAIGeminiChatGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.chat.gemini.VertexAIGeminiChatGenerator",
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
                "safety_settings": {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH},
                "tools": [
                    {
                        "function_declarations": [
                            {
                                "name": "get_current_weather",
                                "description": "Get the current weather in a given location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string",
                                            "description": "The city and state, e.g. San Francisco, CA",
                                        },
                                        "unit": {
                                            "type": "string",
                                            "enum": [
                                                "celsius",
                                                "fahrenheit",
                                            ],
                                        },
                                    },
                                    "required": ["location"],
                                },
                            }
                        ]
                    }
                ],
                "tool_config": {
                    "function_calling_config": {
                        "mode": ToolConfig.FunctionCallingConfig.Mode.ANY,
                        "allowed_function_names": ["get_current_weather_func"],
                    }
                },
                "system_instruction": "Please provide brief answers.",
                "streaming_callback": None,
            },
        }
    )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._project_id == "TestID123"
    assert gemini._location == "TestLocation"
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert repr(gemini._tools) == repr([Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])])
    assert isinstance(gemini._tool_config, ToolConfig)
    assert isinstance(gemini._generation_config, GenerationConfig)
    assert gemini._system_instruction == "Please provide brief answers."
    assert (
        gemini._tool_config._gapic_tool_config.function_calling_config.mode == ToolConfig.FunctionCallingConfig.Mode.ANY
    )


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_run(mock_generative_model):
    mock_model = Mock()
    mock_candidate = Mock(content=Content(parts=[Part.from_text("This is a generated response.")], role="model"))
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
    assert len(response["replies"]) > 0
    assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_run_with_streaming_callback(mock_generative_model):
    mock_model = Mock()
    mock_responses = iter(
        [MagicMock(spec=GenerationResponse, text="First part"), MagicMock(spec=GenerationResponse, text="Second part")]
    )
    mock_model.send_message.return_value = mock_responses
    mock_model.start_chat.return_value = mock_model
    mock_generative_model.return_value = mock_model

    streaming_callback_called = []

    def streaming_callback(_chunk: StreamingChunk) -> None:
        nonlocal streaming_callback_called
        streaming_callback_called = True

    gemini = VertexAIGeminiChatGenerator(streaming_callback=streaming_callback)
    messages = [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France?"),
    ]
    response = gemini.run(messages=messages)
    mock_model.send_message.assert_called_once()
    assert "replies" in response


def test_serialization_deserialization_pipeline():

    pipeline = Pipeline()
    template = [ChatMessage.from_user("Translate to {{ target_language }}. Context: {{ snippet }}; Translation:")]
    pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
    pipeline.add_component("gemini", VertexAIGeminiChatGenerator(project_id="TestID123"))
    pipeline.connect("prompt_builder.prompt", "gemini.messages")

    pipeline_dict = pipeline.to_dict()

    new_pipeline = Pipeline.from_dict(pipeline_dict)
    assert new_pipeline == pipeline
