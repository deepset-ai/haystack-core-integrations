import json
from typing import Annotated, Literal
from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
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


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_init(mock_vertexai_init, _mock_generative_model, tools):

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
        },
    }


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_to_dict_with_params(_mock_vertexai_init, _mock_generative_model):
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
    assert gemini._generation_config is None


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_from_dict_with_param(_mock_vertexai_init, _mock_generative_model):
    tools = [Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)]

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
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert gemini._tools == tools
    assert isinstance(gemini._tool_config, ToolConfig)
    assert isinstance(gemini._generation_config, GenerationConfig)
    assert (
        gemini._tool_config._gapic_tool_config.function_calling_config.mode == ToolConfig.FunctionCallingConfig.Mode.ANY
    )


@patch("haystack_integrations.components.generators.google_vertex.chat.gemini.GenerativeModel")
def test_run(mock_generative_model):
    mock_model = Mock()
    mock_candidate = MagicMock(content=Content(parts=[Part.from_text("This is a generated response.")], role="model"))
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
