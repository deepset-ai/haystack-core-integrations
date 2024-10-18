from unittest.mock import MagicMock, Mock, patch

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import StreamingChunk
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    Tool,
    ToolConfig,
)

from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator

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


@patch("haystack_integrations.components.generators.google_vertex.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
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

    gemini = VertexAIGeminiGenerator(
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


@patch("haystack_integrations.components.generators.google_vertex.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
def test_to_dict(_mock_vertexai_init, _mock_generative_model):

    gemini = VertexAIGeminiGenerator()
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_vertex.gemini.VertexAIGeminiGenerator",
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


@patch("haystack_integrations.components.generators.google_vertex.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
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

    gemini = VertexAIGeminiGenerator(
        project_id="TestID123",
        location="TestLocation",
        generation_config=generation_config,
        safety_settings=safety_settings,
        tools=[tool],
        tool_config=tool_config,
        system_instruction="Please provide brief answers.",
    )
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_vertex.gemini.VertexAIGeminiGenerator",
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


@patch("haystack_integrations.components.generators.google_vertex.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
def test_from_dict(_mock_vertexai_init, _mock_generative_model):
    gemini = VertexAIGeminiGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.gemini.VertexAIGeminiGenerator",
            "init_parameters": {
                "project_id": None,
                "location": None,
                "model": "gemini-1.5-flash",
                "generation_config": None,
                "safety_settings": None,
                "tools": None,
                "streaming_callback": None,
                "tool_config": None,
                "system_instruction": None,
            },
        }
    )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._project_id is None
    assert gemini._location is None
    assert gemini._safety_settings is None
    assert gemini._tools is None
    assert gemini._tool_config is None
    assert gemini._system_instruction is None
    assert gemini._generation_config is None


@patch("haystack_integrations.components.generators.google_vertex.gemini.vertexai_init")
@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
def test_from_dict_with_param(_mock_vertexai_init, _mock_generative_model):
    gemini = VertexAIGeminiGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.gemini.VertexAIGeminiGenerator",
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
                                "description": "Get the current weather in a given location",
                            }
                        ]
                    }
                ],
                "streaming_callback": None,
                "tool_config": {
                    "function_calling_config": {
                        "mode": ToolConfig.FunctionCallingConfig.Mode.ANY,
                        "allowed_function_names": ["get_current_weather_func"],
                    }
                },
                "system_instruction": "Please provide brief answers.",
            },
        }
    )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._project_id == "TestID123"
    assert gemini._location == "TestLocation"
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert repr(gemini._tools) == repr([Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])])
    assert isinstance(gemini._generation_config, GenerationConfig)
    assert isinstance(gemini._tool_config, ToolConfig)
    assert gemini._system_instruction == "Please provide brief answers."
    assert (
        gemini._tool_config._gapic_tool_config.function_calling_config.mode == ToolConfig.FunctionCallingConfig.Mode.ANY
    )


@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
def test_run(mock_generative_model):
    mock_model = Mock()
    mock_model.generate_content.return_value = MagicMock()
    mock_generative_model.return_value = mock_model

    gemini = VertexAIGeminiGenerator()

    response = gemini.run(["What's the weather like today?"])

    mock_model.generate_content.assert_called_once()
    assert "replies" in response
    assert isinstance(response["replies"], list)


@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
def test_run_with_streaming_callback(mock_generative_model):
    mock_model = Mock()
    mock_stream = [
        MagicMock(text="First part", usage_metadata={}),
        MagicMock(text="Second part", usage_metadata={}),
    ]

    mock_model.generate_content.return_value = mock_stream
    mock_generative_model.return_value = mock_model

    streaming_callback_called = False

    def streaming_callback(_chunk: StreamingChunk) -> None:
        nonlocal streaming_callback_called
        streaming_callback_called = True

    gemini = VertexAIGeminiGenerator(model="gemini-pro", streaming_callback=streaming_callback)
    gemini.run(["Come on, stream!"])
    assert streaming_callback_called


def test_serialization_deserialization_pipeline():
    template = """
        Answer the following questions:
        1. What is the weather like today?
        """
    pipeline = Pipeline()

    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("gemini", VertexAIGeminiGenerator(project_id="TestID123"))
    pipeline.connect("prompt_builder", "gemini")

    pipeline_dict = pipeline.to_dict()

    new_pipeline = Pipeline.from_dict(pipeline_dict)
    assert new_pipeline == pipeline
