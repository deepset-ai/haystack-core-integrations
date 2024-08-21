from unittest.mock import MagicMock, Mock, patch

from vertexai.preview.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    Tool,
)

from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator

GET_CURRENT_WEATHER_FUNC = FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type_": "OBJECT",
        "properties": {
            "location": {"type_": "STRING", "description": "The city and state, e.g. San Francisco, CA"},
            "unit": {
                "type_": "STRING",
                "enum": [
                    "celsius",
                    "fahrenheit",
                ],
            },
        },
        "required": ["location"],
    },
)

@patch("haystack_integrations.components.generators.google_vertex.gemini.vertexai.init")
@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
def test_init(_mock_vertexai_init, mock_generative_model):

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

    gemini = VertexAIGeminiGenerator(
        project_id="TestID123",
        location="TestLocation",
        generation_config=generation_config,
        safety_settings=safety_settings,
        tools=[tool],
    )
    _mock_vertexai_init.assert_called()
    assert gemini._model_name == "gemini-pro-vision"
    assert gemini._generation_config == generation_config
    assert gemini._safety_settings == safety_settings
    assert gemini._tools == [tool]

@patch("haystack_integrations.components.generators.google_vertex.gemini.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")

def test_to_dict(_mock_vertexai, _mock_generative_model):
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

    gemini = VertexAIGeminiGenerator(
        project_id="TestID123",
        generation_config=generation_config,
        safety_settings=safety_settings,
        tools=[tool],
    )
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_vertex.gemini.VertexAIGeminiGenerator",
        "init_parameters": {
            "model": "gemini-pro-vision",
            "project_id": "TestID123",
            "location": None,
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
        },
    }


@patch("haystack_integrations.components.generators.google_vertex.gemini.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")

def test_from_dict(_mock_vertexai, _mock_generative_model):
    gemini = VertexAIGeminiGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.gemini.VertexAIGeminiGenerator",
            "init_parameters": {
                "project_id": "TestID123",
                "model": "gemini-pro-vision",
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
                "streaming_callback": None,
            },
        }
    )

    assert gemini._model_name == "gemini-pro-vision"
    assert gemini._project_id == "TestID123"
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

    # assert gemini._tools == [Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])]
    assert isinstance(gemini._generation_config, GenerationConfig)


@patch("haystack_integrations.components.generators.google_vertex.gemini.GenerativeModel")
def test_run(mock_generative_model):
    mock_model = Mock()
    mock_model.generate_content.return_value = MagicMock()
    mock_generative_model.return_value = mock_model

    gemini = VertexAIGeminiGenerator(project_id="TestID123", location=None)

    response = gemini.run(["What's the weather like today?"])

    mock_model.generate_content.assert_called_once()
    assert "replies" in response
    assert isinstance(response["replies"], list)


