from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import StreamingChunk
from vertexai.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
)

from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator


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

    gemini = VertexAIGeminiGenerator(
        project_id="TestID123",
        location="TestLocation",
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction="Please provide brief answers.",
    )
    mock_vertexai_init.assert_called()
    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config == generation_config
    assert gemini._safety_settings == safety_settings
    assert gemini._system_instruction == "Please provide brief answers."


def test_init_fails_with_tools_or_tool_config():
    with pytest.raises(TypeError, match="VertexAIGeminiGenerator does not support `tools`"):
        VertexAIGeminiGenerator(tools=["tool1", "tool2"])

    with pytest.raises(TypeError, match="VertexAIGeminiGenerator does not support `tools`"):
        VertexAIGeminiGenerator(tool_config={"custom": "config"})


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

    gemini = VertexAIGeminiGenerator(
        project_id="TestID123",
        location="TestLocation",
        generation_config=generation_config,
        safety_settings=safety_settings,
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
                "streaming_callback": None,
                "system_instruction": None,
            },
        }
    )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._project_id is None
    assert gemini._location is None
    assert gemini._safety_settings is None
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
                "streaming_callback": None,
                "system_instruction": "Please provide brief answers.",
            },
        }
    )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._project_id == "TestID123"
    assert gemini._location == "TestLocation"
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert isinstance(gemini._generation_config, GenerationConfig)
    assert gemini._system_instruction == "Please provide brief answers."


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
