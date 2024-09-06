import os
from unittest.mock import patch

import pytest
from google.ai.generativelanguage import FunctionDeclaration, Tool
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from haystack.dataclasses import StreamingChunk

from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

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


def test_init(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    generation_config = GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    get_current_weather_func = FunctionDeclaration(
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

    tool = Tool(function_declarations=[get_current_weather_func])
    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure") as mock_genai_configure:
        gemini = GoogleAIGeminiGenerator(
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=[tool],
        )
    mock_genai_configure.assert_called_once_with(api_key="test")
    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config == generation_config
    assert gemini._safety_settings == safety_settings
    assert gemini._tools == [tool]
    assert isinstance(gemini._model, GenerativeModel)


def test_to_dict(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure"):
        gemini = GoogleAIGeminiGenerator()
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator",
        "init_parameters": {
            "model": "gemini-1.5-flash",
            "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
            "generation_config": None,
            "safety_settings": None,
            "streaming_callback": None,
            "tools": None,
        },
    }


def test_to_dict_with_param(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    generation_config = GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=2,
    )
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    get_current_weather_func = FunctionDeclaration(
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

    tool = Tool(function_declarations=[get_current_weather_func])

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure"):
        gemini = GoogleAIGeminiGenerator(
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=[tool],
        )
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator",
        "init_parameters": {
            "model": "gemini-1.5-flash",
            "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
            "generation_config": {
                "temperature": 0.5,
                "top_p": 0.5,
                "top_k": 2,
                "candidate_count": 1,
                "max_output_tokens": 10,
                "stop_sequences": ["stop"],
            },
            "safety_settings": {10: 3},
            "streaming_callback": None,
            "tools": [
                b"\n\xad\x01\n\x13get_current_weather\x12+Get the current weather in a given location\x1ai"
                b"\x08\x06:\x1f\n\x04unit\x12\x17\x08\x01*\x07celsius*\nfahrenheit::\n\x08location\x12.\x08"
                b"\x01\x1a*The city and state, e.g. San Francisco, CAB\x08location"
            ],
        },
    }


def test_from_dict_with_param(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure"):
        gemini = GoogleAIGeminiGenerator.from_dict(
            {
                "type": "haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator",
                "init_parameters": {
                    "model": "gemini-1.5-flash",
                    "generation_config": {
                        "temperature": 0.5,
                        "top_p": 0.5,
                        "top_k": 0.5,
                        "candidate_count": 1,
                        "max_output_tokens": 10,
                        "stop_sequences": ["stop"],
                    },
                    "safety_settings": {10: 3},
                    "streaming_callback": None,
                    "tools": [
                        b"\n\xad\x01\n\x13get_current_weather\x12+Get the current weather in a given location\x1ai"
                        b"\x08\x06:\x1f\n\x04unit\x12\x17\x08\x01*\x07celsius*\nfahrenheit::\n\x08location\x12.\x08"
                        b"\x01\x1a*The city and state, e.g. San Francisco, CAB\x08location"
                    ],
                },
            }
        )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config == GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert gemini._tools == [Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])]
    assert isinstance(gemini._model, GenerativeModel)


def test_from_dict(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure"):
        gemini = GoogleAIGeminiGenerator.from_dict(
            {
                "type": "haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator",
                "init_parameters": {
                    "model": "gemini-1.5-flash",
                    "generation_config": {
                        "temperature": 0.5,
                        "top_p": 0.5,
                        "top_k": 0.5,
                        "candidate_count": 1,
                        "max_output_tokens": 10,
                        "stop_sequences": ["stop"],
                    },
                    "safety_settings": {10: 3},
                    "streaming_callback": None,
                    "tools": [
                        b"\n\xad\x01\n\x13get_current_weather\x12+Get the current weather in a given location\x1ai"
                        b"\x08\x06:\x1f\n\x04unit\x12\x17\x08\x01*\x07celsius*\nfahrenheit::\n\x08location\x12.\x08"
                        b"\x01\x1a*The city and state, e.g. San Francisco, CAB\x08location"
                    ],
                },
            }
        )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config == GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert gemini._tools == [Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])]
    assert isinstance(gemini._model, GenerativeModel)


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
def test_run():
    gemini = GoogleAIGeminiGenerator(model="gemini-pro")
    res = gemini.run("Tell me something cool")
    assert len(res["replies"]) > 0


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
def test_run_with_streaming_callback():
    streaming_callback_called = False

    def streaming_callback(_chunk: StreamingChunk) -> None:
        nonlocal streaming_callback_called
        streaming_callback_called = True

    gemini = GoogleAIGeminiGenerator(model="gemini-pro", streaming_callback=streaming_callback)
    res = gemini.run("Tell me something cool")
    assert len(res["replies"]) > 0
    assert streaming_callback_called
