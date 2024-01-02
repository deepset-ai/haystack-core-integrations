import os
from unittest.mock import patch

from haystack.dataclasses.chat_message import ChatMessage
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from google.ai.generativelanguage import FunctionDeclaration, Tool
import pytest

from google_ai_haystack.generators.chat.gemini import GoogleAIGeminiChatGenerator


def test_init():
    generation_config = GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_ONLY_HIGH}
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
    with patch("google_ai_haystack.generators.chat.gemini.genai.configure") as mock_genai_configure:
        gemini = GoogleAIGeminiChatGenerator(
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=[tool],
        )
    mock_genai_configure.assert_called_once_with(api_key=None)
    assert gemini._model_name == "gemini-pro-vision"
    assert gemini._generation_config == generation_config
    assert gemini._safety_settings == safety_settings
    assert gemini._tools == [tool]
    assert isinstance(gemini._model, GenerativeModel)


def test_to_dict():
    generation_config = GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_ONLY_HIGH}
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

    with patch("google_ai_haystack.generators.chat.gemini.genai.configure"):
        gemini = GoogleAIGeminiChatGenerator(
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=[tool],
        )
    assert gemini.to_dict() == {
        "type": "google_ai_haystack.generators.chat.gemini.GoogleAIGeminiChatGenerator",
        "init_parameters": {
            "model": "gemini-pro-vision",
            "generation_config": {
                "temperature": 0.5,
                "top_p": 0.5,
                "top_k": 0.5,
                "candidate_count": 1,
                "max_output_tokens": 10,
                "stop_sequences": ["stop"],
            },
            "safety_settings": {6: 3},
            "tools": [
                b"\n\xad\x01\n\x13get_current_weather\x12+Get the current weather in a given location\x1ai\x08\x06:\x1f\n\x04unit\x12\x17\x08\x01*\x07celsius*\nfahrenheit::\n\x08location\x12.\x08\x01\x1a*The city and state, e.g. San Francisco, CAB\x08location"
            ],
        },
    }


def test_from_dict():
    with patch("google_ai_haystack.generators.chat.gemini.genai.configure"):
        gemini = GoogleAIGeminiChatGenerator.from_dict(
            {
                "type": "google_ai_haystack.generators.chat.gemini.GoogleAIGeminiChatGenerator",
                "init_parameters": {
                    "model": "gemini-pro-vision",
                    "generation_config": {
                        "temperature": 0.5,
                        "top_p": 0.5,
                        "top_k": 0.5,
                        "candidate_count": 1,
                        "max_output_tokens": 10,
                        "stop_sequences": ["stop"],
                    },
                    "safety_settings": {6: 3},
                    "tools": [
                        b"\n\xad\x01\n\x13get_current_weather\x12+Get the current weather in a given location\x1ai\x08\x06:\x1f\n\x04unit\x12\x17\x08\x01*\x07celsius*\nfahrenheit::\n\x08location\x12.\x08\x01\x1a*The city and state, e.g. San Francisco, CAB\x08location"
                    ],
                },
            }
        )

    assert gemini._model_name == "gemini-pro-vision"
    assert gemini._generation_config == GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert gemini._tools == [
        Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="get_current_weather",
                    description="Get the current weather in a given location",
                    parameters={
                        "type_": "OBJECT",
                        "properties": {
                            "location": {
                                "type_": "STRING",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
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
            ]
        )
    ]
    assert isinstance(gemini._model, GenerativeModel)


@pytest.mark.skipif("GOOGLE_API_KEY" not in os.environ, reason="GOOGLE_API_KEY not set")
def test_run():
    def get_current_weather(location: str, unit: str = "celsius"):
        return {"weather": "sunny", "temperature": 21.8, "unit": unit}

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
    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro", tools=[tool])
    messages = [ChatMessage.from_user(content="What is the temperature in celsius in Berlin?")]
    res = gemini_chat.run(messages=messages)
    assert len(res["replies"]) > 0

    weather = get_current_weather(**res["replies"][0].content)
    messages += res["replies"] + [ChatMessage.from_function(content=weather, name="get_current_weather")]

    res = gemini_chat.run(messages=messages)
    assert len(res["replies"]) > 0
