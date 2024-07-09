import os
from unittest.mock import patch

import pytest
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import FunctionDeclaration, HarmBlockThreshold, HarmCategory, Tool
from haystack.dataclasses.chat_message import ChatMessage

from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator


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
    with patch(
        "haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"
    ) as mock_genai_configure:
        gemini = GoogleAIGeminiChatGenerator(
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=[tool],
        )
    mock_genai_configure.assert_called_once_with(api_key="test")
    assert gemini._model_name == "gemini-pro-vision"
    assert gemini._generation_config == generation_config
    assert gemini._safety_settings == safety_settings
    assert gemini._tools == [tool]
    assert isinstance(gemini._model, GenerativeModel)


def test_to_dict(monkeypatch):
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

    with patch("haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"):
        gemini = GoogleAIGeminiChatGenerator(
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=[tool],
        )
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator",
        "init_parameters": {
            "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
            "model": "gemini-pro-vision",
            "generation_config": {
                "temperature": 0.5,
                "top_p": 0.5,
                "top_k": 2,
                "candidate_count": 1,
                "max_output_tokens": 10,
                "stop_sequences": ["stop"],
            },
            "safety_settings": {10: 3},
            "tools": [
                b"\n\xad\x01\n\x13get_current_weather\x12+Get the current weather in a given location\x1ai"
                b"\x08\x06:\x1f\n\x04unit\x12\x17\x08\x01*\x07celsius*\nfahrenheit::\n\x08location\x12.\x08"
                b"\x01\x1a*The city and state, e.g. San Francisco, CAB\x08location"
            ],
        },
    }


def test_from_dict(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"):
        gemini = GoogleAIGeminiChatGenerator.from_dict(
            {
                "type": "haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator",
                "init_parameters": {
                    "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
                    "model": "gemini-pro-vision",
                    "generation_config": {
                        "temperature": 0.5,
                        "top_p": 0.5,
                        "top_k": 2,
                        "candidate_count": 1,
                        "max_output_tokens": 10,
                        "stop_sequences": ["stop"],
                    },
                    "safety_settings": {10: 3},
                    "tools": [
                        b"\n\xad\x01\n\x13get_current_weather\x12+Get the current weather in a given location\x1ai"
                        b"\x08\x06:\x1f\n\x04unit\x12\x17\x08\x01*\x07celsius*\nfahrenheit::\n\x08location\x12.\x08"
                        b"\x01\x1a*The city and state, e.g. San Francisco, CAB\x08location"
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
        top_k=2,
    )
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert len(gemini._tools) == 1
    assert len(gemini._tools[0].function_declarations) == 1
    assert gemini._tools[0].function_declarations[0].name == "get_current_weather"
    assert gemini._tools[0].function_declarations[0].description == "Get the current weather in a given location"
    assert (
        gemini._tools[0].function_declarations[0].parameters.properties["location"].description
        == "The city and state, e.g. San Francisco, CA"
    )
    assert gemini._tools[0].function_declarations[0].parameters.properties["unit"].enum == ["celsius", "fahrenheit"]
    assert gemini._tools[0].function_declarations[0].parameters.required == ["location"]
    assert isinstance(gemini._model, GenerativeModel)


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
def test_run():
    # We're ignoring the unused function argument check since we must have that argument for the test
    # to run successfully, but we don't actually use it.
    def get_current_weather(location: str, unit: str = "celsius"):  # noqa: ARG001
        return {"weather": "sunny", "temperature": 21.8, "unit": unit}

    get_current_weather_func = FunctionDeclaration.from_function(
        get_current_weather,
        descriptions={
            "location": "The city and state, e.g. San Francisco, CA",
            "unit": "The temperature unit of measurement, e.g. celsius or fahrenheit",
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


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
def test_past_conversation():
    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro")
    messages = [
        ChatMessage.from_user(content="What is 2+2?"),
        ChatMessage.from_system(content="It's an arithmetic operation."),
        ChatMessage.from_user(content="Yeah, but what's the result?"),
    ]
    res = gemini_chat.run(messages=messages)
    assert len(res["replies"]) > 0
