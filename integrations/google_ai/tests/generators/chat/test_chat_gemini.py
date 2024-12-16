import json
import os
from unittest.mock import patch

import pytest
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import FunctionDeclaration, HarmBlockThreshold, HarmCategory, Tool
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.chat_message import ChatMessage, ChatRole

from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator

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
    tool = Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])
    with patch(
        "haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"
    ) as mock_genai_configure:
        gemini = GoogleAIGeminiChatGenerator(
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

    with patch("haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"):
        gemini = GoogleAIGeminiChatGenerator()
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator",
        "init_parameters": {
            "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
            "model": "gemini-1.5-flash",
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
    tool = Tool(function_declarations=[GET_CURRENT_WEATHER_FUNC])

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
            "model": "gemini-1.5-flash",
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


def test_from_dict(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"):
        gemini = GoogleAIGeminiChatGenerator.from_dict(
            {
                "type": "haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator",
                "init_parameters": {
                    "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
                    "model": "gemini-1.5-flash",
                    "generation_config": None,
                    "safety_settings": None,
                    "streaming_callback": None,
                    "tools": None,
                },
            }
        )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config is None
    assert gemini._safety_settings is None
    assert gemini._tools is None
    assert isinstance(gemini._model, GenerativeModel)


def test_from_dict_with_param(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"):
        gemini = GoogleAIGeminiChatGenerator.from_dict(
            {
                "type": "haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator",
                "init_parameters": {
                    "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
                    "model": "gemini-1.5-flash",
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
        )

    assert gemini._model_name == "gemini-1.5-flash"
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
            "location": "The city, e.g. San Francisco",
            "unit": "The temperature unit of measurement, e.g. celsius or fahrenheit",
        },
    )

    tool = Tool(function_declarations=[get_current_weather_func])
    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro", tools=[tool])
    messages = [ChatMessage.from_user(content="What is the temperature in celsius in Berlin?")]
    response = gemini_chat.run(messages=messages)
    assert "replies" in response
    assert len(response["replies"]) > 0
    assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])

    # check the first response is a function call
    chat_message = response["replies"][0]
    assert "function_call" in chat_message.meta
    assert json.loads(chat_message.text) == {"location": "Berlin", "unit": "celsius"}

    weather = get_current_weather(**json.loads(chat_message.text))
    messages += response["replies"] + [ChatMessage.from_function(content=weather, name="get_current_weather")]
    response = gemini_chat.run(messages=messages)
    assert "replies" in response
    assert len(response["replies"]) > 0
    assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])

    # check the second response is not a function call
    chat_message = response["replies"][0]
    assert "function_call" not in chat_message.meta
    assert isinstance(chat_message.text, str)


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
def test_run_with_streaming_callback():
    streaming_callback_called = False

    def streaming_callback(_chunk: StreamingChunk) -> None:
        nonlocal streaming_callback_called
        streaming_callback_called = True

    def get_current_weather(location: str, unit: str = "celsius"):  # noqa: ARG001
        return {"weather": "sunny", "temperature": 21.8, "unit": unit}

    get_current_weather_func = FunctionDeclaration.from_function(
        get_current_weather,
        descriptions={
            "location": "The city, e.g. San Francisco",
            "unit": "The temperature unit of measurement, e.g. celsius or fahrenheit",
        },
    )

    tool = Tool(function_declarations=[get_current_weather_func])
    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro", tools=[tool], streaming_callback=streaming_callback)
    messages = [ChatMessage.from_user(content="What is the temperature in celsius in Berlin?")]
    response = gemini_chat.run(messages=messages)
    assert "replies" in response
    assert len(response["replies"]) > 0
    assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])
    assert streaming_callback_called

    # check the first response is a function call
    chat_message = response["replies"][0]
    assert "function_call" in chat_message.meta
    assert json.loads(chat_message.text) == {"location": "Berlin", "unit": "celsius"}

    weather = get_current_weather(**json.loads(response["replies"][0].text))
    messages += response["replies"] + [ChatMessage.from_function(content=weather, name="get_current_weather")]
    response = gemini_chat.run(messages=messages)
    assert "replies" in response
    assert len(response["replies"]) > 0
    assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])

    # check the second response is not a function call
    chat_message = response["replies"][0]
    assert "function_call" not in chat_message.meta
    assert isinstance(chat_message.text, str)


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
def test_past_conversation():
    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro")
    messages = [
        ChatMessage.from_system(content="You are a knowledageable mathematician."),
        ChatMessage.from_user(content="What is 2+2?"),
        ChatMessage.from_assistant(content="It's an arithmetic operation."),
        ChatMessage.from_user(content="Yeah, but what's the result?"),
    ]
    response = gemini_chat.run(messages=messages)
    assert "replies" in response
    replies = response["replies"]
    assert len(replies) > 0
    assert all(reply.role == ChatRole.ASSISTANT for reply in replies)

    assert all("usage" in reply.meta for reply in replies)
    assert all("prompt_tokens" in reply.meta["usage"] for reply in replies)
    assert all("completion_tokens" in reply.meta["usage"] for reply in replies)
    assert all("total_tokens" in reply.meta["usage"] for reply in replies)
