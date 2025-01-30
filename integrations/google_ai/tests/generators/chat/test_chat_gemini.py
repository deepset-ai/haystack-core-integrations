import json
import os
from typing import Annotated, Literal
from unittest.mock import patch

import google.generativeai as genai
import pytest
from google.ai.generativelanguage import Part
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from haystack import Pipeline
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.chat_message import ChatMessage, ChatRole, TextContent, ToolCall
from haystack.tools import Tool, create_tool_from_function

from haystack_integrations.components.generators.google_ai.chat.gemini import (
    GoogleAIGeminiChatGenerator,
    _convert_chatmessage_to_google_content,
)

TYPE = "haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator"


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
    assert google_content.parts == [Part(text="Hello, how are you?")]
    assert google_content.role == "model"

    message = ChatMessage.from_user("I have a question")
    google_content = _convert_chatmessage_to_google_content(message)
    assert google_content.parts == [Part(text="I have a question")]
    assert google_content.role == "user"

    message = ChatMessage.from_assistant(
        tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})]
    )
    google_content = _convert_chatmessage_to_google_content(message)
    assert google_content.parts == [
        Part(function_call=genai.protos.FunctionCall(name="weather", args={"city": "Paris"}))
    ]
    assert google_content.role == "model"

    tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
    message = ChatMessage.from_tool(
        tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
    )
    google_content = _convert_chatmessage_to_google_content(message)
    assert google_content.parts == [
        Part(function_response=genai.protos.FunctionResponse(name="weather", response={"result": tool_result}))
    ]
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


class TestGoogleAIGeminiChatGenerator:
    def test_init(self, tools, monkeypatch):
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
        with patch(
            "haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"
        ) as mock_genai_configure:
            gemini = GoogleAIGeminiChatGenerator(
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
            )
        mock_genai_configure.assert_called_once_with(api_key="test")
        assert gemini._model_name == "gemini-1.5-flash"
        assert gemini._generation_config == generation_config
        assert gemini._safety_settings == safety_settings
        assert gemini._tools == tools
        assert isinstance(gemini._model, GenerativeModel)

    def test_to_dict(self, monkeypatch):
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
                "tool_config": None,
            },
        }

    def test_to_dict_with_param(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test")
        tools = [Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)]

        tool_config = {
            "function_calling_config": {
                "mode": "any",
                "allowed_function_names": ["name"],
            },
        }

        generation_config = GenerationConfig(
            candidate_count=1,
            stop_sequences=["stop"],
            max_output_tokens=10,
            temperature=0.5,
            top_p=0.5,
            top_k=2,
        )
        safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

        with patch("haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"):
            gemini = GoogleAIGeminiChatGenerator(
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                tool_config=tool_config,
            )
        assert gemini.to_dict() == {
            "type": TYPE,
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
                        "mode": "any",
                        "allowed_function_names": ["name"],
                    },
                },
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test")

        with patch("haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"):
            gemini = GoogleAIGeminiChatGenerator.from_dict(
                {
                    "type": TYPE,
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

    def test_from_dict_with_param(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test")

        with patch("haystack_integrations.components.generators.google_ai.chat.gemini.genai.configure"):
            gemini = GoogleAIGeminiChatGenerator.from_dict(
                {
                    "type": TYPE,
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
        assert gemini._safety_settings == {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
        }
        assert len(gemini._tools) == 1
        assert gemini._tools[0].name == "name"
        assert gemini._tools[0].description == "description"
        assert gemini._tools[0].parameters == {"x": {"type": "string"}}
        assert gemini._tools[0].function == print
        assert isinstance(gemini._model, GenerativeModel)

    def test_serde_in_pipeline(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test")
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = GoogleAIGeminiChatGenerator(
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
                    "type": TYPE,
                    "init_parameters": {
                        "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
                        "model": "gemini-1.5-flash",
                        "generation_config": {
                            "temperature": 0.6,
                            "stop_sequences": ["stop", "words"],
                        },
                        "safety_settings": None,
                        "streaming_callback": None,
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
                        "tool_config": None,
                    },
                }
            },
            "connections": [],
        }

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

    def test_convert_to_google_tool(self, tools):
        tool = tools[0]
        google_tool = GoogleAIGeminiChatGenerator._convert_to_google_tool(tool)

        assert google_tool.name == tool.name
        assert google_tool.description == tool.description

        assert google_tool.parameters
        # check if default values are removed. This type is not easily inspectable
        assert "default" not in str(google_tool.parameters)

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
    def test_run(self):
        gemini_chat = GoogleAIGeminiChatGenerator()
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        response = gemini_chat.run(messages=chat_messages)
        assert "replies" in response
        assert len(response["replies"]) > 0

        reply = response["replies"][0]
        assert reply.role == ChatRole.ASSISTANT
        assert "paris" in reply.text.lower()

        assert not reply.tool_calls
        assert not reply.tool_call_results

        assert "usage" in reply.meta
        assert "prompt_tokens" in reply.meta["usage"]
        assert "completion_tokens" in reply.meta["usage"]
        assert "total_tokens" in reply.meta["usage"]

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
    def test_run_with_tools(self, tools):

        gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-2.0-flash-exp", tools=tools)
        user_message = [ChatMessage.from_user("What is the temperature in celsius in Berlin?")]
        response = gemini_chat.run(messages=user_message)
        assert "replies" in response
        assert len(response["replies"]) > 0
        assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])

        # check the first response contains a tool call
        chat_message = response["replies"][0]
        assert chat_message.tool_calls
        assert chat_message.tool_calls[0].tool_name == "get_current_weather"
        assert chat_message.tool_calls[0].arguments == {"city": "Berlin", "unit": "Celsius"}

        weather = tools[0].invoke(**chat_message.tool_calls[0].arguments)

        messages = (
            user_message
            + response["replies"]
            + [ChatMessage.from_tool(tool_result=weather, origin=chat_message.tool_calls[0])]
        )

        response = gemini_chat.run(messages=messages)
        assert "replies" in response
        assert len(response["replies"]) > 0
        assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])

        # check the second response is not a tool call
        chat_message = response["replies"][0]
        assert not chat_message.tool_calls
        assert chat_message.text
        assert "berlin" in chat_message.text.lower()

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
    def test_run_with_tools_and_tool_config(self, tools):

        def get_population(city: Annotated[str, "the city for which to get the population, e.g. 'Munich'"] = "Munich"):
            """A simple function to get the population for a location."""
            return f"Population of {city}: 1,000,000"

        multiple_tools = [tools[0], create_tool_from_function(get_population)]

        tool_config = {
            "function_calling_config": {
                "mode": "any",
                "allowed_function_names": ["get_current_weather"],
            },
        }

        gemini_chat = GoogleAIGeminiChatGenerator(
            model="gemini-2.0-flash-exp", tools=multiple_tools, tool_config=tool_config
        )
        user_message = [
            ChatMessage.from_user("What is the temperature in celsius in Berlin and how many people live there?")
        ]
        response = gemini_chat.run(messages=user_message)
        assert "replies" in response
        assert len(response["replies"]) > 0
        assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])

        # check the only the allowed function is called
        chat_message = response["replies"][0]
        assert chat_message.tool_calls
        assert len(chat_message.tool_calls) == 1
        assert chat_message.tool_calls[0].tool_name == "get_current_weather"
        assert chat_message.tool_calls[0].arguments == {"city": "Berlin", "unit": "Celsius"}

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
    def test_run_with_streaming_callback_and_tools(self, tools):
        streaming_callback_called = False

        def streaming_callback(_chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        gemini_chat = GoogleAIGeminiChatGenerator(
            model="gemini-2.0-flash-exp", tools=tools, streaming_callback=streaming_callback
        )
        messages = [ChatMessage.from_user("What is the temperature in celsius in Berlin?")]
        response = gemini_chat.run(messages=messages)
        assert "replies" in response
        assert len(response["replies"]) > 0
        assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])
        assert streaming_callback_called

        # check the first response contains a tool call
        chat_message = response["replies"][0]
        assert chat_message.tool_calls
        assert chat_message.tool_calls[0].tool_name == "get_current_weather"
        assert chat_message.tool_calls[0].arguments == {"city": "Berlin", "unit": "Celsius"}
        assert "usage" in chat_message.meta
        assert "prompt_tokens" in chat_message.meta["usage"]
        assert "completion_tokens" in chat_message.meta["usage"]
        assert "total_tokens" in chat_message.meta["usage"]

        weather = tools[0].invoke(**chat_message.tool_calls[0].arguments)
        messages += response["replies"] + [
            ChatMessage.from_tool(tool_result=weather, origin=chat_message.tool_calls[0])
        ]
        response = gemini_chat.run(messages=messages)
        assert "replies" in response
        assert len(response["replies"]) > 0
        assert all(reply.role == ChatRole.ASSISTANT for reply in response["replies"])

        # check the second response is not a tool call
        chat_message = response["replies"][0]
        assert not chat_message.tool_calls
        assert chat_message.text
        assert "berlin" in chat_message.text.lower()
        assert "usage" in chat_message.meta
        assert "prompt_tokens" in chat_message.meta["usage"]
        assert "completion_tokens" in chat_message.meta["usage"]
        assert "total_tokens" in chat_message.meta["usage"]

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
    def test_past_conversation(self):
        gemini_chat = GoogleAIGeminiChatGenerator()
        messages = [
            ChatMessage.from_system("You are a knowledageable mathematician."),
            ChatMessage.from_user("What is 2+2?"),
            ChatMessage.from_assistant("It's an arithmetic operation."),
            ChatMessage.from_user("Yeah, but what's the result?"),
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
