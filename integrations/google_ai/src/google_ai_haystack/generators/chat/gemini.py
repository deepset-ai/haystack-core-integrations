import logging
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
from google.ai.generativelanguage import Content, Part, Tool
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.chat_message import ChatMessage, ChatRole

logger = logging.getLogger(__name__)


@component
class GoogleAIGeminiChatGenerator:
    """
    GoogleAIGeminiGenerator is a multi modal generator supporting Gemini via Google Makersuite.

    Sample usage:
    ```python
    from haystack.dataclasses.chat_message import ChatMessage
    from google_ai_haystack.generators.chat.gemini import GoogleAIGeminiChatGenerator


    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro", api_key="<MY_API_KEY>")

    messages = [ChatMessage.from_user("What is the most interesting thing you know?")]
    res = gemini_chat.run(messages=messages)
    for reply in res["replies"]:
        print(reply.content)

    messages += res["replies"] + [ChatMessage.from_user("Tell me more about it")]
    res = gemini_chat.run(messages=messages)
    for reply in res["replies"]:
        print(reply.content)
    ```


    This is a more advanced usage that also uses function calls:
    ```python
    from haystack.dataclasses.chat_message import ChatMessage
    from google.ai.generativelanguage import FunctionDeclaration, Tool

    from google_ai_haystack.generators.chat.gemini import GoogleAIGeminiChatGenerator

    # Example function to get the current weather
    def get_current_weather(location: str, unit: str = "celsius") -> str:
        # Call a weather API and return some text
        ...

    # Define the function interface so that Gemini can call it
    get_current_weather_func = FunctionDeclaration(
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
    tool = Tool([get_current_weather_func])

    messages = [ChatMessage.from_user("What is the most interesting thing you know?")]

    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro", api_key="<MY_API_KEY>", tools=[tool])

    messages = [ChatMessage.from_user(content = "What is the temperature in celsius in Berlin?")]
    res = gemini_chat.run(messages=messages)

    weather = get_current_weather(**res["replies"][0].content)
    messages += res["replies"] + [ChatMessage.from_function(content=weather, name="get_current_weather")]
    res = gemini_chat.run(messages=messages)
    for reply in res["replies"]:
        print(reply.content)
    ```

    Input:
    - **messages** A list of ChatMessage objects.

    Output:
    - **replies** A list of ChatMessage objects containing the one or more replies from the model.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-pro-vision",
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Initialize a GoogleAIGeminiChatGenerator instance.
        If `api_key` is `None` it will use the `GOOGLE_API_KEY` env variable for authentication.

        To get an API key, visit: https://makersuite.google.com

        It supports the following models:
        * `gemini-pro`
        * `gemini-pro-vision`
        * `gemini-ultra`

        :param api_key: Google Makersuite API key, defaults to None
        :param model: Name of the model to use, defaults to "gemini-pro-vision"
        :param generation_config: The generation config to use, defaults to None.
            Can either be a GenerationConfig object or a dictionary of parameters.
            Accepted fields are:
                - temperature
                - top_p
                - top_k
                - candidate_count
                - max_output_tokens
                - stop_sequences
        :param safety_settings: The safety settings to use, defaults to None.
            A dictionary of HarmCategory to HarmBlockThreshold.
        :param tools: The tools to use, defaults to None.
            A list of Tool objects that can be used to modify the generation process.
        """

        # Authenticate, if api_key is None it will use the GOOGLE_API_KEY env variable
        genai.configure(api_key=api_key)

        self._model_name = model
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._model = GenerativeModel(self._model_name, tools=self._tools)

    def _generation_config_to_dict(self, config: Union[GenerationConfig, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        return {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "candidate_count": config.candidate_count,
            "max_output_tokens": config.max_output_tokens,
            "stop_sequences": config.stop_sequences,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(
            self,
            model=self._model_name,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            tools=self._tools,
        )
        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = [Tool.serialize(t) for t in tools]
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = self._generation_config_to_dict(generation_config)
        if (safety_settings := data["init_parameters"].get("safety_settings")) is not None:
            data["init_parameters"]["safety_settings"] = {k.value: v.value for k, v in safety_settings.items()}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiChatGenerator":
        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = [Tool.deserialize(t) for t in tools]
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = GenerationConfig(**generation_config)
        if (safety_settings := data["init_parameters"].get("safety_settings")) is not None:
            data["init_parameters"]["safety_settings"] = {
                HarmCategory(k): HarmBlockThreshold(v) for k, v in safety_settings.items()
            }
        return default_from_dict(cls, data)

    def _convert_part(self, part: Union[str, ByteStream, Part]) -> Part:
        if isinstance(part, str):
            converted_part = Part()
            converted_part.text = part
            return converted_part
        elif isinstance(part, ByteStream):
            converted_part = Part()
            converted_part.inline_data.data = part.data
            converted_part.inline_data.mime_type = part.mime_type
            return converted_part
        elif isinstance(part, Part):
            return part
        else:
            msg = f"Unsupported type {type(part)} for part {part}"
            raise ValueError(msg)

    def _message_to_part(self, message: ChatMessage) -> Part:
        if message.role == ChatRole.SYSTEM and message.name:
            p = Part()
            p.function_call.name = message.name
            p.function_call.args = {}
            for k, v in message.content.items():
                p.function_call.args[k] = v
            return p
        elif message.role == ChatRole.SYSTEM:
            p = Part()
            p.text = message.content
            return p
        elif message.role == ChatRole.FUNCTION:
            p = Part()
            p.function_response.name = message.name
            p.function_response.response = message.content
            return p
        elif message.role == ChatRole.USER:
            return self._convert_part(message.content)

    def _message_to_content(self, message: ChatMessage) -> Content:
        if message.role == ChatRole.SYSTEM and message.name:
            part = Part()
            part.function_call.name = message.name
            part.function_call.args = {}
            for k, v in message.content.items():
                part.function_call.args[k] = v
        elif message.role == ChatRole.SYSTEM:
            part = Part()
            part.text = message.content
            return part
        elif message.role == ChatRole.FUNCTION:
            part = Part()
            part.function_response.name = message.name
            part.function_response.response = message.content
            return part
        elif message.role == ChatRole.USER:
            part = self._convert_part(message.content)
        else:
            msg = f"Unsupported message role {message.role}"
            raise ValueError(msg)
        role = "user" if message.role in [ChatRole.USER, ChatRole.FUNCTION] else "model"
        return Content(parts=[part], role=role)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        history = [self._message_to_content(m) for m in messages[:-1]]
        session = self._model.start_chat(history=history)

        new_message = self._message_to_part(messages[-1])
        res = session.send_message(
            content=new_message,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
        )

        replies = []
        for candidate in res.candidates:
            for part in candidate.content.parts:
                if part.text != "":
                    replies.append(ChatMessage.from_system(part.text))
                elif part.function_call is not None:
                    replies.append(
                        ChatMessage(
                            content=dict(part.function_call.args.items()),
                            role=ChatRole.SYSTEM,
                            name=part.function_call.name,
                        )
                    )

        return {"replies": replies}
