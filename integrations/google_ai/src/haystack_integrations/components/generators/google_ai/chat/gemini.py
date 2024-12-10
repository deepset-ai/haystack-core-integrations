import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import google.generativeai as genai
from google.ai.generativelanguage import Content, Part
from google.ai.generativelanguage import Tool as ToolProto
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import GenerateContentResponse, HarmBlockThreshold, HarmCategory, Tool
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream, StreamingChunk
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

logger = logging.getLogger(__name__)


@component
class GoogleAIGeminiChatGenerator:
    """
    Completes chats using multimodal Gemini models through Google AI Studio.

    It uses the [`ChatMessage`](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)
      dataclass to interact with the model.

    ### Usage example

    ```python
    from haystack.utils import Secret
    from haystack.dataclasses.chat_message import ChatMessage
    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator


    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro", api_key=Secret.from_token("<MY_API_KEY>"))

    messages = [ChatMessage.from_user("What is the most interesting thing you know?")]
    res = gemini_chat.run(messages=messages)
    for reply in res["replies"]:
        print(reply.text)

    messages += res["replies"] + [ChatMessage.from_user("Tell me more about it")]
    res = gemini_chat.run(messages=messages)
    for reply in res["replies"]:
        print(reply.text)
    ```


    #### With function calling:

    ```python
    from haystack.utils import Secret
    from haystack.dataclasses.chat_message import ChatMessage
    from google.ai.generativelanguage import FunctionDeclaration, Tool

    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator

    # Example function to get the current weather
    def get_current_weather(location: str, unit: str = "celsius") -> str:
        # Call a weather API and return some text
        ...

    # Define the function interface
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

    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-pro", api_key=Secret.from_token("<MY_API_KEY>"),
                                              tools=[tool])

    messages = [ChatMessage.from_user("What is the temperature in celsius in Berlin?")]
    res = gemini_chat.run(messages=messages)

    weather = get_current_weather(**json.loads(res["replies"][0].text))
    messages += res["replies"] + [ChatMessage.from_function(content=weather, name="get_current_weather")]
    res = gemini_chat.run(messages=messages)
    for reply in res["replies"]:
        print(reply.text)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("GOOGLE_API_KEY"),  # noqa: B008
        model: str = "gemini-1.5-flash",
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None,
        tools: Optional[List[Tool]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initializes a `GoogleAIGeminiChatGenerator` instance.

        To get an API key, visit: https://makersuite.google.com

        :param api_key: Google AI Studio API key. To get a key,
        see [Google AI Studio](https://makersuite.google.com).
        :param model: Name of the model to use. For available models, see https://ai.google.dev/gemini-api/docs/models/gemini.
        :param generation_config: The generation configuration to use.
            This can either be a `GenerationConfig` object or a dictionary of parameters.
            For available parameters, see
            [the `GenerationConfig` API reference](https://ai.google.dev/api/python/google/generativeai/GenerationConfig).
        :param safety_settings: The safety settings to use.
            A dictionary with `HarmCategory` as keys and `HarmBlockThreshold` as values.
            For more information, see [the API reference](https://ai.google.dev/api)
        :param tools: A list of Tool objects that can be used for [Function calling](https://ai.google.dev/docs/function_calling).
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        """

        genai.configure(api_key=api_key.resolve_value())

        self._api_key = api_key
        self._model_name = model
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._model = GenerativeModel(self._model_name, tools=self._tools)
        self._streaming_callback = streaming_callback

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
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        callback_name = serialize_callable(self._streaming_callback) if self._streaming_callback else None

        data = default_to_dict(
            self,
            api_key=self._api_key.to_dict(),
            model=self._model_name,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            tools=self._tools,
            streaming_callback=callback_name,
        )
        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = []
            for tool in tools:
                if isinstance(tool, Tool):
                    # There are multiple Tool types in the Google lib, one that is a protobuf class and
                    # another is a simple Python class. They have a similar structure but the Python class
                    # can't be easily serializated to a dict. We need to convert it to a protobuf class first.
                    tool = tool.to_proto()  # noqa: PLW2901
                data["init_parameters"]["tools"].append(ToolProto.serialize(tool))
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = self._generation_config_to_dict(generation_config)
        if (safety_settings := data["init_parameters"].get("safety_settings")) is not None:
            data["init_parameters"]["safety_settings"] = {k.value: v.value for k, v in safety_settings.items()}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])

        if (tools := data["init_parameters"].get("tools")) is not None:
            deserialized_tools = []
            for tool in tools:
                # Tools are always serialized as a protobuf class, so we need to deserialize them first
                # to be able to convert them to the Python class.
                proto = ToolProto.deserialize(tool)
                deserialized_tools.append(
                    Tool(function_declarations=proto.function_declarations, code_execution=proto.code_execution)
                )
            data["init_parameters"]["tools"] = deserialized_tools
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = GenerationConfig(**generation_config)
        if (safety_settings := data["init_parameters"].get("safety_settings")) is not None:
            data["init_parameters"]["safety_settings"] = {
                HarmCategory(k): HarmBlockThreshold(v) for k, v in safety_settings.items()
            }
        if (serialized_callback_handler := data["init_parameters"].get("streaming_callback")) is not None:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
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
        if message.is_from(ChatRole.ASSISTANT) and message.name:
            p = Part()
            p.function_call.name = message.name
            p.function_call.args = {}
            for k, v in json.loads(message.text).items():
                p.function_call.args[k] = v
            return p
        elif message.is_from(ChatRole.SYSTEM) or message.is_from(ChatRole.ASSISTANT):
            p = Part()
            p.text = message.text
            return p
        elif message.is_from(ChatRole.FUNCTION):
            p = Part()
            p.function_response.name = message.name
            p.function_response.response = message.text
            return p
        elif message.is_from(ChatRole.USER):
            return self._convert_part(message.text)

    def _message_to_content(self, message: ChatMessage) -> Content:
        if message.is_from(ChatRole.ASSISTANT) and message.name:
            part = Part()
            part.function_call.name = message.name
            part.function_call.args = {}
            for k, v in json.loads(message.text).items():
                part.function_call.args[k] = v
        elif message.is_from(ChatRole.SYSTEM) or message.is_from(ChatRole.ASSISTANT):
            part = Part()
            part.text = message.text
        elif message.is_from(ChatRole.FUNCTION):
            part = Part()
            part.function_response.name = message.name
            part.function_response.response = message.text
        elif message.is_from(ChatRole.USER):
            part = self._convert_part(message.text)
        else:
            msg = f"Unsupported message role {message.role}"
            raise ValueError(msg)
        role = "user" if message.is_from(ChatRole.USER) or message.is_from(ChatRole.FUNCTION) else "model"
        return Content(parts=[part], role=role)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Generates text based on the provided messages.

        :param messages:
            A list of `ChatMessage` instances, representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :returns:
            A dictionary containing the following key:
            - `replies`:  A list containing the generated responses as `ChatMessage` instances.
        """
        streaming_callback = streaming_callback or self._streaming_callback
        history = [self._message_to_content(m) for m in messages[:-1]]
        session = self._model.start_chat(history=history)

        new_message = self._message_to_part(messages[-1])
        res = session.send_message(
            content=new_message,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            stream=streaming_callback is not None,
        )

        replies = self._get_stream_response(res, streaming_callback) if streaming_callback else self._get_response(res)

        return {"replies": replies}

    def _get_response(self, response_body: GenerateContentResponse) -> List[ChatMessage]:
        """
        Extracts the responses from the Google AI response.

        :param response_body: The response from Google AI request.
        :returns: The extracted responses.
        """
        replies: List[ChatMessage] = []
        metadata = response_body.to_dict()

        # currently Google only supports one candidate and usage metadata reflects this
        # this should be refactored when multiple candidates are supported
        usage_metadata_openai_format = {}

        usage_metadata = metadata.get("usage_metadata")
        if usage_metadata:
            usage_metadata_openai_format = {
                "prompt_tokens": usage_metadata["prompt_token_count"],
                "completion_tokens": usage_metadata["candidates_token_count"],
                "total_tokens": usage_metadata["total_token_count"],
            }

        for idx, candidate in enumerate(response_body.candidates):
            candidate_metadata = metadata["candidates"][idx]
            candidate_metadata.pop("content", None)  # we remove content from the metadata
            if usage_metadata_openai_format:
                candidate_metadata["usage"] = usage_metadata_openai_format

            for part in candidate.content.parts:
                if part.text != "":
                    replies.append(ChatMessage.from_assistant(content=part.text, meta=candidate_metadata))
                elif part.function_call:
                    candidate_metadata["function_call"] = part.function_call
                    new_message = ChatMessage.from_assistant(
                        content=json.dumps(dict(part.function_call.args)), meta=candidate_metadata
                    )
                    new_message.name = part.function_call.name
                    replies.append(new_message)
        return replies

    def _get_stream_response(
        self, stream: GenerateContentResponse, streaming_callback: Callable[[StreamingChunk], None]
    ) -> List[ChatMessage]:
        """
        Extracts the responses from the Google AI streaming response.

        :param stream: The streaming response from the Google AI request.
        :param streaming_callback: The handler for the streaming response.
        :returns: The extracted response with the content of all streaming chunks.
        """
        replies: List[ChatMessage] = []
        for chunk in stream:
            content: Union[str, Dict[str, Any]] = ""
            dict_chunk = chunk.to_dict()
            metadata = dict(dict_chunk)  # we copy and store the whole chunk as metadata in streaming calls
            for candidate in dict_chunk["candidates"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part and part["text"] != "":
                        content = part["text"]
                        replies.append(ChatMessage.from_assistant(content=content, meta=metadata))
                    elif "function_call" in part and len(part["function_call"]) > 0:
                        metadata["function_call"] = part["function_call"]
                        content = json.dumps(dict(part["function_call"]["args"]))
                        new_message = ChatMessage.from_assistant(content=content, meta=metadata)
                        new_message.name = part["function_call"]["name"]
                        replies.append(new_message)

                    streaming_callback(StreamingChunk(content=content, meta=metadata))
        return replies
