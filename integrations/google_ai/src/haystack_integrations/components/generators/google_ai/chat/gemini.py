import logging
from typing import Any, Callable, Dict, List, Optional, Union

import google.generativeai as genai
from google.ai.generativelanguage import Content, Part
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import (
    FunctionDeclaration,
    GenerateContentResponse,
    HarmBlockThreshold,
    HarmCategory,
    content_types,
)
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.chat_message import ChatMessage, ChatRole, ToolCall
from haystack.tools import Tool, _check_duplicate_tool_names, deserialize_tools_inplace
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

logger = logging.getLogger(__name__)


def _convert_chatmessage_to_google_content(message: ChatMessage) -> Content:
    """
    Converts a Haystack `ChatMessage` to a Google AI `Content` object.
    System messages are not supported.

    :param message: The Haystack `ChatMessage` to convert.
    :returns: The Google AI `Content` object.
    """

    if message.is_from("system"):
        msg = "This function does not support system messages."
        raise ValueError(msg)

    text_contents = message.texts
    tool_calls = message.tool_calls
    tool_call_results = message.tool_call_results

    if not text_contents and not tool_calls and not tool_call_results:
        msg = "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
        raise ValueError(msg)

    if len(text_contents) + len(tool_call_results) > 1:
        msg = "A `ChatMessage` can only contain one `TextContent` or one `ToolCallResult`."
        raise ValueError(msg)

    role = "model" if message.is_from(ChatRole.ASSISTANT) else "user"

    if tool_call_results:
        part = Part(
            function_response=genai.protos.FunctionResponse(
                name=tool_call_results[0].origin.tool_name, response={"result": tool_call_results[0].result}
            )
        )
        return Content(parts=[part], role=role)

    parts = []
    if text_contents:
        part = Part()
        part.text = text_contents[0]
        parts.append(part)

    if tool_calls:
        for tc in tool_calls:
            part = Part(function_call=genai.protos.FunctionCall(name=tc.tool_name, args=tc.arguments))
            parts.append(part)

    return Content(parts=parts, role=role)


@component
class GoogleAIGeminiChatGenerator:
    """
    Completes chats using Gemini models through Google AI Studio.

    It uses the [`ChatMessage`](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)
      dataclass to interact with the model.

    ### Usage example

    ```python
    from haystack.utils import Secret
    from haystack.dataclasses.chat_message import ChatMessage
    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator


    gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-1.5-flash", api_key=Secret.from_token("<MY_API_KEY>"))

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
    from typing import Annotated
    from haystack.utils import Secret
    from haystack.dataclasses.chat_message import ChatMessage
    from haystack.components.tools import ToolInvoker
    from haystack.tools import create_tool_from_function

    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator

    # example function to get the current weather
    def get_current_weather(
        location: Annotated[str, "The city for which to get the weather, e.g. 'San Francisco'"] = "Munich",
        unit: Annotated[str, "The unit for the temperature, e.g. 'celsius'"] = "celsius",
    ) -> str:
        return f"The weather in {location} is sunny. The temperature is 20 {unit}."

    tool = create_tool_from_function(get_current_weather)
    tool_invoker = ToolInvoker(tools=[tool])

    gemini_chat = GoogleAIGeminiChatGenerator(
        model="gemini-2.0-flash-exp",
        api_key=Secret.from_token("<MY_API_KEY>"),
        tools=[tool],
    )
    user_message = [ChatMessage.from_user("What is the temperature in celsius in Berlin?")]
    replies = gemini_chat.run(messages=user_message)["replies"]
    print(replies[0].tool_calls)

    # actually invoke the tool
    tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
    messages = user_message + replies + tool_messages

    # transform the tool call result into a human readable message
    final_replies = gemini_chat.run(messages=messages)["replies"]
    print(final_replies[0].text)
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

        To get an API key, visit: https://aistudio.google.com/

        :param api_key: Google AI Studio API key. To get a key,
        see [Google AI Studio](https://aistudio.google.com/).
        :param model: Name of the model to use. For available models, see https://ai.google.dev/gemini-api/docs/models/gemini.
        :param generation_config: The generation configuration to use.
            This can either be a `GenerationConfig` object or a dictionary of parameters.
            For available parameters, see
            [the API reference](https://ai.google.dev/api/generate-content).
        :param safety_settings: The safety settings to use.
            A dictionary with `HarmCategory` as keys and `HarmBlockThreshold` as values.
            For more information, see [the API reference](https://ai.google.dev/api/generate-content)
        :param tools:
            A list of tools for which the model can prepare calls.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        """

        genai.configure(api_key=api_key.resolve_value())
        _check_duplicate_tool_names(tools)

        self._api_key = api_key
        self._model_name = model
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._model = GenerativeModel(self._model_name)
        self._streaming_callback = streaming_callback

    @staticmethod
    def _generation_config_to_dict(config: Union[GenerationConfig, Dict[str, Any]]) -> Dict[str, Any]:
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
            tools=[tool.to_dict() for tool in self._tools] if self._tools else None,
            streaming_callback=callback_name,
        )
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
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = GenerationConfig(**generation_config)
        if (safety_settings := data["init_parameters"].get("safety_settings")) is not None:
            data["init_parameters"]["safety_settings"] = {
                HarmCategory(k): HarmBlockThreshold(v) for k, v in safety_settings.items()
            }
        if (serialized_callback_handler := data["init_parameters"].get("streaming_callback")) is not None:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @staticmethod
    def _convert_tool_to_google_tool(tool: Tool) -> FunctionDeclaration:
        """
        Converts a Haystack `Tool` to a Google AI `FunctionDeclaration` object.

        :param tool: The Haystack `Tool` to convert.
        :returns: The Google AI `FunctionDeclaration` object.
        """
        parameters = tool.parameters.copy()

        # Remove default values as Google API doesn't support them
        for prop in parameters["properties"].values():
            prop.pop("default", None)

        return FunctionDeclaration(name=tool.name, description=tool.description, parameters=parameters)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        *,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Generates text based on the provided messages.

        :param messages:
            A list of `ChatMessage` instances, representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param tools:
            A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
            during component initialization.
        :returns:
            A dictionary containing the following key:
            - `replies`:  A list containing the generated responses as `ChatMessage` instances.
        """
        streaming_callback = streaming_callback or self._streaming_callback

        tools = tools or self._tools
        _check_duplicate_tool_names(tools)
        google_tools = [self._convert_tool_to_google_tool(tool) for tool in tools] if tools else None

        if messages[0].is_from("system"):
            self._model._system_instruction = content_types.to_content(messages[0].text)
            messages = messages[1:]

        google_messages = [_convert_chatmessage_to_google_content(m) for m in messages]

        session = self._model.start_chat(history=google_messages[:-1])

        res = session.send_message(
            content=google_messages[-1],
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            stream=streaming_callback is not None,
            tools=google_tools,
        )

        replies = (
            self._stream_response_and_convert_to_messages(res, streaming_callback)
            if streaming_callback
            else self._convert_response_to_messages(res)
        )

        return {"replies": replies}

    @staticmethod
    def _convert_response_to_messages(response_body: GenerateContentResponse) -> List[ChatMessage]:
        """
        Converts the Google AI response to a list of `ChatMessage` instances.

        :param response_body: The response from Google AI request.
        :returns: List of `ChatMessage` instances.
        """
        metadata = response_body.to_dict()

        # Currently, only one candidate is supported
        candidate = response_body.candidates[0]
        candidate_metadata = metadata["candidates"][0]
        candidate_metadata.pop("content", None)  # we remove content from the metadata

        # adapt usage metadata to OpenAI format
        if usage_metadata := metadata.get("usage_metadata"):
            candidate_metadata["usage"] = {
                "prompt_tokens": usage_metadata["prompt_token_count"],
                "completion_tokens": usage_metadata["candidates_token_count"],
                "total_tokens": usage_metadata["total_token_count"],
            }

        text = ""
        tool_calls = []

        for part in candidate.content.parts:
            if part.text:
                text += part.text
            elif part.function_call:
                tool_calls.append(
                    ToolCall(
                        tool_name=part.function_call.name,
                        arguments=dict(part.function_call.args),
                    )
                )

        return [ChatMessage.from_assistant(text=text or None, meta=candidate_metadata, tool_calls=tool_calls)]

    @staticmethod
    def _stream_response_and_convert_to_messages(
        stream: GenerateContentResponse, streaming_callback: Callable[[StreamingChunk], None]
    ) -> List[ChatMessage]:
        """
        Streams the Google AI response and converts it to a list of `ChatMessage` instances.

        :param stream: The streaming response from the Google AI request.
        :param streaming_callback: The handler for the streaming response.
        :returns: List of `ChatMessage` instances.
        """
        text = ""
        tool_calls = []
        last_metadata = None

        for chunk in stream:
            chunk_dict = chunk.to_dict()
            last_metadata = chunk_dict
            candidate = chunk_dict["candidates"][0]

            for part in candidate["content"]["parts"]:
                if part.get("text"):
                    text += part["text"]
                elif part.get("function_call"):
                    tool_calls.append(
                        ToolCall(
                            tool_name=part["function_call"]["name"],
                            arguments=dict(part["function_call"]["args"]),
                        )
                    )

            streaming_callback(StreamingChunk(content=text, meta=chunk_dict))

        return [ChatMessage.from_assistant(text=text or None, meta=last_metadata, tool_calls=tool_calls)]
