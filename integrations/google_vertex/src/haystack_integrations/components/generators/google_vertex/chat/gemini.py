import json
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional, Union

from haystack import logging
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import AsyncStreamingCallbackT, StreamingCallbackT, StreamingChunk, select_streaming_callback
from haystack.dataclasses.chat_message import ChatMessage, ChatRole, ToolCall
from haystack.tools import Tool, _check_duplicate_tool_names
from haystack.utils import deserialize_callable, serialize_callable

# Compatibility with Haystack 2.12.0 and 2.13.0 - remove after 2.13.0 is released
try:
    from haystack.tools import deserialize_tools_or_toolset_inplace
except ImportError:
    from haystack.tools import deserialize_tools_inplace as deserialize_tools_or_toolset_inplace

from vertexai import init as vertexai_init
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    ToolConfig,
)
from vertexai.generative_models import Tool as VertexTool

logger = logging.getLogger(__name__)


def _convert_chatmessage_to_google_content(message: ChatMessage) -> Content:
    """
    Converts a Haystack `ChatMessage` to a Google `Content` object.
    System messages are not supported.

    :param message: The Haystack `ChatMessage` to convert.
    :returns: The Google `Content` object.
    """

    if message.is_from(ChatRole.SYSTEM):
        msg = "This function does not support system messages."
        raise ValueError(msg)

    texts = message.texts
    tool_calls = message.tool_calls
    tool_call_results = message.tool_call_results

    if not texts and not tool_calls and not tool_call_results:
        msg = "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
        raise ValueError(msg)

    if len(texts) + len(tool_call_results) > 1:
        msg = "A `ChatMessage` can only contain one `TextContent` or one `ToolCallResult`."
        raise ValueError(msg)

    role = "model" if message.is_from(ChatRole.ASSISTANT) else "user"

    if tool_call_results:
        part = Part.from_function_response(
            name=tool_call_results[0].origin.tool_name, response={"result": tool_call_results[0].result}
        )
        return Content(parts=[part], role=role)

    parts = [Part.from_text(texts[0])] if texts else []
    for tc in tool_calls:
        part = Part.from_dict(
            {
                "function_call": {
                    "name": tc.tool_name,
                    "args": tc.arguments,
                }
            }
        )
        parts.append(part)

    return Content(parts=parts, role=role)


@component
class VertexAIGeminiChatGenerator:
    """
    `VertexAIGeminiChatGenerator` enables chat completion using Google Gemini models.

    Authenticates using Google Cloud Application Default Credentials (ADCs).
    For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

    ### Usage example
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.google_vertex import VertexAIGeminiChatGenerator

    gemini_chat = VertexAIGeminiChatGenerator()

    messages = [ChatMessage.from_user("Tell me the name of a movie")]
    res = gemini_chat.run(messages)

    print(res["replies"][0].text)
    >>> The Shawshank Redemption

    #### With Tool calling:

    ```python
    from typing import Annotated
    from haystack.utils import Secret
    from haystack.dataclasses.chat_message import ChatMessage
    from haystack.components.tools import ToolInvoker
    from haystack.tools import create_tool_from_function

    from haystack_integrations.components.generators.google_vertex import VertexAIGeminiChatGenerator

    # example function to get the current weather
    def get_current_weather(
        location: Annotated[str, "The city for which to get the weather, e.g. 'San Francisco'"] = "Munich",
        unit: Annotated[str, "The unit for the temperature, e.g. 'celsius'"] = "celsius",
    ) -> str:
        return f"The weather in {location} is sunny. The temperature is 20 {unit}."

    tool = create_tool_from_function(get_current_weather)
    tool_invoker = ToolInvoker(tools=[tool])

    gemini_chat = VertexAIGeminiChatGenerator(
        model="gemini-2.0-flash-exp",
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
        model: str = "gemini-1.5-flash",
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None,
        tools: Optional[List[Tool]] = None,
        tool_config: Optional[ToolConfig] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
    ):
        """
        `VertexAIGeminiChatGenerator` enables chat completion using Google Gemini models.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

        :param model: Name of the model to use. For available models, see https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models.
        :param project_id: ID of the GCP project to use. By default, it is set during Google Cloud authentication.
        :param location: The default location to use when making API calls, if not set uses us-central-1.
            Defaults to None.
        :param generation_config: Configuration for the generation process.
            See the [GenerationConfig documentation](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.GenerationConfig
            for a list of supported arguments.
        :param safety_settings: Safety settings to use when generating content. See the documentation
            for [HarmBlockThreshold](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.HarmBlockThreshold)
            and [HarmCategory](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.HarmCategory)
            for more details.
        :param tools:
            A list of tools for which the model can prepare calls.
        :param tool_config: The tool config to use. See the documentation for [ToolConfig]
            (https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest/vertexai.generative_models.ToolConfig)
        :param streaming_callback: A callback function that is called when a new token is received from
            the stream. The callback function accepts StreamingChunk as an argument.

        """
        soft_deprecation_msg = (
            "This component uses a deprecated SDK. We recommend using the GoogleGenAIChatGenerator instead. "
            "Documentation is available at https://docs.haystack.deepset.ai/docs/googlegenaichatgenerator."
        )
        logger.warning(soft_deprecation_msg)

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        vertexai_init(project=project_id, location=location)

        _check_duplicate_tool_names(tools)

        self._model_name = model
        self._project_id = project_id
        self._location = location

        # model parameters
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._tool_config = tool_config
        self._streaming_callback = streaming_callback

        self._model = GenerativeModel(
            self._model_name,
            tool_config=self._tool_config,
        )

    @staticmethod
    def _generation_config_to_dict(config: Union[GenerationConfig, Dict[str, Any]]) -> Dict[str, Any]:
        """Converts the GenerationConfig object to a dictionary."""
        if isinstance(config, dict):
            return config
        return config.to_dict()

    @staticmethod
    def _tool_config_to_dict(tool_config: ToolConfig) -> Dict[str, Any]:
        """Serializes the ToolConfig object into a dictionary."""
        mode = tool_config._gapic_tool_config.function_calling_config.mode
        allowed_function_names = tool_config._gapic_tool_config.function_calling_config.allowed_function_names
        config_dict = {"function_calling_config": {"mode": mode}}

        if allowed_function_names:
            config_dict["function_calling_config"]["allowed_function_names"] = allowed_function_names

        return config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        callback_name = serialize_callable(self._streaming_callback) if self._streaming_callback else None

        data = default_to_dict(
            self,
            model=self._model_name,
            project_id=self._project_id,
            location=self._location,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            tools=[tool.to_dict() for tool in self._tools] if self._tools else None,
            tool_config=self._tool_config,
            streaming_callback=callback_name,
        )
        if (tool_config := data["init_parameters"].get("tool_config")) is not None:
            data["init_parameters"]["tool_config"] = self._tool_config_to_dict(tool_config)
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = self._generation_config_to_dict(generation_config)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAIGeminiChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """

        def _tool_config_from_dict(config_dict: Dict[str, Any]) -> ToolConfig:
            """Deserializes the ToolConfig object from a dictionary."""
            function_calling_config = config_dict["function_calling_config"]
            return ToolConfig(
                function_calling_config=ToolConfig.FunctionCallingConfig(
                    mode=function_calling_config["mode"],
                    allowed_function_names=function_calling_config.get("allowed_function_names"),
                )
            )

        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = GenerationConfig.from_dict(generation_config)
        if (tool_config := data["init_parameters"].get("tool_config")) is not None:
            data["init_parameters"]["tool_config"] = _tool_config_from_dict(tool_config)
        if (serialized_callback_handler := data["init_parameters"].get("streaming_callback")) is not None:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @staticmethod
    def _convert_to_vertex_tools(tools: List[Tool]) -> List[VertexTool]:
        """
        Converts a list of Haystack `Tool` to a list of Vertex `Tool` objects.

        :param tools: The list of Haystack `Tool` to convert.
        :returns: The list of Vertex `Tool` objects.
        """
        function_declarations = []

        for tool in tools:
            parameters = tool.parameters.copy()

            # Remove default values as Google API doesn't support them
            for prop in parameters["properties"].values():
                prop.pop("default", None)

            function_declarations.append(
                FunctionDeclaration(name=tool.name, description=tool.description, parameters=parameters)
            )
        return [VertexTool(function_declarations=function_declarations)]

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        tools: Optional[List[Tool]] = None,
    ):
        """
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
        google_tools = self._convert_to_vertex_tools(tools) if tools else None

        if messages[0].is_from(ChatRole.SYSTEM):
            self._model._system_instruction = Part.from_text(messages[0].text)
            messages = messages[1:]

        google_messages = [_convert_chatmessage_to_google_content(m) for m in messages]

        session = self._model.start_chat(history=google_messages[:-1])

        candidate_count = 1
        if self._generation_config:
            config_dict = self._generation_config_to_dict(self._generation_config)
            candidate_count = config_dict.get("candidate_count", 1)

        if streaming_callback and candidate_count > 1:
            msg = "Streaming is not supported with multiple candidates. Set candidate_count to 1."
            raise ValueError(msg)

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

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Async version of the run method. Generates text based on the provided messages.
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
        streaming_callback = select_streaming_callback(
            self._streaming_callback, streaming_callback, requires_async=True
        )

        tools = tools or self._tools
        _check_duplicate_tool_names(tools)
        google_tools = self._convert_to_vertex_tools(tools) if tools else None

        if messages[0].is_from(ChatRole.SYSTEM):
            self._model._system_instruction = Part.from_text(messages[0].text)
            messages = messages[1:]

        google_messages = [_convert_chatmessage_to_google_content(m) for m in messages]

        session = self._model.start_chat(history=google_messages[:-1])

        candidate_count = 1
        if self._generation_config:
            config_dict = self._generation_config_to_dict(self._generation_config)
            candidate_count = config_dict.get("candidate_count", 1)

        if streaming_callback and candidate_count > 1:
            msg = "Streaming is not supported with multiple candidates. Set candidate_count to 1."
            raise ValueError(msg)

        res = await session.send_message_async(
            content=google_messages[-1],
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            stream=streaming_callback is not None,
            tools=google_tools,
            tool_config=self._tool_config,
        )

        replies = (
            await self._stream_response_and_convert_to_messages_async(res, streaming_callback)
            if streaming_callback
            else self._convert_response_to_messages(res)
        )

        return {"replies": replies}

    @staticmethod
    def _convert_response_to_messages(response_body: GenerationResponse) -> List[ChatMessage]:
        """
        Converts the Google Vertex AI response to a list of `ChatMessage` instances.

        :param response_body: The response from Google AI request.
        :returns: List of `ChatMessage` instances.
        """
        replies: List[ChatMessage] = []

        usage_metadata = response_body.usage_metadata
        openai_usage = {
            "prompt_tokens": usage_metadata.prompt_token_count or 0,
            "completion_tokens": usage_metadata.candidates_token_count or 0,
            "total_tokens": usage_metadata.total_token_count or 0,
        }

        for candidate in response_body.candidates:
            candidate_metadata = candidate.to_dict()
            candidate_metadata.pop("content", None)
            candidate_metadata["usage"] = openai_usage

            text = ""
            tool_calls = []
            for part in candidate.content.parts:
                # we need this strange check: calling part.text directly raises an error if the part has no text
                if "text" in part._raw_part:
                    text += part.text
                elif "function_call" in part._raw_part:
                    tool_calls.append(
                        ToolCall(
                            tool_name=part.function_call.name,
                            arguments=dict(part.function_call.args),
                        )
                    )
            reply = ChatMessage.from_assistant(text=text or None, tool_calls=tool_calls, meta=candidate_metadata)
            replies.append(reply)
        return replies

    def _stream_response_and_convert_to_messages(
        self, stream: Iterable[GenerationResponse], streaming_callback: StreamingCallbackT
    ) -> List[ChatMessage]:
        """
        Streams the Google Vertex AI response and converts it to a list of `ChatMessage` instances.

        :param stream: The streaming response from the Google AI request.
        :param streaming_callback: The handler for the streaming response.
        :returns: List of `ChatMessage` instances.
        """

        text = ""
        tool_calls = []
        chunk_dict = {}

        for chunk in stream:
            content_to_stream = ""
            chunk_dict = chunk.to_dict()

            # Only one candidate is supported with streaming
            candidate = chunk_dict["candidates"][0]

            for part in candidate["content"]["parts"]:
                if new_text := part.get("text"):
                    content_to_stream += new_text
                    text += new_text
                elif new_function_call := part.get("function_call"):
                    content_to_stream += json.dumps(dict(new_function_call))
                    tool_calls.append(
                        ToolCall(
                            tool_name=new_function_call["name"],
                            arguments=dict(new_function_call["args"]),
                        )
                    )

            streaming_callback(StreamingChunk(content=content_to_stream, meta=chunk_dict))

        # store the last chunk metadata
        meta = chunk_dict

        # format the usage metadata to be compatible with OpenAI
        usage_metadata = meta.pop("usage_metadata", {})

        openai_usage = {
            "prompt_tokens": usage_metadata.get("prompt_token_count", 0),
            "completion_tokens": usage_metadata.get("candidates_token_count", 0),
            "total_tokens": usage_metadata.get("total_token_count", 0),
        }

        meta["usage"] = openai_usage

        return [ChatMessage.from_assistant(text=text or None, meta=meta, tool_calls=tool_calls)]

    @staticmethod
    async def _stream_response_and_convert_to_messages_async(
        stream: AsyncIterable[GenerationResponse], streaming_callback: AsyncStreamingCallbackT
    ) -> List[ChatMessage]:
        """
        Streams the Google Vertex AI response and converts it to a list of `ChatMessage` instances.

        :param stream: The streaming response from the Google AI request.
        :param streaming_callback: The handler for the streaming response.
        :returns: List of `ChatMessage` instances.
        """

        text = ""
        tool_calls = []
        chunk_dict = {}

        async for chunk in stream:
            content_to_stream = ""
            chunk_dict = chunk.to_dict()

            # Only one candidate is supported with streaming
            candidate = chunk_dict["candidates"][0]

            for part in candidate["content"]["parts"]:
                if new_text := part.get("text"):
                    content_to_stream += new_text
                    text += new_text
                elif new_function_call := part.get("function_call"):
                    content_to_stream += json.dumps(dict(new_function_call))
                    tool_calls.append(
                        ToolCall(
                            tool_name=new_function_call["name"],
                            arguments=new_function_call["args"],
                        )
                    )

            await streaming_callback(StreamingChunk(content=content_to_stream, meta=chunk_dict))

        # store the last chunk metadata
        meta = chunk_dict

        # format the usage metadata to be compatible with OpenAI
        usage_metadata = meta.pop("usage_metadata", {})

        openai_usage = {
            "prompt_tokens": usage_metadata.get("prompt_token_count", 0),
            "completion_tokens": usage_metadata.get("candidates_token_count", 0),
            "total_tokens": usage_metadata.get("total_token_count", 0),
        }

        meta["usage"] = openai_usage

        return [ChatMessage.from_assistant(text=text or None, meta=meta, tool_calls=tool_calls)]
