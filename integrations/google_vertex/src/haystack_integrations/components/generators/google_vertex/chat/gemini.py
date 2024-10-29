import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
from haystack.utils import deserialize_callable, serialize_callable
from vertexai import init as vertexai_init
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Tool,
    ToolConfig,
)

logger = logging.getLogger(__name__)


@component
class VertexAIGeminiChatGenerator:
    """
    `VertexAIGeminiChatGenerator` enables chat completion using Google Gemini models.

    Authenticates using Google Cloud Application Default Credentials (ADCs).
    For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.google_vertex import VertexAIGeminiChatGenerator

    gemini_chat = VertexAIGeminiChatGenerator()

    messages = [ChatMessage.from_user("Tell me the name of a movie")]
    res = gemini_chat.run(messages)

    print(res["replies"][0].content)
    >>> The Shawshank Redemption
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
        system_instruction: Optional[Union[str, ByteStream, Part]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        `VertexAIGeminiChatGenerator` enables chat completion using Google Gemini models.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

        :param project_id: ID of the GCP project to use. By default, it is set during Google Cloud authentication.
        :param model: Name of the model to use. For available models, see https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models.
        :param location: The default location to use when making API calls, if not set uses us-central-1.
            Defaults to None.
        :param generation_config: Configuration for the generation process.
            See the [GenerationConfig documentation](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.GenerationConfig
            for a list of supported arguments.
        :param safety_settings: Safety settings to use when generating content. See the documentation
            for [HarmBlockThreshold](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.HarmBlockThreshold)
            and [HarmCategory](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.HarmCategory)
            for more details.
        :param tools: List of tools to use when generating content. See the documentation for
            [Tool](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.Tool)
            the list of supported arguments.
        :param tool_config: The tool config to use. See the documentation for [ToolConfig]
            (https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest/vertexai.generative_models.ToolConfig)
        :param system_instruction: Default system instruction to use for generating content.
        :param streaming_callback: A callback function that is called when a new token is received from
            the stream. The callback function accepts StreamingChunk as an argument.

        """

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        vertexai_init(project=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._location = location

        # model parameters
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._tool_config = tool_config
        self._system_instruction = system_instruction
        self._streaming_callback = streaming_callback

        # except streaming_callback, all other model parameters can be passed during initialization
        self._model = GenerativeModel(
            self._model_name,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            tools=self._tools,
            tool_config=self._tool_config,
            system_instruction=self._system_instruction,
        )

    def _generation_config_to_dict(self, config: Union[GenerationConfig, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        return {
            "temperature": config._raw_generation_config.temperature,
            "top_p": config._raw_generation_config.top_p,
            "top_k": config._raw_generation_config.top_k,
            "candidate_count": config._raw_generation_config.candidate_count,
            "max_output_tokens": config._raw_generation_config.max_output_tokens,
            "stop_sequences": config._raw_generation_config.stop_sequences,
        }

    def _tool_config_to_dict(self, tool_config: ToolConfig) -> Dict[str, Any]:
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
            tools=self._tools,
            tool_config=self._tool_config,
            system_instruction=self._system_instruction,
            streaming_callback=callback_name,
        )
        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = [Tool.to_dict(t) for t in tools]
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

        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = [Tool.from_dict(t) for t in tools]
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = GenerationConfig.from_dict(generation_config)
        if (tool_config := data["init_parameters"].get("tool_config")) is not None:
            data["init_parameters"]["tool_config"] = _tool_config_from_dict(tool_config)
        if (serialized_callback_handler := data["init_parameters"].get("streaming_callback")) is not None:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _convert_part(self, part: Union[str, ByteStream, Part]) -> Part:
        if isinstance(part, str):
            return Part.from_text(part)
        elif isinstance(part, ByteStream):
            return Part.from_data(part.data, part.mime_type)
        elif isinstance(part, Part):
            return part
        else:
            msg = f"Unsupported type {type(part)} for part {part}"
            raise ValueError(msg)

    def _message_to_part(self, message: ChatMessage) -> Part:
        if message.role == ChatRole.ASSISTANT and message.name:
            p = Part.from_dict({"function_call": {"name": message.name, "args": {}}})
            for k, v in message.content.items():
                p.function_call.args[k] = v
            return p
        elif message.role in {ChatRole.SYSTEM, ChatRole.ASSISTANT}:
            return Part.from_text(message.content)
        elif message.role == ChatRole.FUNCTION:
            return Part.from_function_response(name=message.name, response=message.content)
        elif message.role == ChatRole.USER:
            return self._convert_part(message.content)

    def _message_to_content(self, message: ChatMessage) -> Content:
        if message.role == ChatRole.ASSISTANT and message.name:
            part = Part.from_dict({"function_call": {"name": message.name, "args": {}}})
            for k, v in message.content.items():
                part.function_call.args[k] = v
        elif message.role in {ChatRole.SYSTEM, ChatRole.ASSISTANT}:
            part = Part.from_text(message.content)
        elif message.role == ChatRole.FUNCTION:
            part = Part.from_function_response(name=message.name, response=message.content)
        elif message.role == ChatRole.USER:
            part = self._convert_part(message.content)
        else:
            msg = f"Unsupported message role {message.role}"
            raise ValueError(msg)
        role = "user" if message.role in [ChatRole.USER, ChatRole.FUNCTION] else "model"
        return Content(parts=[part], role=role)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """Prompts Google Vertex AI Gemini model to generate a response to a list of messages.

        :param messages: The last message is the prompt, the rest are the history.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :returns: A dictionary with the following keys:
            - `replies`: A list of ChatMessage objects representing the model's replies.
        """
        # check if streaming_callback is passed
        streaming_callback = streaming_callback or self._streaming_callback

        history = [self._message_to_content(m) for m in messages[:-1]]
        session = self._model.start_chat(history=history)

        new_message = self._message_to_part(messages[-1])
        res = session.send_message(
            content=new_message,
            stream=streaming_callback is not None,
        )

        replies = self._get_stream_response(res, streaming_callback) if streaming_callback else self._get_response(res)

        return {"replies": replies}

    def _get_response(self, response_body: GenerationResponse) -> List[ChatMessage]:
        """
        Extracts the responses from the Vertex AI response.

        :param response_body: The response from Vertex AI request.
        :returns: The extracted responses.
        """
        replies: List[ChatMessage] = []
        for candidate in response_body.candidates:
            metadata = candidate.to_dict()
            for part in candidate.content.parts:
                # Remove content from metadata
                metadata.pop("content", None)
                if part._raw_part.text != "":
                    replies.append(
                        ChatMessage(content=part._raw_part.text, role=ChatRole.ASSISTANT, name=None, meta=metadata)
                    )
                elif part.function_call:
                    metadata["function_call"] = part.function_call
                    replies.append(
                        ChatMessage(
                            content=dict(part.function_call.args.items()),
                            role=ChatRole.ASSISTANT,
                            name=part.function_call.name,
                            meta=metadata,
                        )
                    )
        return replies

    def _get_stream_response(
        self, stream: Iterable[GenerationResponse], streaming_callback: Callable[[StreamingChunk], None]
    ) -> List[ChatMessage]:
        """
        Extracts the responses from the Vertex AI streaming response.

        :param stream: The streaming response from the Vertex AI request.
        :param streaming_callback: The handler for the streaming response.
        :returns: The extracted response with the content of all streaming chunks.
        """
        replies: List[ChatMessage] = []

        for chunk in stream:
            content: Union[str, Dict[str, Any]] = ""
            metadata = chunk.to_dict()  # we store whole chunk as metadata for streaming
            for candidate in chunk.candidates:
                for part in candidate.content.parts:
                    if part._raw_part.text:
                        content = chunk.text
                        replies.append(ChatMessage(content, role=ChatRole.ASSISTANT, name=None, meta=metadata))
                    elif part.function_call:
                        metadata["function_call"] = part.function_call
                        content = dict(part.function_call.args.items())
                        replies.append(
                            ChatMessage(
                                content=content,
                                role=ChatRole.ASSISTANT,
                                name=part.function_call.name,
                                meta=metadata,
                            )
                        )
                    streaming_callback(StreamingChunk(content=content, meta=metadata))

        return replies
