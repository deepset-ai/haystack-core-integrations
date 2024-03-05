import logging
from typing import Any, Dict, List, Optional, Union

import vertexai
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
from vertexai.preview.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Tool,
)

logger = logging.getLogger(__name__)


@component
class VertexAIGeminiChatGenerator:
    """
    `VertexAIGeminiChatGenerator` enables chat completion using Google Gemini models.

    `VertexAIGeminiChatGenerator` supports both `gemini-pro` and `gemini-pro-vision` models.
    Prompting with images requires `gemini-pro-vision`. Function calling, instead, requires `gemini-pro`.

    Authenticates using Google Cloud Application Default Credentials (ADCs).
    For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.google_vertex import VertexAIGeminiChatGenerator

    gemini_chat = VertexAIGeminiChatGenerator(project_id=project_id)

    messages = [ChatMessage.from_user("Tell me the name of a movie")]
    res = gemini_chat.run(messages)

    print(res["replies"][0].content)
    >>> The Shawshank Redemption
    ```
    """

    def __init__(
        self,
        *,
        model: str = "gemini-pro",
        project_id: str,
        location: Optional[str] = None,
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        `VertexAIGeminiChatGenerator` enables chat completion using Google Gemini models.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

        :param project_id: ID of the GCP project to use.
        :param model: Name of the model to use, defaults to "gemini-pro-vision".
        :param location: The default location to use when making API calls, if not set uses us-central-1.
            Defaults to None.
        :param generation_config: Configuration for the generation process.
            See the [GenerationConfig documentation](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.preview.generative_models.GenerationConfig
            for a list of supported arguments.
        :param safety_settings: Safety settings to use when generating content. See the documentation
            for [HarmBlockThreshold](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.preview.generative_models.HarmBlockThreshold)
            and [HarmCategory](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.preview.generative_models.HarmCategory)
            for more details.
        :param tools: List of tools to use when generating content. See the documentation for
            [Tool](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.preview.generative_models.Tool)
            the list of supported arguments.
        """

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        vertexai.init(project=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._location = location
        self._model = GenerativeModel(self._model_name)

        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools

    def _function_to_dict(self, function: FunctionDeclaration) -> Dict[str, Any]:
        return {
            "name": function._raw_function_declaration.name,
            "parameters": function._raw_function_declaration.parameters,
            "description": function._raw_function_declaration.description,
        }

    def _tool_to_dict(self, tool: Tool) -> Dict[str, Any]:
        return {
            "function_declarations": [self._function_to_dict(f) for f in tool._raw_tool.function_declarations],
        }

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        data = default_to_dict(
            self,
            model=self._model_name,
            project_id=self._project_id,
            location=self._location,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            tools=self._tools,
        )
        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = [self._tool_to_dict(t) for t in tools]
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
        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = [Tool.from_dict(t) for t in tools]
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = GenerationConfig.from_dict(generation_config)

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
        if message.role == ChatRole.SYSTEM and message.name:
            p = Part.from_dict({"function_call": {"name": message.name, "args": {}}})
            for k, v in message.content.items():
                p.function_call.args[k] = v
            return p
        elif message.role == ChatRole.SYSTEM:
            return Part.from_text(message.content)
        elif message.role == ChatRole.FUNCTION:
            return Part.from_function_response(name=message.name, response=message.content)
        elif message.role == ChatRole.USER:
            return self._convert_part(message.content)

    def _message_to_content(self, message: ChatMessage) -> Content:
        if message.role == ChatRole.SYSTEM and message.name:
            part = Part.from_dict({"function_call": {"name": message.name, "args": {}}})
            for k, v in message.content.items():
                part.function_call.args[k] = v
        elif message.role == ChatRole.SYSTEM:
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
    def run(self, messages: List[ChatMessage]):
        """Prompts Google Vertex AI Gemini model to generate a response to a list of messages.

        :param messages: The last message is the prompt, the rest are the history.
        :returns: A dictionary with the following keys:
            - `replies`: A list of ChatMessage objects representing the model's replies.
        """
        history = [self._message_to_content(m) for m in messages[:-1]]
        session = self._model.start_chat(history=history)

        new_message = self._message_to_part(messages[-1])
        res = session.send_message(
            content=new_message,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            tools=self._tools,
        )

        replies = []
        for candidate in res.candidates:
            for part in candidate.content.parts:
                if part._raw_part.text != "":
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
