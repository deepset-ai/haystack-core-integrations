import logging
from typing import Any, Callable, Dict, List, Optional, Union

import vertexai
from haystack.core.component import component
from haystack.core.component.types import Variadic
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream, StreamingChunk
from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Tool,
)

logger = logging.getLogger(__name__)


@component
class VertexAIGeminiGenerator:
    """
    `VertexAIGeminiGenerator` enables text generation using Google Gemini models.

    `VertexAIGeminiGenerator` supports both `gemini-pro` and `gemini-pro-vision` models.
    Prompting with images requires `gemini-pro-vision`. Function calling, instead, requires `gemini-pro`.

    Usage example:
    ```python
    from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator


    gemini = VertexAIGeminiGenerator(project_id=project_id)
    result = gemini.run(parts = ["What is the most interesting thing you know?"])
    for answer in result["replies"]:
        print(answer)

    >>> 1. **The Origin of Life:** How and where did life begin? The answers to this ...
    >>> 2. **The Unseen Universe:** The vast majority of the universe is ...
    >>> 3. **Quantum Entanglement:** This eerie phenomenon in quantum mechanics allows ...
    >>> 4. **Time Dilation:** Einstein's theory of relativity revealed that time can ...
    >>> 5. **The Fermi Paradox:** Despite the vastness of the universe and the ...
    >>> 6. **Biological Evolution:** The idea that life evolves over time through natural ...
    >>> 7. **Neuroplasticity:** The brain's ability to adapt and change throughout life, ...
    >>> 8. **The Goldilocks Zone:** The concept of the habitable zone, or the Goldilocks zone, ...
    >>> 9. **String Theory:** This theoretical framework in physics aims to unify all ...
    >>> 10. **Consciousness:** The nature of human consciousness and how it arises ...
    ```
    """

    def __init__(
        self,
        *,
        model: str = "gemini-pro-vision",
        project_id: str,
        location: Optional[str] = None,
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None,
        tools: Optional[List[Tool]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Multi-modal generator using Gemini model via Google Vertex AI.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

        :param project_id: ID of the GCP project to use.
        :param model: Name of the model to use.
        :param location: The default location to use when making API calls, if not set uses us-central-1.
        :param generation_config: The generation config to use.
            Can either be a [`GenerationConfig`](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.preview.generative_models.GenerationConfig)
            object or a dictionary of parameters.
            Accepted fields are:
                - temperature
                - top_p
                - top_k
                - candidate_count
                - max_output_tokens
                - stop_sequences
        :param safety_settings: The safety settings to use. See the documentation
            for [HarmBlockThreshold](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.preview.generative_models.HarmBlockThreshold)
            and [HarmCategory](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.preview.generative_models.HarmCategory)
            for more details.
        :param tools: List of tools to use when generating content. See the documentation for
            [Tool](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.preview.generative_models.Tool)
            the list of supported arguments.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
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
        self._streaming_callback = streaming_callback

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
            streaming_callback=self._streaming_callback,
        )
        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = [Tool.to_dict(t) for t in tools]
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = self._generation_config_to_dict(generation_config)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAIGeminiGenerator":
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

    @component.output_types(replies=List[Union[str, Dict[str, str]]])
    def run(self, parts: Variadic[Union[str, ByteStream, Part]]):
        """
        Generates content using the Gemini model.

        :param parts: Prompt for the model.
        :returns: A dictionary with the following keys:
            - `replies`: A list of generated content.
        """
        converted_parts = [self._convert_part(p) for p in parts]

        contents = [Content(parts=converted_parts, role="user")]
        res = self._model.generate_content(
            contents=contents,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            tools=self._tools,
            stream=self._streaming_callback is not None,
        )
        self._model.start_chat()
        replies = (
            self.get_stream_response(res, self._streaming_callback)
            if self._streaming_callback
            else self.get_response(res)
        )

        return {"replies": replies}

    def get_response(self, response_body) -> List[str]:
        """
        Extracts the responses from the Vertex AI response.

        :param response_body: The response body from the Amazon Bedrock request.

        :returns: A list of string responses.
        """
        replies = []
        for candidate in response_body.candidates:
            for part in candidate.content.parts:
                if part._raw_part.text != "":
                    replies.append(part.text)
                elif part.function_call is not None:
                    function_call = {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args.items()),
                    }
                    replies.append(function_call)
        return replies

    def get_stream_response(self, stream, streaming_callback: Callable[[StreamingChunk], None]) -> List[str]:
        """
        Extracts the responses from the Vertex AI streaming response.

        :param stream: The streaming response from the Vertex AI request.
        :param streaming_callback: The handler for the streaming response.
        :returns: A list of string responses.
        """
        streaming_chunks: List[StreamingChunk] = []

        for chunk in stream:
            streaming_chunk = StreamingChunk(content=chunk.text, meta=chunk.usage_metadata)
            streaming_chunks.append(streaming_chunk)
            streaming_callback(streaming_chunk)

        responses = ["".join(streaming_chunk.content for streaming_chunk in streaming_chunks).lstrip()]
        return responses
