import logging
from typing import Any, Callable, Dict, List, Optional, Union

import google.generativeai as genai
from google.ai.generativelanguage import Content, Part
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import GenerateContentResponse, HarmBlockThreshold, HarmCategory
from haystack.core.component import component
from haystack.core.component.types import Variadic
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream, StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

logger = logging.getLogger(__name__)


@component
class GoogleAIGeminiGenerator:
    """
    Generates text using multimodal Gemini models through Google AI Studio.

    ### Usage example

    ```python
    from haystack.utils import Secret
    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

    gemini = GoogleAIGeminiGenerator(model="gemini-pro", api_key=Secret.from_token("<MY_API_KEY>"))
    res = gemini.run(parts = ["What is the most interesting thing you know?"])
    for answer in res["replies"]:
        print(answer)
    ```

    #### Multimodal example

    ```python
    import requests
    from haystack.utils import Secret
    from haystack.dataclasses.byte_stream import ByteStream
    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

    BASE_URL = (
        "https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations"
        "/main/integrations/google_ai/example_assets"
    )

    URLS = [
        f"{BASE_URL}/robot1.jpg",
        f"{BASE_URL}/robot2.jpg",
        f"{BASE_URL}/robot3.jpg",
        f"{BASE_URL}/robot4.jpg"
    ]
    images = [
        ByteStream(data=requests.get(url).content, mime_type="image/jpeg")
        for url in URLS
    ]

    gemini = GoogleAIGeminiGenerator(model="gemini-1.5-flash", api_key=Secret.from_token("<MY_API_KEY>"))
    result = gemini.run(parts = ["What can you tell me about this robots?", *images])
    for answer in result["replies"]:
        print(answer)
    ```
    """

    def __new__(cls, *_, **kwargs):
        if "tools" in kwargs:
            msg = (
                "GoogleAIGeminiGenerator does not support the `tools` parameter. "
                " Use GoogleAIGeminiChatGenerator instead."
            )
            raise TypeError(msg)
        return super(GoogleAIGeminiGenerator, cls).__new__(cls)  # noqa: UP008
        # super(__class__, cls) is needed because of the component decorator

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("GOOGLE_API_KEY"),  # noqa: B008
        model: str = "gemini-1.5-flash",
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initializes a `GoogleAIGeminiGenerator` instance.

        To get an API key, visit: https://makersuite.google.com

        :param api_key: Google AI Studio API key.
        :param model: Name of the model to use. For available models, see https://ai.google.dev/gemini-api/docs/models/gemini
        :param generation_config: The generation configuration to use.
            This can either be a `GenerationConfig` object or a dictionary of parameters.
            For available parameters, see
            [the `GenerationConfig` API reference](https://ai.google.dev/api/python/google/generativeai/GenerationConfig).
        :param safety_settings: The safety settings to use.
            A dictionary with `HarmCategory` as keys and `HarmBlockThreshold` as values.
            For more information, see [the API reference](https://ai.google.dev/api)
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        """
        genai.configure(api_key=api_key.resolve_value())

        self._api_key = api_key
        self._model_name = model
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._model = GenerativeModel(self._model_name)
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
            streaming_callback=callback_name,
        )
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = self._generation_config_to_dict(generation_config)
        if (safety_settings := data["init_parameters"].get("safety_settings")) is not None:
            data["init_parameters"]["safety_settings"] = {k.value: v.value for k, v in safety_settings.items()}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])

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

    @component.output_types(replies=List[str])
    def run(
        self,
        parts: Variadic[Union[str, ByteStream, Part]],
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Generates text based on the given input parts.

        :param parts:
            A heterogeneous list of strings, `ByteStream` or `Part` objects.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :returns:
            A dictionary containing the following key:
            - `replies`: A list of strings containing the generated responses.
        """

        # check if streaming_callback is passed
        streaming_callback = streaming_callback or self._streaming_callback
        converted_parts = [self._convert_part(p) for p in parts]
        contents = [Content(parts=converted_parts, role="user")]
        res = self._model.generate_content(
            contents=contents,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
            stream=streaming_callback is not None,
        )
        self._model.start_chat()
        replies = self._get_stream_response(res, streaming_callback) if streaming_callback else self._get_response(res)

        return {"replies": replies}

    def _get_response(self, response_body: GenerateContentResponse) -> List[str]:
        """
        Extracts the responses from the Google AI request.
        :param response_body: The response body from the Google AI request.
        :returns: A list of string responses.
        """
        replies = []
        for candidate in response_body.candidates:
            for part in candidate.content.parts:
                if part.text != "":
                    replies.append(part.text)
        return replies

    def _get_stream_response(
        self, stream: GenerateContentResponse, streaming_callback: Callable[[StreamingChunk], None]
    ) -> List[str]:
        """
        Extracts the responses from the Google AI streaming response.
        :param stream: The streaming response from the Google AI request.
        :param streaming_callback: The handler for the streaming response.
        :returns: A list of string responses.
        """

        responses = []
        for chunk in stream:
            content = chunk.text if len(chunk.parts) > 0 and "text" in chunk.parts[0] else ""
            streaming_callback(StreamingChunk(content=content, meta=chunk.to_dict()))
            responses.append(content)

        combined_response = ["".join(responses).lstrip()]
        return combined_response
