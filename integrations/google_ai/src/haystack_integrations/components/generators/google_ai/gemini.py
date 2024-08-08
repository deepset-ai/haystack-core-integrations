import logging
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
from google.ai.generativelanguage import Content, Part, Tool
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from haystack.core.component import component
from haystack.core.component.types import Variadic
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

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

    gemini = GoogleAIGeminiGenerator(model="gemini-pro-vision", api_key=Secret.from_token("<MY_API_KEY>"))
    result = gemini.run(parts = ["What can you tell me about this robots?", *images])
    for answer in result["replies"]:
        print(answer)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("GOOGLE_API_KEY"),  # noqa: B008
        model: str = "gemini-pro-vision",
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Initializes a `GoogleAIGeminiGenerator` instance.

        To get an API key, visit: https://makersuite.google.com

        It supports the following models:
        * `gemini-pro`
        * `gemini-pro-vision`
        * `gemini-ultra`

        :param api_key: Google AI Studio API key.
        :param model: Name of the model to use.
        :param generation_config: The generation configuration to use.
            This can either be a `GenerationConfig` object or a dictionary of parameters.
            For available parameters, see
            [the `GenerationConfig` API reference](https://ai.google.dev/api/python/google/generativeai/GenerationConfig).
        :param safety_settings: The safety settings to use.
            A dictionary with `HarmCategory` as keys and `HarmBlockThreshold` as values.
            For more information, see [the API reference](https://ai.google.dev/api)
        :param tools: A list of Tool objects that can be used for [Function calling](https://ai.google.dev/docs/function_calling).
        """
        genai.configure(api_key=api_key.resolve_value())

        self._api_key = api_key
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
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        data = default_to_dict(
            self,
            api_key=self._api_key.to_dict(),
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
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])

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

    @component.output_types(replies=List[Union[str, Dict[str, str]]])
    def run(self, parts: Variadic[Union[str, ByteStream, Part]]):
        """
        Generates text based on the given input parts.

        :param parts:
            A heterogeneous list of strings, `ByteStream` or `Part` objects.
        :returns:
            A dictionary containing the following key:
            - `replies`: A list of strings or dictionaries with function calls.
        """

        converted_parts = [self._convert_part(p) for p in parts]

        contents = [Content(parts=converted_parts, role="user")]
        res = self._model.generate_content(
            contents=contents,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
        )
        self._model.start_chat()
        replies = []
        for candidate in res.candidates:
            for part in candidate.content.parts:
                if part.text != "":
                    replies.append(part.text)
                elif part.function_call is not None:
                    function_call = {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args.items()),
                    }
                    replies.append(function_call)

        return {"replies": replies}
