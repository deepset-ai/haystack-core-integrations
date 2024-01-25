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

logger = logging.getLogger(__name__)


@component
class GoogleAIGeminiGenerator:
    """
    GoogleAIGeminiGenerator is a multi modal generator supporting Gemini via Google Makersuite.

    Sample usage:
    ```python
    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

    gemini = GoogleAIGeminiGenerator(model="gemini-pro", api_key="<MY_API_KEY>")
    res = gemini.run(parts = ["What is the most interesting thing you know?"])
    for answer in res["answers"]:
        print(answer)
    ```

    This is a more advanced usage that also uses text and images as input:
    ```python
    import requests
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

    gemini = GoogleAIGeminiGenerator(model="gemini-pro-vision", api_key="<MY_API_KEY>")
    result = gemini.run(parts = ["What can you tell me about this robots?", *images])
    for answer in result["answers"]:
        print(answer)
    ```

    Input:
    - **parts** A eterogeneous list of strings, ByteStream or Part objects.

    Output:
    - **answers** A list of strings or dictionaries with function calls.
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
        Initialize a GoogleAIGeminiGenerator instance.
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
            Accepted parameters are:
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
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiGenerator":
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

    @component.output_types(answers=List[Union[str, Dict[str, str]]])
    def run(self, parts: Variadic[Union[str, ByteStream, Part]]):
        converted_parts = [self._convert_part(p) for p in parts]

        contents = [Content(parts=converted_parts, role="user")]
        res = self._model.generate_content(
            contents=contents,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
        )
        self._model.start_chat()
        answers = []
        for candidate in res.candidates:
            for part in candidate.content.parts:
                if part.text != "":
                    answers.append(part.text)
                elif part.function_call is not None:
                    function_call = {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args.items()),
                    }
                    answers.append(function_call)

        return {"answers": answers}
