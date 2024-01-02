import logging
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
from haystack.core.component import component
from haystack.core.component.types import Variadic
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from google.ai.generativelanguage import FunctionDeclaration, Tool, Part, Content

logger = logging.getLogger(__name__)


@component
class GoogleAIGeminiGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-pro-vision",
        project_id: str,
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Multi modal generator using Gemini model via Makersuite
        """

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        genai.configure(api_key=api_key)

        self._model_name = model
        self._project_id = project_id

        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._model = GenerativeModel(self._model_name, tools=self._tools)

    def _function_to_dict(self, function: FunctionDeclaration) -> Dict[str, Any]:
        return {
            "name": function.name,
            "parameters": function.parameters,
            "description": function.description,
        }

    def _tool_to_dict(self, tool: Tool) -> Dict[str, Any]:
        return {
            "function_declarations": [self._function_to_dict(f) for f in tool.function_declarations],
        }

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
            project_id=self._project_id,
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
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiGenerator":
        if (tools := data["init_parameters"].get("tools")) is not None:
            data["init_parameters"]["tools"] = [Tool(t) for t in tools]
        if (generation_config := data["init_parameters"].get("generation_config")) is not None:
            data["init_parameters"]["generation_config"] = GenerationConfig.from_dict(generation_config)

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
