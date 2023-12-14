import logging
from typing import Any, Dict, List, Optional, Union

import vertexai
from haystack.core.component import component
from haystack.core.component.types import Variadic
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream
from vertexai.preview.generative_models import (
    Content,
    GenerativeModel,
    Part,
)

logger = logging.getLogger(__name__)


@component
class GeminiGenerator:
    def __init__(self, *, model: str = "gemini-pro-vision", project_id: str, location: Optional[str] = None, **kwargs):
        """
        Multi modal generator using Gemini model via Google Vertex AI.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official Google documentation:
        https://cloud.google.com/docs/authentication/provide-credentials-adc

        :param project_id: ID of the GCP project to use.
        :param model: Name of the model to use, defaults to "gemini-pro-vision".
        :param location: The default location to use when making API calls, if not set uses us-central-1.
            Defaults to None.
        :param kwargs: Additional keyword arguments to pass to the model.
            For a list of supported arguments see the `GenerativeModel.generate_content()` documentation.
        """

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        vertexai.init(project=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._location = location
        self._kwargs = kwargs

        if kwargs.get("stream"):
            msg = "The `stream` parameter is not supported by the Gemini generator."
            raise ValueError(msg)

        self._model = GenerativeModel(self._model_name)

    def to_dict(self) -> Dict[str, Any]:
        # TODO: This is not fully implemented yet
        return default_to_dict(
            self, model=self._model_name, project_id=self._project_id, location=self._location, **self._kwargs
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiGenerator":
        # TODO: This is not fully implemented yet
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

    @component.output_types(answers=List[Union[str, Dict[str, str]]])
    def run(self, parts: Variadic[List[Union[str, ByteStream, Part]]]):
        converted_parts = [self._convert_part(p) for p in parts]

        contents = [Content(parts=converted_parts, role="user")]
        res = self._model.generate_content(contents=contents, **self._kwargs)
        self._model.start_chat()
        answers = []
        for candidate in res.candidates:
            for part in candidate.content.parts:
                if part._raw_part.text != "":
                    answers.append(part.text)
                elif part.function_call is not None:
                    function_call = {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args.items()),
                    }
                    answers.append(function_call)

        return {"answers": answers}


# generator = GeminiGenerator(project_id="infinite-byte-223810")
# res = generator.run(["What can you do for me?"])
# res
# another_res = generator.run(["Can you solve this math problems?", "2 + 2", "3 + 3", "1 / 1"])
# another_res["answers"]
# from pathlib import Path

# image = ByteStream.from_file_path(
#     Path("/Users/silvanocerza/Downloads/photo_2023-11-07_11-45-42.jpg"), mime_type="image/jpeg"
# )
# res = generator.run(["What is this about?", image])
# res["answers"]
