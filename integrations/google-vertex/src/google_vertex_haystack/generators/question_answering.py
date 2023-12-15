import logging
import os
from typing import Any, Dict, List, Optional

from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream
from vertexai.vision_models import Image, ImageTextModel

from google_vertex_haystack.generators.utils import authenticate

logger = logging.getLogger(__name__)


@component
class VertexAIImageQA:
    def __init__(
        self, *, model: str = "imagetext", project_id: str, api_key: str = "", location: Optional[str] = None, **kwargs
    ):
        """
        Answers questions about an image using a Google Vertex AI model.

        :param project_id: ID of the GCP project to use.
        :param api_key: API key to use for authentication, if not set uses `GOOGLE_API_KEY` environment variable.
            If neither are set, will attempt to use Application Default Credentials (ADCs).
            For more information on ADC see the official Google documentation:
            https://cloud.google.com/docs/authentication/provide-credentials-adc
        :param model: Name of the model to use, defaults to "imagetext".
        :param location: The default location to use when making API calls, if not set uses us-central-1.
            Defaults to None.
        :param kwargs: Additional keyword arguments to pass to the model.
            For a list of supported arguments see the `ImageTextModel.ask_question()` documentation.
        """

        authenticate(api_key=api_key, project_id=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._api_key = api_key
        self._location = location
        self._kwargs = kwargs

        self._model = ImageTextModel.from_pretrained(self._model_name)

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(
            self,
            model=self._model_name,
            project_id=self._project_id,
            api_key=self._api_key,
            location=self._location,
            **self._kwargs,
        )
        if data["init_parameters"].get("api_key"):
            data["init_parameters"]["api_key"] = "GOOGLE_API_KEY"
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAIImageQA":
        if (api_key := data["init_parameters"].get("api_key")) in os.environ:
            data["init_parameters"]["api_key"] = os.environ[api_key]
        return default_from_dict(cls, data)

    @component.output_types(answers=List[str])
    def run(self, image: ByteStream, question: str):
        answers = self._model.ask_question(image=Image(image.data), question=question, **self._kwargs)
        return {"answers": answers}
