import logging
import os
from typing import Any, Dict, List, Optional

from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from vertexai.language_models import CodeGenerationModel

from google_vertex_haystack.generators.utils import authenticate

logger = logging.getLogger(__name__)


@component
class VertexAICodeGenerator:
    def __init__(
        self, *, model: str = "code-bison", project_id: str, api_key: str = "", location: Optional[str] = None, **kwargs
    ):
        """
        Generate code using a Google Vertex AI model.


        :param project_id: ID of the GCP project to use.
        :param api_key: API key to use for authentication, if not set uses `GOOGLE_API_KEY` environment variable.
            If neither are set, will attempt to use Application Default Credentials (ADCs).
            For more information on ADC see the official Google documentation:
            https://cloud.google.com/docs/authentication/provide-credentials-adc
        :param model: Name of the model to use, defaults to "text-bison".
        :param location: The default location to use when making API calls, if not set uses us-central-1.
            Defaults to None.
        :param kwargs: Additional keyword arguments to pass to the model.
            For a list of supported arguments see the `TextGenerationModel.predict()` documentation.
        """

        authenticate(api_key=api_key, project_id=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._api_key = api_key
        self._location = location
        self._kwargs = kwargs

        self._model = CodeGenerationModel.from_pretrained(self._model_name)

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
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAICodeGenerator":
        if (api_key := data["init_parameters"].get("api_key")) in os.environ:
            data["init_parameters"]["api_key"] = os.environ[api_key]
        return default_from_dict(cls, data)

    @component.output_types(answers=List[str])
    def run(self, prefix: str, suffix: Optional[str] = None):
        res = self._model.predict(prefix=prefix, suffix=suffix, **self._kwargs)
        # Handle the case where the model returns multiple candidates
        answers = [c.text for c in res.candidates] if hasattr(res, "candidates") else [res.text]
        return {"answers": answers}
