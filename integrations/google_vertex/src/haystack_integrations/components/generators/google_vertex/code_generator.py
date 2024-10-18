import logging
from typing import Any, Dict, List, Optional

import vertexai
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from vertexai.language_models import CodeGenerationModel

logger = logging.getLogger(__name__)


@component
class VertexAICodeGenerator:
    """
    This component enables code generation using Google Vertex AI generative model.

    `VertexAICodeGenerator` supports `code-bison`, `code-bison-32k`, and `code-gecko`.

    Usage example:
    ```python
        from haystack_integrations.components.generators.google_vertex import VertexAICodeGenerator

        generator = VertexAICodeGenerator()

        result = generator.run(prefix="def to_json(data):")

        for answer in result["replies"]:
            print(answer)

        >>> ```python
        >>> import json
        >>>
        >>> def to_json(data):
        >>>   \"\"\"Converts a Python object to a JSON string.
        >>>
        >>>   Args:
        >>>     data: The Python object to convert.
        >>>
        >>>   Returns:
        >>>     A JSON string representing the Python object.
        >>>   \"\"\"
        >>>
        >>>   return json.dumps(data)
        >>> ```
    ```
    """

    def __init__(
        self, *, model: str = "code-bison", project_id: Optional[str] = None, location: Optional[str] = None, **kwargs
    ):
        """
        Generate code using a Google Vertex AI model.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

        :param project_id: ID of the GCP project to use. By default, it is set during Google Cloud authentication.
        :param model: Name of the model to use.
        :param location: The default location to use when making API calls, if not set uses us-central-1.
        :param kwargs: Additional keyword arguments to pass to the model.
            For a list of supported arguments see the `TextGenerationModel.predict()` documentation.
        """

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        vertexai.init(project=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._location = location
        self._kwargs = kwargs

        self._model = CodeGenerationModel.from_pretrained(self._model_name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self, model=self._model_name, project_id=self._project_id, location=self._location, **self._kwargs
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAICodeGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
           Deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str])
    def run(self, prefix: str, suffix: Optional[str] = None):
        """
        Generate code using a Google Vertex AI model.

        :param prefix: Code before the current point.
        :param suffix: Code after the current point.
        :returns: A dictionary with the following keys:
            - `replies`: A list of generated code snippets.
        """
        res = self._model.predict(prefix=prefix, suffix=suffix, **self._kwargs)
        # Handle the case where the model returns multiple candidates
        replies = [c.text for c in res.candidates] if hasattr(res, "candidates") else [res.text]
        return {"replies": replies}
