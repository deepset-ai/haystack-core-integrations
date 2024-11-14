import importlib
import logging
from dataclasses import fields
from typing import Any, Dict, List, Optional

import vertexai
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from vertexai.language_models import TextGenerationModel

logger = logging.getLogger(__name__)


@component
class VertexAITextGenerator:
    """
    This component enables text generation using Google Vertex AI generative models.

    `VertexAITextGenerator` supports `text-bison`, `text-unicorn` and `text-bison-32k` models.

    Authenticates using Google Cloud Application Default Credentials (ADCs).
    For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

    Usage example:
    ```python
        from haystack_integrations.components.generators.google_vertex import VertexAITextGenerator

        generator = VertexAITextGenerator()
        res = generator.run("Tell me a good interview question for a software engineer.")

        print(res["replies"][0])

        >>> **Question:**
        >>> You are given a list of integers and a target sum.
        >>> Find all unique combinations of numbers in the list that add up to the target sum.
        >>>
        >>> **Example:**
        >>>
        >>> ```
        >>> Input: [1, 2, 3, 4, 5], target = 7
        >>> Output: [[1, 2, 4], [3, 4]]
        >>> ```
        >>>
        >>> **Follow-up:** What if the list contains duplicate numbers?
    ```
    """

    def __init__(
        self, *, model: str = "text-bison", project_id: Optional[str] = None, location: Optional[str] = None, **kwargs
    ):
        """
        Generate text using a Google Vertex AI model.

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

        self._model = TextGenerationModel.from_pretrained(self._model_name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        data = default_to_dict(
            self, model=self._model_name, project_id=self._project_id, location=self._location, **self._kwargs
        )

        if (grounding_source := data["init_parameters"].get("grounding_source")) is not None:
            # Handle the grounding source dataclasses
            class_type = f"{grounding_source.__module__}.{grounding_source.__class__.__name__}"
            init_fields = {f.name: getattr(grounding_source, f.name) for f in fields(grounding_source) if f.init}
            data["init_parameters"]["grounding_source"] = {
                "type": class_type,
                "init_parameters": init_fields,
            }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAITextGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
           Deserialized component.
        """
        if (grounding_source := data["init_parameters"].get("grounding_source")) is not None:
            module_name, class_name = grounding_source["type"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            data["init_parameters"]["grounding_source"] = getattr(module, class_name)(
                **grounding_source["init_parameters"]
            )
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], safety_attributes=Dict[str, float], citations=List[Dict[str, Any]])
    def run(self, prompt: str):
        """Prompts the model to generate text.

        :param prompt: The prompt to use for text generation.
        :returns:  A dictionary with the following keys:
            - `replies`: A list of generated replies.
            - `safety_attributes`: A dictionary with the [safety scores](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/responsible-ai#safety_attribute_descriptions)
              of each answer.
            - `citations`: A list of citations for each answer.
        """
        res = self._model.predict(prompt=prompt, **self._kwargs)

        replies = []
        safety_attributes = []
        citations = []

        for prediction in res.raw_prediction_response.predictions:
            replies.append(prediction["content"])
            safety_attributes.append(prediction["safetyAttributes"])
            citations.append(prediction["citationMetadata"]["citations"])

        return {"replies": replies, "safety_attributes": safety_attributes, "citations": citations}
