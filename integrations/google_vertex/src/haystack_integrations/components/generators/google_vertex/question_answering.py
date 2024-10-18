import logging
from typing import Any, Dict, List, Optional

import vertexai
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream
from vertexai.vision_models import Image, ImageTextModel

logger = logging.getLogger(__name__)


@component
class VertexAIImageQA:
    """
    This component enables text generation (image captioning) using Google Vertex AI generative models.

    Authenticates using Google Cloud Application Default Credentials (ADCs).
    For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

    Usage example:
    ```python
    from haystack.dataclasses.byte_stream import ByteStream
    from haystack_integrations.components.generators.google_vertex import VertexAIImageQA

    qa = VertexAIImageQA()

    image = ByteStream.from_file_path("dog.jpg")

    res = qa.run(image=image, question="What color is this dog")

    print(res["replies"][0])

    >>> white
    ```
    """

    def __init__(
        self, *, model: str = "imagetext", project_id: Optional[str] = None, location: Optional[str] = None, **kwargs
    ):
        """
        Answers questions about an image using a Google Vertex AI model.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

        :param project_id: ID of the GCP project to use. By default, it is set during Google Cloud authentication.
        :param model: Name of the model to use.
        :param location: The default location to use when making API calls, if not set uses us-central-1.
        :param kwargs: Additional keyword arguments to pass to the model.
            For a list of supported arguments see the `ImageTextModel.ask_question()` documentation.
        """

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        vertexai.init(project=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._location = location
        self._kwargs = kwargs

        self._model = ImageTextModel.from_pretrained(self._model_name)

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
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAIImageQA":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
           Deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str])
    def run(self, image: ByteStream, question: str):
        """Prompts model to answer a question about an image.

        :param image: The image to ask the question about.
        :param question: The question to ask.
        :returns: A dictionary with the following keys:
            - `replies`: A list of answers to the question.
        """
        replies = self._model.ask_question(image=Image(image.data), question=question, **self._kwargs)
        return {"replies": replies}
