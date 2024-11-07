import logging
from typing import Any, Dict, List, Optional

import vertexai
from haystack.core.component import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.byte_stream import ByteStream
from vertexai.vision_models import ImageGenerationModel

logger = logging.getLogger(__name__)


@component
class VertexAIImageGenerator:
    """
    This component enables image generation using Google Vertex AI generative model.

    Authenticates using Google Cloud Application Default Credentials (ADCs).
    For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

    Usage example:
    ```python
    from pathlib import Path

    from haystack_integrations.components.generators.google_vertex import VertexAIImageGenerator

    generator = VertexAIImageGenerator()
    result = generator.run(prompt="Generate an image of a cute cat")
    result["images"][0].to_file(Path("my_image.png"))
    ```
    """

    def __init__(
        self,
        *,
        model: str = "imagegeneration",
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs,
    ):
        """
        Generates images using a Google Vertex AI model.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

        :param project_id: ID of the GCP project to use. By default, it is set during Google Cloud authentication.
        :param model: Name of the model to use.
        :param location: The default location to use when making API calls, if not set uses us-central-1.
        :param kwargs: Additional keyword arguments to pass to the model.
            For a list of supported arguments see the `ImageGenerationModel.generate_images()` documentation.
        """

        # Login to GCP. This will fail if user has not set up their gcloud SDK
        vertexai.init(project=project_id, location=location)

        self._model_name = model
        self._project_id = project_id
        self._location = location
        self._kwargs = kwargs

        self._model = ImageGenerationModel.from_pretrained(self._model_name)

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
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAIImageGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
           Deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(images=List[ByteStream])
    def run(self, prompt: str, negative_prompt: Optional[str] = None):
        """Produces images based on the given prompt.

        :param prompt: The prompt to generate images from.
        :param negative_prompt: A description of what you want to omit in
                the generated images.
        :returns: A dictionary with the following keys:
            - `images`: A list of ByteStream objects, each containing an image.
        """
        negative_prompt = negative_prompt or self._kwargs.get("negative_prompt")
        res = self._model.generate_images(prompt=prompt, negative_prompt=negative_prompt, **self._kwargs)
        images = [ByteStream(data=i._image_bytes, meta=i.generation_parameters) for i in res.images]
        return {"images": images}
