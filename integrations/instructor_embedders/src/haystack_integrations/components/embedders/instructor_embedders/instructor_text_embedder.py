# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace

from .embedding_backend.instructor_backend import _InstructorEmbeddingBackendFactory


@component
class InstructorTextEmbedder:
    """
    A component for embedding strings using INSTRUCTOR embedding models.

    Usage example:
    ```python
    # To use this component, install the "instructor-embedders-haystack" package.
    # pip install instructor-embedders-haystack

    from haystack.utils.device import ComponentDevice
    from haystack_integrations.components.embedders.instructor_embedders import InstructorTextEmbedder

    text = ("It clearly says online this will work on a Mac OS system. The disk comes and it does not, only Windows.
            "Do Not order this if you have a Mac!!")
    instruction = (
        "Represent the Amazon comment for classifying the sentence as positive or negative"
    )

    text_embedder = InstructorTextEmbedder(
        model="hkunlp/instructor-base", instruction=instruction,
        device=ComponentDevice.from_str("cpu")
    )
    text_embedder.warm_up()

    embedding = text_embedder.run(text)
    ```
    """

    def __init__(
        self,
        model: str = "hkunlp/instructor-base",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),  # noqa: B008
        instruction: str = "Represent the sentence",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
    ):
        """
        Create an InstructorTextEmbedder component.

        :param model: Local path or name of the model in Hugging Face's model hub,
            such as ``'hkunlp/instructor-base'``.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected.
        :param token: The API token used to download private models from Hugging Face.
        :param instruction: The instruction string to be used while computing domain-specific embeddings.
            The instruction follows the unified template of the form:
            "Represent the 'domain' 'text_type' for 'task_objective'", where:
            - "domain" is optional, and it specifies the domain of the text, e.g., science, finance, medicine, etc.
            - "text_type" is required, and it specifies the encoding unit, e.g., sentence, document, paragraph, etc.
            - "task_objective" is optional, and it specifies the objective of embedding, e.g., retrieve a document,
            classify the sentence, etc.
            Check some examples of instructions [here](https://github.com/xlang-ai/instructor-embedding#use-cases).
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true, displays progress bar during embedding.
        :param normalize_embeddings: If set to true, returned vectors will have the length of 1.
        """

        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.instruction = instruction
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            instruction=self.instruction,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructorTextEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        serialized_device = data["init_parameters"]["device"]
        data["init_parameters"]["device"] = ComponentDevice.from_dict(serialized_device)

        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
                model=self.model, device=self.device.to_torch_str(), token=self.token
            )

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            msg = (
                "InstructorTextEmbedder expects a string as input. "
                "In case you want to embed a list of Documents, please use the InstructorDocumentEmbedder."
            )
            raise TypeError(msg)
        if not hasattr(self, "embedding_backend"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        text_to_embed = [self.instruction, text]
        embedding = self.embedding_backend.embed(
            [text_to_embed],
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )[0]
        return {"embedding": embedding}
