from enum import Enum


class Client(Enum):
    """
    Client to use for NVIDIA NIMs.
    """

    NVIDIA_GENERATOR = "nvidia_generator"
    NVIDIA_TEXT_EMBEDDER = "nvidia_text_embedder"
    NVIDIA_DOCUMENT_EMBEDDER = "nvidia_document_embedder"
    NVIDIA_RANKER = "nvidia_ranker"

    def __str__(self):
        return self.value
