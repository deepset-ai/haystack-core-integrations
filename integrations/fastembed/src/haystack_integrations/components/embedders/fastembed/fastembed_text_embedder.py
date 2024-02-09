from typing import Any, Dict, List

from haystack import component, default_from_dict, default_to_dict

from .embedding_backend.fastembed_backend import _FastembedEmbeddingBackendFactory


@component
class FastembedTextEmbedder:
    """
    A component for embedding strings using fastembed embedding models.

    Usage example:
    ```python
    # To use this component, install the "fastembed" package.
    # pip install fastembed

    from fastembed_haystack.fastembed_text_embedder import FastembedTextEmbedder

    text = "It clearly says online this will work on a Mac OS system. The disk comes and it does not, only Windows. Do Not order this if you have a Mac!!"

    text_embedder = FastembedTextEmbedder(
        model="BAAI/bge-small-en-v1.5"
    )

    embedding = text_embedder.run(text)
    ```
    """  # noqa: E501

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 256,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
    ):
        """
        Create a FastembedTextEmbedder component.

        :param model: Local path or name of the model in Fastembed's model hub,
            such as ``'BAAI/bge-small-en-v1.5'``.
        :param batch_size: Number of strings to encode at once.
        :param normalize_embeddings: If set to true, returned vectors will have the length of 1.
        """

        # TODO add parallel

        self.model_name = model
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FastembedTextEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = (
                _FastembedEmbeddingBackendFactory.get_embedding_backend(
                    model_name=self.model_name
                )
            )

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            msg = (
                "FastembedTextEmbedder expects a string as input. "
                "In case you want to embed a list of Documents, please use the FastembedDocumentEmbedder."
            )
            raise TypeError(msg)
        if not hasattr(self, "embedding_backend"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        text_to_embed = [text]
        embedding = list(
            self.embedding_backend.embed(
                text_to_embed,
                batch_size=self.batch_size,
                show_progress_bar=self.progress_bar,
                normalize_embeddings=self.normalize_embeddings,
            )[0]
        )
        return {"embedding": embedding}
