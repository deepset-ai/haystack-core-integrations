from typing import Any, Dict, List, Optional

from haystack import component, default_to_dict

from .embedding_backend.fastembed_backend import _FastembedEmbeddingBackendFactory


@component
class FastembedTextEmbedder:
    """
    A component for embedding strings using fastembed embedding models.

    Usage example:
    ```python
    # To use this component, install the "fastembed-haystack" package.
    # pip install fastembed-haystack

    from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder

    text = "It clearly says online this will work on a Mac OS system. The disk comes and it does not, only Windows. Do Not order this if you have a Mac!!"

    text_embedder = FastembedTextEmbedder(
        model="BAAI/bge-small-en-v1.5"
    )
    text_embedder.warm_up()

    embedding = text_embedder.run(text)["embedding"]
    ```
    """  # noqa: E501

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 256,
        progress_bar: bool = True,
        parallel: Optional[int] = None,
    ):
        """
        Create a FastembedTextEmbedder component.

        :param model: Local path or name of the model in Fastembed's model hub,
            such as ``'BAAI/bge-small-en-v1.5'``.
        :param batch_size: Number of strings to encode at once.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param progress_bar: If true, displays progress bar during embedding.
        :param parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
        """

        # TODO add parallel

        self.model_name = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.parallel = parallel

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            parallel=self.parallel,
        )

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _FastembedEmbeddingBackendFactory.get_embedding_backend(model_name=self.model_name)

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

        text_to_embed = [self.prefix + text + self.suffix]
        embedding = list(
            self.embedding_backend.embed(
                text_to_embed,
                batch_size=self.batch_size,
                show_progress_bar=self.progress_bar,
                parallel=self.parallel,
            )[0]
        )
        return {"embedding": embedding}
