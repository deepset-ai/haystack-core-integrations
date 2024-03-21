from typing import Any, Dict, List, Optional, Union

from haystack import component, default_to_dict

from .embedding_backend.fastembed_backend import _FastembedSparseEmbeddingBackendFactory


@component
class FastembedSparseTextEmbedder:
    """
    FastembedSparseTextEmbedder computes string embedding using fastembed sparse models.

    Usage example:
    ```python
    # To use this component, install the "fastembed-haystack" package.
    # pip install fastembed-haystack

    from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

    text = "It clearly says online this will work on a Mac OS system. The disk comes and it does not, only Windows. Do Not order this if you have a Mac!!"

    text_embedder = FastembedSparseTextEmbedder(
        model="prithvida/Splade_PP_en_v1"
    )
    text_embedder.warm_up()

    embedding = text_embedder.run(text)["embedding"]
    ```
    """  # noqa: E501

    def __init__(
        self,
        model: str = "prithvida/Splade_PP_en_v1",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 256,
        progress_bar: bool = True,
        parallel: Optional[int] = None,
    ):
        """
        Create a FastembedSparseTextEmbedder component.

        :param model: Local path or name of the model in Fastembed's model hub, such as `prithvida/Splade_PP_en_v1`
        :param cache_dir: The path to the cache directory.
                Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                Defaults to `fastembed_cache` in the system's temp directory.
        :param threads: The number of threads single onnxruntime session can use. Defaults to None.
        :param batch_size: Number of strings to encode at once.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param progress_bar: If true, displays progress bar during embedding.
        :param parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
        """

        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.parallel = parallel

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            cache_dir=self.cache_dir,
            threads=self.threads,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            parallel=self.parallel,
        )

    def warm_up(self):
        """
        Initializes the component.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _FastembedSparseEmbeddingBackendFactory.get_embedding_backend(
                model_name=self.model_name, cache_dir=self.cache_dir, threads=self.threads
            )

    @component.output_types(sparse_embedding=Dict[str, Union[List[int], List[float]]])
    def run(self, text: str):
        """
        Embeds text using the Fastembed model.

        :param text: A string to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: A list of floats representing the embedding of the input text.
        :raises TypeError: If the input is not a string.
        :raises RuntimeError: If the embedding model has not been loaded.
        """
        if not isinstance(text, str):
            msg = (
                "FastembedSparseTextEmbedder expects a string as input. "
                "In case you want to embed a list of Documents, please use the FastembedDocumentEmbedder."
            )
            raise TypeError(msg)
        if not hasattr(self, "embedding_backend"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        text_to_embed = [self.prefix + text + self.suffix]
        embedding = self.embedding_backend.embed(
            text_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            parallel=self.parallel,
        )[0]
        return {"embedding": embedding}
