# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_to_dict

from .embedding_backend.fastembed_backend import _FastembedEmbeddingBackend, _FastembedEmbeddingBackendFactory


@component
class FastembedTextEmbedder:
    """
    FastembedTextEmbedder computes string embedding using fastembed embedding models.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder

    text = ("It clearly says online this will work on a Mac OS system. "
            "The disk comes and it does not, only Windows. Do Not order this if you have a Mac!!")

    text_embedder = FastembedTextEmbedder(
        model="BAAI/bge-small-en-v1.5"
    )

    embedding = text_embedder.run(text)["embedding"]
    ```
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
        threads: int | None = None,
        prefix: str = "",
        suffix: str = "",
        progress_bar: bool = True,
        parallel: int | None = None,
        local_files_only: bool = False,
    ) -> None:
        """
        Create a FastembedTextEmbedder component.

        :param model: Local path or name of the model in Fastembed's model hub, such as `BAAI/bge-small-en-v1.5`
        :param cache_dir: The path to the cache directory.
                Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                Defaults to `fastembed_cache` in the system's temp directory.
        :param threads: The number of threads single onnxruntime session can use. Defaults to None.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param progress_bar: If `True`, displays progress bar during embedding.
        :param parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
        :param local_files_only: If `True`, only use the model files in the `cache_dir`.
        """

        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.prefix = prefix
        self.suffix = suffix
        self.progress_bar = progress_bar
        self.parallel = parallel
        self.local_files_only = local_files_only
        self.embedding_backend: _FastembedEmbeddingBackend | None = None

    def to_dict(self) -> dict[str, Any]:
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
            progress_bar=self.progress_bar,
            parallel=self.parallel,
            local_files_only=self.local_files_only,
        )

    def warm_up(self) -> None:
        """
        Initializes the component.
        """
        if self.embedding_backend is None:
            self.embedding_backend = _FastembedEmbeddingBackendFactory.get_embedding_backend(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
                local_files_only=self.local_files_only,
            )

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        """
        Embeds text using the Fastembed model.

        :param text: A string to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: A list of floats representing the embedding of the input text.
        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            msg = (
                "FastembedTextEmbedder expects a string as input. "
                "In case you want to embed a list of Documents, please use the FastembedDocumentEmbedder."
            )
            raise TypeError(msg)

        if self.embedding_backend is None:
            self.warm_up()

        text_to_embed = [self.prefix + text + self.suffix]
        embedding = list(
            self.embedding_backend.embed(  # type: ignore[union-attr]
                text_to_embed,
                progress_bar=self.progress_bar,
                parallel=self.parallel,
            )[0]
        )
        return {"embedding": embedding}
