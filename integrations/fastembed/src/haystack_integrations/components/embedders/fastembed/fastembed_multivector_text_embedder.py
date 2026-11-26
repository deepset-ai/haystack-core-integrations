# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_to_dict

from fastembed import LateInteractionTextEmbedding


@component
class FastembedMultivectorTextEmbedder:
    """
    Embed text as token-level vectors using a Fastembed late-interaction model.

    The resulting multivector can be used for late-interaction retrieval or reranking against
    Documents with `multivector_embedding` populated by `FastembedMultivectorDocumentEmbedder`.
    See the [Fastembed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
    for compatible late-interaction models; the list is not exhaustive for local models.

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.fastembed import FastembedMultivectorTextEmbedder

    embedder = FastembedMultivectorTextEmbedder(model="colbert-ir/colbertv2.0")
    result = embedder.run(text="What is the capital of France?")
    print(result["multivector_embedding"])
    # [[0.12, -0.03, ...], ...]
    ```
    """

    def __init__(
        self,
        *,
        model: str = "colbert-ir/colbertv2.0",
        cache_dir: str | None = None,
        threads: int | None = None,
        prefix: str = "",
        suffix: str = "",
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a FastembedMultivectorTextEmbedder component.

        :param model: Fastembed late-interaction model name or local path.
        :param cache_dir: Model cache directory; defaults to Fastembed's cache location (configurable with
            `FASTEMBED_CACHE_PATH`).
        :param threads: Number of threads used by each ONNX Runtime session; `None` uses its default.
        :param prefix: Text to prepend to the query before embedding. Fastembed applies the model's query
            markers itself, so do not duplicate them here.
        :param suffix: Text to append to the query before embedding.
        :param local_files_only: Whether to use only cached model files.
        :param model_kwargs: Additional keyword arguments passed to `LateInteractionTextEmbedding`, such as
            `providers` for ONNX Runtime execution providers or `lazy_load`.
        """
        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.prefix = prefix
        self.suffix = suffix
        self.local_files_only = local_files_only
        self.model_kwargs = model_kwargs
        self._model: LateInteractionTextEmbedding | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component.

        :returns: A dictionary containing the component type and initialization parameters.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            cache_dir=self.cache_dir,
            threads=self.threads,
            prefix=self.prefix,
            suffix=self.suffix,
            local_files_only=self.local_files_only,
            model_kwargs=self.model_kwargs,
        )

    def warm_up(self) -> None:
        """Load the late-interaction model if it has not already been loaded."""
        if self._model is None:
            self._model = LateInteractionTextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
                local_files_only=self.local_files_only,
                **(self.model_kwargs or {}),
            )

    @component.output_types(multivector_embedding=list[list[float]])
    def run(self, text: str) -> dict[str, list[list[float]]]:
        """
        Embed a query as a token-level multivector.

        :param text: Query text to embed.
        :returns: A dictionary containing `multivector_embedding` as a list of token vectors.
        """
        self.warm_up()
        model = self._model
        assert model is not None  # noqa: S101
        embedding = next(iter(model.query_embed(self.prefix + text + self.suffix)))
        return {"multivector_embedding": embedding.tolist()}
