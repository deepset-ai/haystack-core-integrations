# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any

from haystack import Document, component, default_to_dict
from tqdm import tqdm

from fastembed import LateInteractionTextEmbedding


@component
class FastembedMultivectorDocumentEmbedder:
    """
    Embed Documents as token-level vectors using a Fastembed late-interaction model.

    The returned copies contain the vectors in `multivector_embedding`; original Documents and their
    dense `embedding` fields are unchanged. Use a document store supporting multivectors to index them.
    See the [Fastembed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
    for compatible late-interaction models; the list is not exhaustive for local models.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.fastembed import FastembedMultivectorDocumentEmbedder

    documents = [Document(content="Paris is the capital of France.")]
    embedder = FastembedMultivectorDocumentEmbedder(model="colbert-ir/colbertv2.0")
    result = embedder.run(documents=documents)
    print(result["documents"][0].multivector_embedding)
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
        batch_size: int = 256,
        progress_bar: bool = True,
        parallel: int | None = None,
        local_files_only: bool = False,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a FastembedMultivectorDocumentEmbedder component.

        :param model: Fastembed late-interaction model name or local path.
        :param cache_dir: Model cache directory; defaults to Fastembed's cache location (configurable with
            `FASTEMBED_CACHE_PATH`).
        :param threads: Number of threads used by each ONNX Runtime session; `None` uses its default.
        :param prefix: Text to prepend to each Document before embedding. Fastembed applies the model's document
            markers itself, so do not duplicate them here.
        :param suffix: Text to append to each Document before embedding.
        :param batch_size: Positive number of Documents to encode per batch.
        :param progress_bar: Whether to display embedding progress.
        :param parallel: Number of data-parallel workers; `0` uses all available cores, and `None` uses
            ONNX Runtime's default threading.
        :param local_files_only: Whether to use only cached model files.
        :param meta_fields_to_embed: Document metadata fields to prepend to the content, in the given order.
        :param embedding_separator: Separator joining the metadata values and Document content.
        :param model_kwargs: Additional keyword arguments passed to `LateInteractionTextEmbedding`, such as
            `providers` for ONNX Runtime execution providers or `lazy_load`.
        :raises ValueError: `batch_size` is not positive.
        """
        if batch_size <= 0:
            msg = "batch_size must be greater than zero"
            raise ValueError(msg)
        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.parallel = parallel
        self.local_files_only = local_files_only
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
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
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            parallel=self.parallel,
            local_files_only=self.local_files_only,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
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

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Embed Documents as token-level multivectors.

        :param documents: Documents to embed.
        :returns: A dictionary containing copied `documents` with their `multivector_embedding` fields populated.
        """
        if not documents:
            return {"documents": []}

        self.warm_up()
        model = self._model
        assert model is not None  # noqa: S101

        texts = []
        for doc in documents:
            metadata = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            texts.append(self.prefix + self.embedding_separator.join([*metadata, doc.content or ""]) + self.suffix)

        embeddings = model.embed(texts, batch_size=self.batch_size, parallel=self.parallel)
        embedded = [
            replace(doc, multivector_embedding=embedding.tolist())
            for doc, embedding in zip(
                documents,
                tqdm(embeddings, total=len(documents), disable=not self.progress_bar, desc="Calculating embeddings"),
                strict=True,
            )
        ]
        return {"documents": embedded}
