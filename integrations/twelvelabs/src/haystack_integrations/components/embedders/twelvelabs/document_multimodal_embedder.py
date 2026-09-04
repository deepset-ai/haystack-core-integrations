# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm

from ._media_embed import MODALITIES, detect_modality, embed_media, embed_media_async

DEFAULT_MODEL = "marengo3.0"


@component
class TwelveLabsDocumentMultimodalEmbedder:
    """
    Embeds the media referenced by Documents using TwelveLabs Marengo.

    Each Document points at an image, audio, or video file (or URL) through its metadata
    (`meta["file_path"]` by default). This component computes a Marengo embedding for that
    media and stores it on `Document.embedding`. Because Marengo embeds every modality into
    one shared space, the resulting embeddings support cross-modal retrieval — for example,
    search a store of video embeddings with a text query embedded by `TwelveLabsTextEmbedder`.

    This is the media counterpart to `TwelveLabsDocumentEmbedder`, which embeds a Document's
    text `content`.

    The modality of each Document is inferred from its file extension / MIME type; set
    `meta["modality"]` (or the field named by `modality_meta_field`) to override it.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.twelvelabs import TwelveLabsDocumentMultimodalEmbedder

    # Set the TWELVELABS_API_KEY environment variable
    embedder = TwelveLabsDocumentMultimodalEmbedder()
    docs = [Document(meta={"file_path": "cat.jpg"})]
    docs = embedder.run(documents=docs)["documents"]
    print(docs[0].embedding)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),
        model: str = DEFAULT_MODEL,
        file_path_meta_field: str = "file_path",
        root_path: str = "",
        modality_meta_field: str = "modality",
        batch_size: int = 32,
        progress_bar: bool = True,
    ) -> None:
        """
        Create a TwelveLabsDocumentMultimodalEmbedder.

        :param api_key: The TwelveLabs API key. Read from the `TWELVELABS_API_KEY`
            environment variable by default.
        :param model: The Marengo model name.
        :param file_path_meta_field: The metadata field on each Document that holds the media's
            local file path or URL.
        :param root_path: An absolute path prepended to each Document's relative `file_path`
            (ignored for URLs). When set, it also acts as a sandbox: paths that resolve outside
            `root_path` (via `..` segments or an absolute `file_path`) are rejected.
        :param modality_meta_field: The metadata field that, when present, overrides the inferred
            modality (`"image"`, `"audio"` or `"video"`).
        :param batch_size: Number of Documents per batch; within a batch `run_async` embeds concurrently.
        :param progress_bar: Whether to show a progress bar while embedding. Can be helpful
            to disable in production deployments to keep the logs clean.
        """
        self.api_key = api_key
        self.model = model
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path
        self.modality_meta_field = modality_meta_field
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    def _get_telemetry_data(self) -> dict[str, Any]:
        """Data sent to Posthog for usage analytics."""
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
            modality_meta_field=self.modality_meta_field,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TwelveLabsDocumentMultimodalEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _validate(self, documents: list[Document]) -> None:
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            msg = (
                "TwelveLabsDocumentMultimodalEmbedder expects a list of Documents. To embed a "
                "single media source, use the TwelveLabsMultimodalEmbedder."
            )
            raise TypeError(msg)

    def _resolve_source(self, document: Document) -> str:
        source = document.meta.get(self.file_path_meta_field)
        if not isinstance(source, str) or not source:
            msg = (
                f"Document '{document.id}' has no media path in meta['{self.file_path_meta_field}']. "
                "Populate it with a local file path or a publicly accessible URL."
            )
            raise ValueError(msg)
        if urlparse(source).scheme in ("http", "https"):
            return source
        if not self.root_path:
            return source
        # Treat root_path as a sandbox: reject anything that resolves outside it
        # (absolute file_path values or `..` traversal).
        root = Path(self.root_path).expanduser().resolve()
        resolved = (root / source).resolve()
        if not resolved.is_relative_to(root):
            msg = f"Media path for document '{document.id}' resolves to '{resolved}', outside root_path '{root}'."
            raise ValueError(msg)
        return str(resolved)

    def _resolve_modality(self, document: Document, source: str) -> str:
        override = document.meta.get(self.modality_meta_field)
        if override is not None and not isinstance(override, str):
            msg = (
                f"meta['{self.modality_meta_field}'] for document '{document.id}' must be a string, "
                f"got {type(override).__name__}."
            )
            raise ValueError(msg)
        resolved = (override or detect_modality(source)).lower()
        if resolved not in MODALITIES:
            msg = f"Unsupported modality {override!r} for document '{document.id}'. Expected one of {MODALITIES}."
            raise ValueError(msg)
        return resolved

    def _embedded_document(self, document: Document, embedding: list[float], modality: str) -> Document:
        return replace(
            document,
            embedding=embedding,
            meta={
                **document.meta,
                "embedding_source": {"type": modality, "file_path_meta_field": self.file_path_meta_field},
            },
        )

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embed the media referenced by a list of Documents.

        :param documents: The Documents to embed. Each must reference its media through
            `meta[file_path_meta_field]` (a local file path or a URL).
        :returns: A dictionary with keys:
            - `documents`: New Documents that are copies of the inputs with `embedding` populated
              and an `embedding_source` entry added to their metadata.
            - `meta`: Metadata about the request (the model used).
        :raises TypeError: If the input is not a list of Documents.
        :raises ValueError: If a Document has no media path or an unsupported modality.
        """
        self._validate(documents)
        key = self.api_key.resolve_value() or ""
        embedded: list[Document] = []
        for document in tqdm(documents, disable=not self.progress_bar, desc="Calculating embeddings"):
            source = self._resolve_source(document)
            modality = self._resolve_modality(document, source)
            embedding = embed_media(source, modality, self.model, key)
            embedded.append(self._embedded_document(document, embedding, modality))
        return {"documents": embedded, "meta": {"model": self.model}}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Asynchronously embed the media referenced by a list of Documents.

        Documents within each batch of `batch_size` are embedded concurrently.

        :param documents: The Documents to embed.
        :returns: A dictionary with keys `documents` (copies with `embedding` populated) and `meta`.
        :raises TypeError: If the input is not a list of Documents.
        :raises ValueError: If a Document has no media path or an unsupported modality.
        """
        self._validate(documents)
        key = self.api_key.resolve_value() or ""
        sources = [self._resolve_source(document) for document in documents]
        modalities = [
            self._resolve_modality(document, source) for document, source in zip(documents, sources, strict=True)
        ]
        embeddings: list[list[float]] = []
        for i in tqdm(
            range(0, len(documents), self.batch_size),
            disable=not self.progress_bar,
            desc="Calculating embeddings",
        ):
            batch = list(zip(sources[i : i + self.batch_size], modalities[i : i + self.batch_size], strict=True))
            batch_embeddings = await asyncio.gather(
                *(embed_media_async(source, modality, self.model, key) for source, modality in batch)
            )
            embeddings.extend(batch_embeddings)
        embedded = [
            self._embedded_document(document, embedding, modality)
            for document, embedding, modality in zip(documents, embeddings, modalities, strict=True)
        ]
        return {"documents": embedded, "meta": {"model": self.model}}
