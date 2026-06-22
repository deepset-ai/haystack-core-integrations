# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import replace
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm

from ._embed import embed_text, embed_text_async

DEFAULT_MODEL = "marengo3.0"


@component
class TwelveLabsDocumentEmbedder:
    """
    Embeds the text content of Documents using TwelveLabs Marengo.

    Computes a Marengo embedding for each Document's `content` and stores it on
    `Document.embedding`. Because Marengo embeds text, images, audio, and video
    into one shared space, these embeddings support cross-modal retrieval.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.twelvelabs import TwelveLabsDocumentEmbedder

    # Set the TWELVELABS_API_KEY environment variable
    doc_embedder = TwelveLabsDocumentEmbedder()
    docs = [Document(content="a cat playing piano")]
    docs = doc_embedder.run(documents=docs)["documents"]
    print(docs[0].embedding)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),
        model: str = DEFAULT_MODEL,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
    ) -> None:
        """
        Create a TwelveLabsDocumentEmbedder.

        :param api_key: The TwelveLabs API key. Read from the `TWELVELABS_API_KEY`
            environment variable by default.
        :param model: The Marengo model name.
        :param prefix: A string to add to the beginning of each text before embedding.
        :param suffix: A string to add to the end of each text before embedding.
        :param batch_size: Number of Documents to embed concurrently per request batch.
        :param progress_bar: Whether to show a progress bar while embedding. Can be helpful
            to disable in production deployments to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """
        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

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
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TwelveLabsDocumentEmbedder":
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
                "TwelveLabsDocumentEmbedder expects a list of Documents. To embed a "
                "string, use the TwelveLabsTextEmbedder."
            )
            raise TypeError(msg)

    def _prepare_texts_to_embed(self, documents: list[Document]) -> list[str]:
        """Concatenate each Document's selected meta fields with its content, plus prefix/suffix."""
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    @staticmethod
    def _build_result(documents: list[Document], embeddings: list[list[float]], model: str) -> dict[str, Any]:
        new_documents = [replace(doc, embedding=emb) for doc, emb in zip(documents, embeddings, strict=True)]
        return {"documents": new_documents, "meta": {"model": model}}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embed a list of Documents.

        :param documents: The Documents to embed (their `content` is embedded).
        :returns: A dictionary with keys:
            - `documents`: New Documents that are copies of the inputs with `embedding` populated.
            - `meta`: Metadata about the request (the model used).
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate(documents)
        texts_to_embed = self._prepare_texts_to_embed(documents)
        key = self.api_key.resolve_value() or ""
        embeddings: list[list[float]] = []
        for i in tqdm(
            range(0, len(texts_to_embed), self.batch_size),
            disable=not self.progress_bar,
            desc="Calculating embeddings",
        ):
            for text in texts_to_embed[i : i + self.batch_size]:
                embeddings.append(embed_text(text, self.model, key))
        return self._build_result(documents, embeddings, self.model)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Asynchronously embed a list of Documents.

        Documents within each batch of `batch_size` are embedded concurrently.

        :param documents: The Documents to embed.
        :returns: A dictionary with keys `documents` (copies with `embedding` populated) and `meta`.
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate(documents)
        texts_to_embed = self._prepare_texts_to_embed(documents)
        key = self.api_key.resolve_value() or ""
        embeddings: list[list[float]] = []
        for i in tqdm(
            range(0, len(texts_to_embed), self.batch_size),
            disable=not self.progress_bar,
            desc="Calculating embeddings",
        ):
            batch = texts_to_embed[i : i + self.batch_size]
            batch_embeddings = await asyncio.gather(*(embed_text_async(text, self.model, key) for text in batch))
            embeddings.extend(batch_embeddings)
        return self._build_result(documents, embeddings, self.model)
