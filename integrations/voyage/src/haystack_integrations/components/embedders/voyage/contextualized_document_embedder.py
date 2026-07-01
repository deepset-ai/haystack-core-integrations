# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from typing import Any, ClassVar

import voyageai
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm


@component
class VoyageContextualizedDocumentEmbedder:
    """
    A component for computing contextualized chunk embeddings using Voyage AI models.

    Unlike standard embedders, each Document is embedded **in the context of the other Documents from the same
    source document**, capturing both local chunk details and global document-level semantics. Documents are
    grouped by the `group_by` meta field, so chunks belonging to the same source are embedded together.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.voyage import VoyageContextualizedDocumentEmbedder

    # Make sure that the environment variable VOYAGE_API_KEY is set

    document_embedder = VoyageContextualizedDocumentEmbedder()

    docs = [
        Document(content="The first chunk of the document.", meta={"source_id": "doc-1"}),
        Document(content="The second chunk of the document.", meta={"source_id": "doc-1"}),
    ]

    result = document_embedder.run(docs)
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "voyage-context-4",
        "voyage-context-3",
    ]
    """A non-exhaustive list of contextualized chunk embedding models supported by this component.
    See https://docs.voyageai.com/docs/contextualized-chunk-embeddings for the full list."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        model: str = "voyage-context-4",
        prefix: str = "",
        suffix: str = "",
        input_type: str | None = "document",
        output_dimension: int | None = None,
        output_dtype: str | None = None,
        timeout: float | None = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        group_by: str | None = "source_id",
    ) -> None:
        """
        Create a VoyageContextualizedDocumentEmbedder component.

        :param api_key: The Voyage API key. It can be explicitly provided or automatically read from the
            environment variable `VOYAGE_API_KEY` (recommended).
        :param model: The name of the Voyage contextualized chunk embedding model to use. Check the list of available
            models on [Voyage documentation](https://docs.voyageai.com/docs/contextualized-chunk-embeddings).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param input_type: The type of input text. Can be `"query"`, `"document"`, or `None`.
            Voyage prepends a prompt tailored to the input type to improve retrieval quality.
        :param output_dimension: The number of dimensions of the output embeddings.
            Supported values are 256, 512, 1024, and 2048; defaults to the model's native dimension.
        :param output_dtype: The data type of the output embeddings. Can be `"float"`, `"int8"`, `"uint8"`,
            `"binary"`, or `"ubinary"`. Defaults to `"float"`.
        :param timeout: Request timeout in seconds.
        :param batch_size: Number of source documents (groups of chunks) to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
            to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        :param group_by: The meta field used to group Documents that belong to the same source document. Chunks
            sharing the same value are embedded together so each embedding is aware of the others. If `None`, all
            input Documents are treated as chunks of a single source document.
        """
        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.input_type = input_type
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.timeout = timeout
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.group_by = group_by

        self._client = voyageai.Client(api_key=self.api_key.resolve_value(), timeout=self.timeout)
        self._async_client = voyageai.AsyncClient(api_key=self.api_key.resolve_value(), timeout=self.timeout)

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            prefix=self.prefix,
            suffix=self.suffix,
            input_type=self.input_type,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            timeout=self.timeout,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            group_by=self.group_by,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoyageContextualizedDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _validate_input(self, documents: list[Document]) -> None:
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "VoyageContextualizedDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the VoyageTextEmbedder."
            )
            raise TypeError(msg)

    def _prepare_text_to_embed(self, document: Document) -> str:
        meta_values_to_embed = [
            str(document.meta[key])
            for key in self.meta_fields_to_embed
            if key in document.meta and document.meta[key] is not None
        ]
        return (
            self.prefix + self.embedding_separator.join([*meta_values_to_embed, document.content or ""]) + self.suffix
        )

    def _group_documents(self, documents: list[Document]) -> list[list[int]]:
        """
        Group the indices of Documents that belong to the same source document, preserving order.
        """
        if self.group_by is None:
            return [list(range(len(documents)))]

        groups: dict[Any, list[int]] = {}
        order: list[Any] = []
        for idx, doc in enumerate(documents):
            key = doc.meta.get(self.group_by)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(idx)
        return [groups[key] for key in order]

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Compute the contextualized chunk embeddings for a list of Documents.

        :param documents: A list of Documents to embed.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents, each with an `embedding` field containing the computed embedding.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate_input(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        texts_to_embed = [self._prepare_text_to_embed(doc) for doc in documents]
        groups = self._group_documents(documents)

        all_embeddings: list[Any] = [None] * len(documents)
        total_tokens = 0
        for i in tqdm(
            range(0, len(groups), self.batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = groups[i : i + self.batch_size]
            inputs = [[texts_to_embed[idx] for idx in group] for group in batch]
            result = self._client.contextualized_embed(
                inputs=inputs,
                model=self.model,
                input_type=self.input_type,
                output_dimension=self.output_dimension,
                output_dtype=self.output_dtype,
                chunk_size=32000,
            )
            for group, res in zip(batch, result.results, strict=True):
                for idx, embedding in zip(group, res.embeddings, strict=True):
                    all_embeddings[idx] = embedding
            total_tokens += result.total_tokens

        new_documents = [replace(doc, embedding=emb) for doc, emb in zip(documents, all_embeddings, strict=True)]
        return {"documents": new_documents, "meta": {"model": self.model, "total_tokens": total_tokens}}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Asynchronously compute the contextualized chunk embeddings for a list of Documents.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param documents: A list of Documents to embed.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents, each with an `embedding` field containing the computed embedding.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate_input(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        texts_to_embed = [self._prepare_text_to_embed(doc) for doc in documents]
        groups = self._group_documents(documents)

        all_embeddings: list[Any] = [None] * len(documents)
        total_tokens = 0
        for i in tqdm(
            range(0, len(groups), self.batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = groups[i : i + self.batch_size]
            inputs = [[texts_to_embed[idx] for idx in group] for group in batch]
            result = await self._async_client.contextualized_embed(
                inputs=inputs,
                model=self.model,
                input_type=self.input_type,
                output_dimension=self.output_dimension,
                output_dtype=self.output_dtype,
                chunk_size=32000,
            )
            for group, res in zip(batch, result.results, strict=True):
                for idx, embedding in zip(group, res.embeddings, strict=True):
                    all_embeddings[idx] = embedding
            total_tokens += result.total_tokens

        new_documents = [replace(doc, embedding=emb) for doc, emb in zip(documents, all_embeddings, strict=True)]
        return {"documents": new_documents, "meta": {"model": self.model, "total_tokens": total_tokens}}
