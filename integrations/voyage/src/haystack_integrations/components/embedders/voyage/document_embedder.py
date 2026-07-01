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
class VoyageDocumentEmbedder:
    """
    A component for computing Document embeddings using Voyage AI models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.voyage import VoyageDocumentEmbedder

    # Make sure that the environment variable VOYAGE_API_KEY is set

    document_embedder = VoyageDocumentEmbedder()

    doc = Document(content="I love pizza!")

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "voyage-3.5",
        "voyage-3.5-lite",
        "voyage-3-large",
        "voyage-code-3",
        "voyage-finance-2",
        "voyage-law-2",
        "voyage-multilingual-2",
    ]
    """A non-exhaustive list of embed models supported by this component.
    See https://docs.voyageai.com/docs/embeddings for the full list."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        model: str = "voyage-3.5",
        prefix: str = "",
        suffix: str = "",
        input_type: str | None = "document",
        truncation: bool = True,
        output_dimension: int | None = None,
        output_dtype: str | None = None,
        timeout: float | None = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
    ) -> None:
        """
        Create a VoyageDocumentEmbedder component.

        :param api_key: The Voyage API key. It can be explicitly provided or automatically read from the
            environment variable `VOYAGE_API_KEY` (recommended).
        :param model: The name of the Voyage model to use.
            Check the list of available models on [Voyage documentation](https://docs.voyageai.com/docs/embeddings).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param input_type: The type of input text. Can be `"query"`, `"document"`, or `None`.
            Voyage prepends a prompt tailored to the input type to improve retrieval quality.
        :param truncation: Whether to truncate the input texts that exceed the model's context length.
            If `False`, an error is raised when an input exceeds the context length.
        :param output_dimension: The number of dimensions of the output embeddings.
            Only supported by a subset of models; defaults to the model's native dimension.
        :param output_dtype: The data type of the output embeddings. Can be `"float"`, `"int8"`, `"uint8"`,
            `"binary"`, or `"ubinary"`. Defaults to `"float"`.
        :param timeout: Request timeout in seconds.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
            to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """
        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.input_type = input_type
        self.truncation = truncation
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.timeout = timeout
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

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
            truncation=self.truncation,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            timeout=self.timeout,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoyageDocumentEmbedder":
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
                "VoyageDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the VoyageTextEmbedder."
            )
            raise TypeError(msg)

    def _prepare_texts_to_embed(self, documents: list[Document]) -> list[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
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

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Compute the embeddings for a list of Documents.

        :param documents: A list of Documents to embed.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents, each with an `embedding` field containing the computed embedding.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate_input(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        texts_to_embed = self._prepare_texts_to_embed(documents)

        all_embeddings: list[Any] = []
        total_tokens = 0
        for i in tqdm(
            range(0, len(texts_to_embed), self.batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + self.batch_size]
            result = self._client.embed(
                texts=batch,
                model=self.model,
                input_type=self.input_type,
                truncation=self.truncation,
                output_dimension=self.output_dimension,
                output_dtype=self.output_dtype,
            )
            all_embeddings.extend(result.embeddings)
            total_tokens += result.total_tokens

        new_documents = [replace(doc, embedding=emb) for doc, emb in zip(documents, all_embeddings, strict=True)]
        return {"documents": new_documents, "meta": {"model": self.model, "total_tokens": total_tokens}}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Asynchronously compute the embeddings for a list of Documents.

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

        texts_to_embed = self._prepare_texts_to_embed(documents)

        all_embeddings: list[Any] = []
        total_tokens = 0
        for i in tqdm(
            range(0, len(texts_to_embed), self.batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + self.batch_size]
            result = await self._async_client.embed(
                texts=batch,
                model=self.model,
                input_type=self.input_type,
                truncation=self.truncation,
                output_dimension=self.output_dimension,
                output_dtype=self.output_dtype,
            )
            all_embeddings.extend(result.embeddings)
            total_tokens += result.total_tokens

        new_documents = [replace(doc, embedding=emb) for doc, emb in zip(documents, all_embeddings, strict=True)]
        return {"documents": new_documents, "meta": {"model": self.model, "total_tokens": total_tokens}}
