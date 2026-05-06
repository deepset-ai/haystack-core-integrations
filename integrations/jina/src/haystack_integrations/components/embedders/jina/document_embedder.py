# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from typing import Any

import httpx
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


@component
class JinaDocumentEmbedder:
    """
    A component for computing Document embeddings using Jina AI models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.jina import JinaDocumentEmbedder

    # Make sure that the environment variable JINA_API_KEY is set

    document_embedder = JinaDocumentEmbedder(task="retrieval.query")

    doc = Document(content="I love pizza!")

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),  # noqa: B008
        model: str = "jina-embeddings-v3",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        task: str | None = None,
        dimensions: int | None = None,
        late_chunking: bool | None = None,
        *,
        base_url: str = JINA_API_URL,
    ) -> None:
        """
        Create a JinaDocumentEmbedder component.

        :param api_key: The Jina API key.
        :param model: The name of the Jina model to use.
            Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        :param task: The downstream task for which the embeddings will be used.
            The model will return the optimized embeddings for that task.
            Check the list of available tasks on [Jina documentation](https://jina.ai/embeddings/).
        :param dimensions: Number of desired dimension.
            Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
        :param late_chunking: A boolean to enable or disable late chunking.
            Apply the late chunking technique to leverage the model's long-context capabilities for
            generating contextual chunk embeddings.
        :param base_url: The base URL of the Jina API.

            The support of `task` and `late_chunking` parameters is only available for jina-embeddings-v3.
        """
        resolved_api_key = api_key.resolve_value()

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self._headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Accept-Encoding": "identity",
            "Content-type": "application/json",
        }
        self.task = task
        self.dimensions = dimensions
        self.late_chunking = late_chunking

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        kwargs = {
            "api_key": self.api_key.to_dict(),
            "model": self.model_name,
            "base_url": self.base_url,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "batch_size": self.batch_size,
            "progress_bar": self.progress_bar,
            "meta_fields_to_embed": self.meta_fields_to_embed,
            "embedding_separator": self.embedding_separator,
        }
        # Optional parameters, the following two are only supported by embeddings-v3 for now
        if self.task is not None:
            kwargs["task"] = self.task
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        if self.late_chunking is not None:
            kwargs["late_chunking"] = self.late_chunking

        return default_to_dict(self, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JinaDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

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

    def _validate_input(self, documents: list[Document]) -> None:
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "JinaDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the JinaTextEmbedder."
            )
            raise TypeError(msg)

    def _prepare_parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        if self.task is not None:
            parameters["task"] = self.task
        if self.dimensions is not None:
            parameters["dimensions"] = self.dimensions
        if self.late_chunking is not None:
            parameters["late_chunking"] = self.late_chunking
        return parameters

    @staticmethod
    def _process_batch_response(
        response: dict[str, Any], all_embeddings: list[list[float]], metadata: dict[str, Any]
    ) -> None:
        if "data" not in response:
            raise RuntimeError(response["detail"])

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(response["data"], key=lambda e: e["index"])
        embeddings = [result["embedding"] for result in sorted_embeddings]
        all_embeddings.extend(embeddings)
        if "model" not in metadata:
            metadata["model"] = response["model"]
        if "usage" not in metadata:
            metadata["usage"] = dict(response["usage"].items())
        else:
            metadata["usage"]["prompt_tokens"] += response["usage"]["prompt_tokens"]
            metadata["usage"]["total_tokens"] += response["usage"]["total_tokens"]

    def _embed_batch(
        self, texts_to_embed: list[str], batch_size: int, parameters: dict | None = None
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """Embed a list of texts in batches."""
        all_embeddings: list[list[float]] = []
        metadata: dict[str, Any] = {}
        with httpx.Client() as client:
            for i in tqdm(
                range(0, len(texts_to_embed), batch_size),
                disable=not self.progress_bar,
                desc="Calculating embeddings",
            ):
                batch = texts_to_embed[i : i + batch_size]
                response = client.post(
                    self.base_url,
                    json={"input": batch, "model": self.model_name, **(parameters or {})},
                    headers=self._headers,
                ).json()
                self._process_batch_response(response, all_embeddings, metadata)

        return all_embeddings, metadata

    async def _embed_batch_async(
        self, texts_to_embed: list[str], batch_size: int, parameters: dict | None = None
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """Asynchronously embed a list of texts in batches."""
        all_embeddings: list[list[float]] = []
        metadata: dict[str, Any] = {}
        async with httpx.AsyncClient() as client:
            for i in tqdm(
                range(0, len(texts_to_embed), batch_size),
                disable=not self.progress_bar,
                desc="Calculating embeddings",
            ):
                batch = texts_to_embed[i : i + batch_size]
                response = await client.post(
                    self.base_url,
                    json={"input": batch, "model": self.model_name, **(parameters or {})},
                    headers=self._headers,
                )
                self._process_batch_response(response.json(), all_embeddings, metadata)

        return all_embeddings, metadata

    @staticmethod
    def _build_result(
        documents: list[Document], embeddings: list[list[float]], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        new_documents: list[Document] = []
        for doc, emb in zip(documents, embeddings, strict=True):
            new_documents.append(replace(doc, embedding=emb))
        return {"documents": new_documents, "meta": metadata}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Compute the embeddings for a list of Documents.

        :param documents: A list of Documents to embed.
        :returns: A dictionary with following keys:
            - `documents`: List of Documents, each with an `embedding` field containing the computed embedding.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate_input(documents)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        parameters = self._prepare_parameters()
        embeddings, metadata = self._embed_batch(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size, parameters=parameters
        )

        return self._build_result(documents, embeddings, metadata)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Asynchronously compute the embeddings for a list of Documents.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param documents: A list of Documents to embed.
        :returns: A dictionary with following keys:
            - `documents`: List of Documents, each with an `embedding` field containing the computed embedding.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate_input(documents)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        parameters = self._prepare_parameters()
        embeddings, metadata = await self._embed_batch_async(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size, parameters=parameters
        )

        return self._build_result(documents, embeddings, metadata)
