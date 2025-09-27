# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from google.genai import types
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from more_itertools import batched
from tqdm import tqdm

from haystack_integrations.components.common.google_genai.utils import _get_client

logger = logging.getLogger(__name__)


@component
class GoogleGenAIDocumentEmbedder:
    """
    Computes document embeddings using Google AI models.

    ### Authentication examples

    **1. Gemini Developer API (API Key Authentication)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    document_embedder = GoogleGenAIDocumentEmbedder(model="text-embedding-004")

    **2. Vertex AI (Application Default Credentials)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

    # Using Application Default Credentials (requires gcloud auth setup)
    document_embedder = GoogleGenAIDocumentEmbedder(
        api="vertex",
        vertex_ai_project="my-project",
        vertex_ai_location="us-central1",
        model="text-embedding-004"
    )
    ```

    **3. Vertex AI (API Key Authentication)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    document_embedder = GoogleGenAIDocumentEmbedder(
        api="vertex",
        model="text-embedding-004"
    )
    ```

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = GoogleGenAIDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
        api: Literal["gemini", "vertex"] = "gemini",
        vertex_ai_project: Optional[str] = None,
        vertex_ai_location: Optional[str] = None,
        model: str = "text-embedding-004",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Creates an GoogleGenAIDocumentEmbedder component.

        :param api_key: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
            Not needed if using Vertex AI with Application Default Credentials.
            Go to https://aistudio.google.com/app/apikey for a Gemini API key.
            Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
        :param api: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
        :param vertex_ai_project: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
            Application Default Credentials.
        :param vertex_ai_location: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
            Required when using Vertex AI with Application Default Credentials.
        :param model:
            The name of the model to use for calculating embeddings.
            The default model is `text-embedding-ada-002`.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param batch_size:
            Number of documents to embed at once.
        :param progress_bar:
            If `True`, shows a progress bar when running.
        :param meta_fields_to_embed:
            List of metadata fields to embed along with the document text.
        :param embedding_separator:
            Separator used to concatenate the metadata fields to the document text.
        :param config:
            A dictionary of keyword arguments to configure embedding content configuration `types.EmbedContentConfig`.
            If not specified, it defaults to {"task_type": "SEMANTIC_SIMILARITY"}.
            For more information, see the [Google AI Task types](https://ai.google.dev/gemini-api/docs/embeddings#task-types).
        """
        self._api_key = api_key
        self._api = api
        self._vertex_ai_project = vertex_ai_project
        self._vertex_ai_location = vertex_ai_location
        self._model = model
        self._prefix = prefix
        self._suffix = suffix
        self._batch_size = batch_size
        self._progress_bar = progress_bar
        self._meta_fields_to_embed = meta_fields_to_embed or []
        self._embedding_separator = embedding_separator
        self._config = config if config is not None else {"task_type": "SEMANTIC_SIMILARITY"}

        self._client = _get_client(
            api_key=api_key,
            api=api,
            vertex_ai_project=vertex_ai_project,
            vertex_ai_location=vertex_ai_location,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self._model,
            prefix=self._prefix,
            suffix=self._suffix,
            batch_size=self._batch_size,
            progress_bar=self._progress_bar,
            meta_fields_to_embed=self._meta_fields_to_embed,
            embedding_separator=self._embedding_separator,
            api_key=self._api_key.to_dict(),
            api=self._api,
            vertex_ai_project=self._vertex_ai_project,
            vertex_ai_location=self._vertex_ai_location,
            config=self._config,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleGenAIDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed: List[str] = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key])
                for key in self._meta_fields_to_embed
                if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self._prefix + self._embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self._suffix
            )
            texts_to_embed.append(text_to_embed)

        return texts_to_embed

    def _embed_batch(
        self, texts_to_embed: List[str], batch_size: int
    ) -> Tuple[List[Optional[List[float]]], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """
        resolved_config = types.EmbedContentConfig(**self._config) if self._config else None

        all_embeddings = []
        meta: Dict[str, Any] = {}
        for batch in tqdm(
            batched(texts_to_embed, batch_size), disable=not self._progress_bar, desc="Calculating embeddings"
        ):
            args: Dict[str, Any] = {"model": self._model, "contents": [b[1] for b in batch]}
            if resolved_config:
                args["config"] = resolved_config

            response = self._client.models.embed_content(**args)

            embeddings = []
            if response.embeddings:
                for el in response.embeddings:
                    embeddings.append(el.values if el.values else None)
                all_embeddings.extend(embeddings)
            else:
                all_embeddings.extend([None] * len(batch))

            if "model" not in meta:
                meta["model"] = self._model

        return all_embeddings, meta

    async def _embed_batch_async(
        self, texts_to_embed: List[str], batch_size: int
    ) -> Tuple[List[Optional[List[float]]], Dict[str, Any]]:
        """
        Embed a list of texts in batches asynchronously.
        """

        all_embeddings = []
        meta: Dict[str, Any] = {}
        for batch in tqdm(
            batched(texts_to_embed, batch_size), disable=not self._progress_bar, desc="Calculating embeddings"
        ):
            args: Dict[str, Any] = {"model": self._model, "contents": [b[1] for b in batch]}
            if self._config:
                args["config"] = types.EmbedContentConfig(**self._config) if self._config else None

            response = await self._client.aio.models.embed_content(**args)

            embeddings = []
            if response.embeddings:
                for el in response.embeddings:
                    embeddings.append(el.values if el.values else None)
                all_embeddings.extend(embeddings)
            else:
                all_embeddings.extend([None] * len(batch))

            if "model" not in meta:
                meta["model"] = self._model

        return all_embeddings, meta

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]) -> Union[Dict[str, List[Document]], Dict[str, Any]]:
        """
        Embeds a list of documents.

        :param documents:
            A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            error_message_documents = (
                "GoogleGenAIDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the GoogleGenAITextEmbedder."
            )
            raise TypeError(error_message_documents)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        meta: Dict[str, Any]
        embeddings, meta = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self._batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    async def run_async(self, documents: List[Document]) -> Union[Dict[str, List[Document]], Dict[str, Any]]:
        """
        Embeds a list of documents asynchronously.

        :param documents:
            A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            error_message_documents = (
                "GoogleGenAIDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the GoogleGenAITextEmbedder."
            )
            raise TypeError(error_message_documents)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings, meta = await self._embed_batch_async(texts_to_embed=texts_to_embed, batch_size=self._batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}
