# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import types
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from more_itertools import batched
from tqdm import tqdm

logger = logging.getLogger(__name__)


@component
class GoogleGenAIDocumentEmbedder:
    """
    Computes document embeddings using Google AI models.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders import GoogleGenAIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = GoogleGenAIDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        *,
        api_key: Secret = Secret.from_env_var("GOOGLE_API_KEY"),
        model: str = "text-embedding-004",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an GoogleGenAIDocumentEmbedder component.

        Before initializing the component, you can set the 'GoogleGenAI_TIMEOUT' and 'GoogleGenAI_MAX_RETRIES'
        environment variables to override the `timeout` and `max_retries` parameters respectively
        in the GoogleGenAI client.

        :param api_key:
            The Google API key.
            You can set it with the environment variable `GOOGLE_API_KEY`, or pass it via this parameter
            during initialization.
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
            For more information, see the [Google AI Task types](https://ai.google.dev/gemini-api/docs/embeddings#task-types).
        """
        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.client = genai.Client(api_key=api_key.resolve_value())
        self.config = config if config is not None else {"task_type": "SEMANTIC_SIMILARITY"}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            api_key=self.api_key.to_dict(),
            config=self.config,
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

    def _prepare_texts_to_embed(self, documents: List[Document]) -> Dict[str, str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = {}
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]

            texts_to_embed[doc.id] = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )

        return texts_to_embed

    def _embed_batch(self, texts_to_embed: Dict[str, str], batch_size: int) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        meta: Dict[str, Any] = {}
        for batch in tqdm(
            batched(texts_to_embed.items(), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            args: Dict[str, Any] = {"model": self.model, "contents": [b[1] for b in batch]}
            if self.config:
                args["config"] = types.EmbedContentConfig(**self.config) if self.config else None

            try:
                response = self.client.models.embed_content(**args)
            except Exception as exc:
                ids = ", ".join(b[0] for b in batch)
                msg = "Failed embedding of documents {ids} caused by {exc}"
                logger.exception(msg, ids=ids, exc=exc)
                continue

            embeddings = [el.values for el in response.embeddings]
            all_embeddings.extend(embeddings)

            if "model" not in meta:
                meta["model"] = self.model

        return all_embeddings, meta

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]) -> Dict[str, Union[List[Document], Dict[str, Any]]]:
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

        embeddings, meta = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}
