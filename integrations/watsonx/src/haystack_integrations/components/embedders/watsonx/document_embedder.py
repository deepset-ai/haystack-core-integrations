# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings


@component
class WatsonxDocumentEmbedder:
    """
    Computes document embeddings using IBM watsonx.ai models.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.watsonx.document_embedder import WatsonxDocumentEmbedder

    documents = [
        Document(content="I love pizza!"),
        Document(content="Pasta is great too"),
    ]

    document_embedder = WatsonxDocumentEmbedder(
        model="ibm/slate-30m-english-rtrvr-v2",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        api_base_url="https://us-south.ml.cloud.ibm.com",
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    )

    result = document_embedder.run(documents=documents)
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        *,
        model: str = "ibm/slate-30m-english-rtrvr-v2",
        api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),  # noqa: B008
        api_base_url: str = "https://us-south.ml.cloud.ibm.com",
        project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),  # noqa: B008
        truncate_input_tokens: int | None = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 1000,
        concurrency_limit: int = 5,
        timeout: float | None = None,
        max_retries: int | None = None,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
    ):
        """
        Creates a WatsonxDocumentEmbedder component.

        :param model:
            The name of the model to use for calculating embeddings.
            Default is "ibm/slate-30m-english-rtrvr-v2".
        :param api_key:
            The WATSONX API key. Can be set via environment variable WATSONX_API_KEY.
        :param api_base_url:
            The WATSONX URL for the watsonx.ai service.
            Default is "https://us-south.ml.cloud.ibm.com".
        :param project_id:
            The ID of the Watson Studio project.
            Can be set via environment variable WATSONX_PROJECT_ID.
        :param truncate_input_tokens:
            Maximum number of tokens to use from the input text.
            If set to `None` (or not provided), the full input text is used, up to the model's maximum token limit.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param batch_size:
            Number of documents to embed in one API call. Default is 1000.
        :param concurrency_limit:
            Number of parallel requests to make. Default is 5.
        :param timeout:
            Timeout for API requests in seconds.
        :param max_retries:
            Maximum number of retries for API requests.
        """

        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.project_id = project_id
        self.truncate_input_tokens = truncate_input_tokens
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

        # Initialize the embeddings client
        credentials = Credentials(api_key=api_key.resolve_value(), url=api_base_url)

        params = {}
        if truncate_input_tokens is not None:
            params["truncate_input_tokens"] = truncate_input_tokens

        self.embedder = Embeddings(
            model_id=model,
            credentials=credentials,
            project_id=project_id.resolve_value(),
            params=params if params else None,
            batch_size=batch_size,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
            project_id=self.project_id.to_dict(),
            truncate_input_tokens=self.truncate_input_tokens,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            concurrency_limit=self.concurrency_limit,
            timeout=self.timeout,
            max_retries=self.max_retries,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatsonxDocumentEmbedder":  # noqa: UP037
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "project_id"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: list[Document]) -> list[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key])
                for key in self.meta_fields_to_embed
                if key in doc.meta and doc.meta[key]  # noqa: RUF019
            ]
            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)

        return texts_to_embed

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Embeds a list of documents.

        :param documents:
            A list of documents to embed.
        :returns:
            A dictionary with:
            - 'documents': List of Documents with embeddings added
            - 'meta': Information about the model usage
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "WatsonxDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the WatsonxTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        embeddings = self.embedder.embed_documents(texts_to_embed)

        for doc, emb in zip(documents, embeddings, strict=True):
            doc.embedding = emb

        return {
            "documents": documents,
            "meta": {
                "model": self.model,
                "truncate_input_tokens": self.truncate_input_tokens,
                "batch_size": self.batch_size,
            },
        }
