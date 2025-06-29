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
class WatsonXDocumentEmbedder:
    """
    Computes document embeddings using IBM watsonx.ai models.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.watsonx.document_embedder import (
        WatsonXDocumentEmbedder,
    )

    documents = [
        Document(content="I love pizza!"),
        Document(content="Pasta is great too"),
    ]

    document_embedder = WatsonXDocumentEmbedder(
        model="ibm/slate-30m-english-rtrvr",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        url="https://us-south.ml.cloud.ibm.com",
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
        model: str = "ibm/slate-30m-english-rtrvr",
        api_key: Secret | None = None,
        url: str = "https://us-south.ml.cloud.ibm.com",
        project_id: Secret | None = None,
        space_id: Secret | None = None,
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
        Creates a WatsonXDocumentEmbedder component.

        :param model:
            The name of the model to use for calculating embeddings.
            Default is "ibm/slate-30m-english-rtrvr".
        :param api_key:
            The WATSONX API key. Can be set via environment variable WATSONX_API_KEY.
        :param url:
            The WATSONX URL for the watsonx.ai service.
            Default is "https://us-south.ml.cloud.ibm.com".
        :param project_id:
            The ID of the Watson Studio project. Either project_id or space_id must be provided.
            Can be set via environment variable WATSONX_PROJECT_ID.
        :param space_id:
            The ID of the Watson Studio space. Either project_id or space_id must be provided.
            Can be set via environment variable WATSONX_SPACE_ID.
        :param truncate_input_tokens:
            Maximum number of tokens to use from the input text.
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
        if api_key is None:
            api_key = Secret.from_env_var("WATSONX_API_KEY")

        if not project_id and not space_id:
            msg = "Either project_id or space_id must be provided"
            raise ValueError(msg)

        self.model = model
        self.api_key = api_key
        self.url = url
        self.project_id = project_id
        if project_id:
            self.space_id = None
        else:
            self.space_id = space_id or Secret.from_env_var("WATSONX_SPACE_ID")
        self.truncate_input_tokens = truncate_input_tokens
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.meta_fields_to_embed = meta_fields_to_embed
        self.embedding_separator = embedding_separator

        # Initialize the embeddings client
        credentials = Credentials(api_key=api_key.resolve_value(), url=url)

        params = {}
        if truncate_input_tokens is not None:
            params["truncate_input_tokens"] = truncate_input_tokens

        project_id_value = None
        space_id_value = None

        if project_id:
            project_id_value = project_id.resolve_value() if isinstance(project_id, Secret) else project_id
            # When project_id is provided, space_id should be None
            space_id_value = None
        elif space_id:
            space_id_value = space_id.resolve_value() if isinstance(space_id, Secret) else space_id

        self.embedder = Embeddings(
            model_id=model,
            credentials=credentials,
            project_id=project_id_value,
            space_id=space_id_value,
            params=params if params else None,
            batch_size=batch_size,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries or 10,
        )

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key.to_dict(),
            url=self.url,
            project_id=self.project_id.to_dict() if self.project_id else None,
            space_id=self.space_id.to_dict() if self.space_id else None,
            truncate_input_tokens=self.truncate_input_tokens,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            concurrency_limit=self.concurrency_limit,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WatsonXDocumentEmbedder:
        """
        Deserializes the component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "project_id", "space_id"])
        return default_from_dict(cls, data)

    def _prepare_text(self, text: str) -> str:
        """
        Prepares text for embedding by adding prefix and suffix.
        """
        return self.prefix + text + self.suffix

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
                "WatsonXDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the WatsonXTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = [self._prepare_text(doc.content or "") for doc in documents]
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
