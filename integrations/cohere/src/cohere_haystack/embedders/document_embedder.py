# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from typing import Any, Dict, List, Optional

from cohere import COHERE_API_URL, AsyncClient, Client
from haystack import Document, component, default_to_dict

from cohere_haystack.embedders.utils import get_async_response, get_response


@component
class CohereDocumentEmbedder:
    """
    A component for computing Document embeddings using Cohere models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage Example:
    ```python
    from haystack import Document
    from cohere_haystack.embedders.document_embedder import CohereDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = CohereDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [-0.453125, 1.2236328, 2.0058594, ...]
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "embed-english-v2.0",
        input_type: str = "search_document",
        api_base_url: str = COHERE_API_URL,
        truncate: str = "END",
        use_async_client: bool = False,
        max_retries: int = 3,
        timeout: int = 120,
        batch_size: int = 32,
        progress_bar: bool = True,
        metadata_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a CohereDocumentEmbedder component.

        :param api_key: The Cohere API key. It can be explicitly provided or automatically read from the environment
            variable COHERE_API_KEY (recommended).
        :param model_name: The name of the model to use, defaults to `"embed-english-v2.0"`. Supported Models are:
            `"embed-english-v3.0"`, `"embed-english-light-v3.0"`, `"embed-multilingual-v3.0"`,
            `"embed-multilingual-light-v3.0"`, `"embed-english-v2.0"`, `"embed-english-light-v2.0"`,
            `"embed-multilingual-v2.0"`. This list of all supported models can be found in the
            [model documentation](https://docs.cohere.com/docs/models#representation).
        :param input_type: Specifies the type of input you're giving to the model. Supported values are
        "search_document", "search_query", "classification" and "clustering". Defaults to "search_document". Not
        required for older versions of the embedding models (meaning anything lower than v3), but is required for more
        recent versions (meaning anything bigger than v2).
        :param api_base_url: The Cohere API Base url, defaults to `https://api.cohere.ai/v1/embed`.
        :param truncate: Truncate embeddings that are too long from start or end, ("NONE"|"START"|"END"), defaults to
            `"END"`. Passing START will discard the start of the input. END will discard the end of the input. In both
            cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
            If NONE is selected, when the input exceeds the maximum input token length an error will be returned.
        :param use_async_client: Flag to select the AsyncClient, defaults to `False`. It is recommended to use
            AsyncClient for applications with many concurrent calls.
        :param max_retries: maximal number of retries for requests, defaults to `3`.
        :param timeout: request timeout in seconds, defaults to `120`.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param metadata_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """

        if api_key is None:
            try:
                api_key = os.environ["COHERE_API_KEY"]
            except KeyError as error_msg:
                msg = (
                    "CohereDocumentEmbedder expects an Cohere API key. Please provide one by setting the environment "
                    "variable COHERE_API_KEY (recommended) or by passing it explicitly."
                )
                raise ValueError(msg) from error_msg

        self.api_key = api_key
        self.model_name = model_name
        self.input_type = input_type
        self.api_base_url = api_base_url
        self.truncate = truncate
        self.use_async_client = use_async_client
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary omitting the api_key field.
        """
        return default_to_dict(
            self,
            model_name=self.model_name,
            input_type=self.input_type,
            api_base_url=self.api_base_url,
            truncate=self.truncate,
            use_async_client=self.use_async_client,
            max_retries=self.max_retries,
            timeout=self.timeout,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed: List[str] = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.metadata_fields_to_embed if doc.meta.get(key) is not None
            ]

            text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])  # noqa: RUF005
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    @component.output_types(documents=List[Document], metadata=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents: A list of Documents to embed.
        """

        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "CohereDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the CohereTextEmbedder."
            )
            raise TypeError(msg)

        if not documents:
            # return early if we were passed an empty list
            return {"documents": [], "metadata": {}}

        texts_to_embed = self._prepare_texts_to_embed(documents)

        if self.use_async_client:
            cohere_client = AsyncClient(
                self.api_key, api_url=self.api_base_url, max_retries=self.max_retries, timeout=self.timeout
            )
            all_embeddings, metadata = asyncio.run(
                get_async_response(cohere_client, texts_to_embed, self.model_name, self.input_type, self.truncate)
            )
        else:
            cohere_client = Client(
                self.api_key, api_url=self.api_base_url, max_retries=self.max_retries, timeout=self.timeout
            )
            all_embeddings, metadata = get_response(
                cohere_client,
                texts_to_embed,
                self.model_name,
                self.input_type,
                self.truncate,
                self.batch_size,
                self.progress_bar,
            )

        for doc, embeddings in zip(documents, all_embeddings):
            doc.embedding = embeddings

        return {"documents": documents, "metadata": metadata}
