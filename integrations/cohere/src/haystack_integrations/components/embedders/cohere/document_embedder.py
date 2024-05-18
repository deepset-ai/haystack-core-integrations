# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack_integrations.components.embedders.cohere.utils import get_async_response, get_response

from cohere import AsyncClient, Client


@component
class CohereDocumentEmbedder:
    """
    A component for computing Document embeddings using Cohere models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
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
        api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
        model: str = "embed-english-v2.0",
        input_type: str = "search_document",
        api_base_url: str = "https://api.cohere.com",
        truncate: str = "END",
        use_async_client: bool = False,
        timeout: int = 120,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        :param api_key: the Cohere API key.
        :param model: the name of the model to use. Supported Models are:
            `"embed-english-v3.0"`, `"embed-english-light-v3.0"`, `"embed-multilingual-v3.0"`,
            `"embed-multilingual-light-v3.0"`, `"embed-english-v2.0"`, `"embed-english-light-v2.0"`,
            `"embed-multilingual-v2.0"`. This list of all supported models can be found in the
            [model documentation](https://docs.cohere.com/docs/models#representation).
        :param input_type: specifies the type of input you're giving to the model. Supported values are
            "search_document", "search_query", "classification" and "clustering". Not
            required for older versions of the embedding models (meaning anything lower than v3), but is required for
            more recent versions (meaning anything bigger than v2).
        :param api_base_url: the Cohere API Base url.
        :param truncate: truncate embeddings that are too long from start or end, ("NONE"|"START"|"END").
            Passing "START" will discard the start of the input. "END" will discard the end of the input. In both
            cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
            If "NONE" is selected, when the input exceeds the maximum input token length an error will be returned.
        :param use_async_client: flag to select the AsyncClient. It is recommended to use
            AsyncClient for applications with many concurrent calls.
        :param timeout: request timeout in seconds.
        :param batch_size: number of Documents to encode at once.
        :param progress_bar: whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param meta_fields_to_embed: list of meta fields that should be embedded along with the Document text.
        :param embedding_separator: separator used to concatenate the meta fields to the Document text.
        """

        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.api_base_url = api_base_url
        self.truncate = truncate
        self.use_async_client = use_async_client
        self.timeout = timeout
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            input_type=self.input_type,
            api_base_url=self.api_base_url,
            truncate=self.truncate,
            use_async_client=self.use_async_client,
            timeout=self.timeout,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
                Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed: List[str] = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if doc.meta.get(key) is not None
            ]

            text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])  # noqa: RUF005
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]):
        """Embed a list of `Documents`.

        :param documents: documents to embed.
        :returns:  A dictionary with the following keys:
            - `documents`: documents with the `embedding` field set.
            - `meta`: metadata about the embedding process.
        :raises TypeError: if the input is not a list of `Documents`.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "CohereDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the CohereTextEmbedder."
            )
            raise TypeError(msg)

        if not documents:
            # return early if we were passed an empty list
            return {"documents": [], "meta": {}}

        texts_to_embed = self._prepare_texts_to_embed(documents)

        api_key = self.api_key.resolve_value()
        assert api_key is not None

        if self.use_async_client:
            cohere_client = AsyncClient(
                api_key,
                base_url=self.api_base_url,
                timeout=self.timeout,
                client_name="haystack",
            )
            all_embeddings, metadata = asyncio.run(
                get_async_response(cohere_client, texts_to_embed, self.model, self.input_type, self.truncate)
            )
        else:
            cohere_client = Client(
                api_key,
                base_url=self.api_base_url,
                timeout=self.timeout,
                client_name="haystack",
            )
            all_embeddings, metadata = get_response(
                cohere_client,
                texts_to_embed,
                self.model,
                self.input_type,
                self.truncate,
                self.batch_size,
                self.progress_bar,
            )

        for doc, embeddings in zip(documents, all_embeddings):
            doc.embedding = embeddings

        return {"documents": documents, "meta": metadata}
