# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from typing import Any, Dict, List, Optional

from haystack import component, default_to_dict
from haystack_integrations.components.embedders.cohere.utils import get_async_response, get_response

from cohere import COHERE_API_URL, AsyncClient, Client


@component
class CohereTextEmbedder:
    """
    A component for embedding strings using Cohere models.

    Usage Example:
    ```python
    from cohere_haystack.embedders.text_embedder import CohereTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = CohereTextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [-0.453125, 1.2236328, 2.0058594, ...]
    # 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v2.0",
        input_type: str = "search_query",
        api_base_url: str = COHERE_API_URL,
        truncate: str = "END",
        use_async_client: bool = False,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        """
        Create a CohereTextEmbedder component.

        :param api_key: The Cohere API key. It can be explicitly provided or automatically read from the environment
            variable COHERE_API_KEY (recommended).
        :param model: The name of the model to use, defaults to `"embed-english-v2.0"`. Supported Models are:
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
        :param max_retries: Maximum number of retries for requests, defaults to `3`.
        :param timeout: Request timeout in seconds, defaults to `120`.
        """

        api_key = api_key or os.environ.get("COHERE_API_KEY")
        # we check whether api_key is None or an empty string
        if not api_key:
            msg = (
                "CohereTextEmbedder expects an API key. "
                "Set the COHERE_API_KEY environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)

        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.api_base_url = api_base_url
        self.truncate = truncate
        self.use_async_client = use_async_client
        self.max_retries = max_retries
        self.timeout = timeout

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary omitting the api_key field.
        """
        return default_to_dict(
            self,
            model=self.model,
            input_type=self.input_type,
            api_base_url=self.api_base_url,
            truncate=self.truncate,
            use_async_client=self.use_async_client,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            msg = (
                "CohereTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the CohereDocumentEmbedder."
            )
            raise TypeError(msg)

        # Establish connection to API

        if self.use_async_client:
            cohere_client = AsyncClient(
                self.api_key,
                api_url=self.api_base_url,
                max_retries=self.max_retries,
                timeout=self.timeout,
                client_name="haystack",
            )
            embedding, metadata = asyncio.run(
                get_async_response(cohere_client, [text], self.model, self.input_type, self.truncate)
            )
        else:
            cohere_client = Client(
                self.api_key,
                api_url=self.api_base_url,
                max_retries=self.max_retries,
                timeout=self.timeout,
                client_name="haystack",
            )
            embedding, metadata = get_response(cohere_client, [text], self.model, self.input_type, self.truncate)

        return {"embedding": embedding[0], "meta": metadata}
