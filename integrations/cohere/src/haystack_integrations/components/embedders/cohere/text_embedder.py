# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Any, Dict, List

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack_integrations.components.embedders.cohere.utils import get_async_response, get_response

from cohere import AsyncClient, Client


@component
class CohereTextEmbedder:
    """
    A component for embedding strings using Cohere models.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.cohere import CohereDocumentEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = CohereTextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [-0.453125, 1.2236328, 2.0058594, ...]
    # 'meta': {'api_version': {'version': '1'}, 'billed_units': {'input_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
        model: str = "embed-english-v2.0",
        input_type: str = "search_query",
        api_base_url: str = "https://api.cohere.com",
        truncate: str = "END",
        use_async_client: bool = False,
        timeout: int = 120,
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
        """

        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.api_base_url = api_base_url
        self.truncate = truncate
        self.use_async_client = use_async_client
        self.timeout = timeout

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
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereTextEmbedder":
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

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
        """Embed text.

        :param text: the text to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: the embedding of the text.
            - `meta`: metadata about the request.
        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            msg = (
                "CohereTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the CohereDocumentEmbedder."
            )
            raise TypeError(msg)

        # Establish connection to API

        api_key = self.api_key.resolve_value()
        assert api_key is not None

        if self.use_async_client:
            cohere_client = AsyncClient(
                api_key,
                base_url=self.api_base_url,
                timeout=self.timeout,
                client_name="haystack",
            )
            embedding, metadata = asyncio.run(
                get_async_response(cohere_client, [text], self.model, self.input_type, self.truncate)
            )
        else:
            cohere_client = Client(
                api_key,
                base_url=self.api_base_url,
                timeout=self.timeout,
                client_name="haystack",
            )
            embedding, metadata = get_response(cohere_client, [text], self.model, self.input_type, self.truncate)

        return {"embedding": embedding[0], "meta": metadata}
