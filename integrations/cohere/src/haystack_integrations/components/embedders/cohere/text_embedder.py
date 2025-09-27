# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from cohere import AsyncClientV2, ClientV2

from .embedding_types import EmbeddingTypes
from .utils import get_async_response, get_response


@component
class CohereTextEmbedder:
    """
    A component for embedding strings using Cohere models.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.cohere import CohereTextEmbedder

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
        timeout: float = 120.0,
        embedding_type: Optional[EmbeddingTypes] = None,
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
        :param timeout: request timeout in seconds.
        :param embedding_type: the type of embeddings to return. Defaults to float embeddings.
            Note that int8, uint8, binary, and ubinary are only valid for v3 models.
        """

        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.api_base_url = api_base_url
        self.truncate = truncate
        self.timeout = timeout
        self.embedding_type = embedding_type or EmbeddingTypes.FLOAT

        self._client = ClientV2(
            api_key=self.api_key.resolve_value(),
            base_url=self.api_base_url,
            timeout=self.timeout,
            client_name="haystack",
        )

        self._async_client = AsyncClientV2(
            api_key=self.api_key.resolve_value(),
            base_url=self.api_base_url,
            timeout=self.timeout,
            client_name="haystack",
        )

    def _validate_input(self, text: str) -> None:
        if not isinstance(text, str):
            msg = (
                "CohereTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the CohereDocumentEmbedder."
            )
            raise TypeError(msg)

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
            timeout=self.timeout,
            embedding_type=self.embedding_type.value,
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

        # drop legacy use_async_client parameter
        init_params.pop("use_async_client", None)

        deserialize_secrets_inplace(init_params, ["api_key"])

        # Convert embedding_type string to EmbeddingTypes enum value
        init_params["embedding_type"] = EmbeddingTypes.from_str(init_params["embedding_type"])

        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str) -> Dict[str, Union[List[float], Dict[str, Any]]]:
        """
        Embed text.

        :param text:
            the text to embed.
        :returns:
            A dictionary with the following keys:
                - `embedding`: the embedding of the text.
                - `meta`: metadata about the request.
        :raises TypeError:
            If the input is not a string.
        """
        self._validate_input(text=text)

        embedding, metadata = get_response(
            cohere_client=self._client,
            texts=[text],
            model_name=self.model,
            input_type=self.input_type,
            truncate=self.truncate,
            embedding_type=self.embedding_type,
        )

        return {"embedding": embedding[0], "meta": metadata}

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    async def run_async(self, text: str) -> Dict[str, Union[List[float], Dict[str, Any]]]:
        """
        Asynchronously embed text.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

         :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: the embedding of the text.
            - `meta`: metadata about the request.

        :raises TypeError:
            If the input is not a string.
        """
        self._validate_input(text=text)

        embedding, metadata = await get_async_response(
            cohere_async_client=self._async_client,
            texts=[text],
            model_name=self.model,
            input_type=self.input_type,
            truncate=self.truncate,
            embedding_type=self.embedding_type,
        )

        return {"embedding": embedding[0], "meta": metadata}
