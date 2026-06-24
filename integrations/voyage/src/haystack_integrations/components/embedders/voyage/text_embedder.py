# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

import voyageai
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class VoyageTextEmbedder:
    """
    A component for embedding strings using Voyage AI models.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.voyage import VoyageTextEmbedder

    # Make sure that the environment variable VOYAGE_API_KEY is set

    text_embedder = VoyageTextEmbedder()

    text_to_embed = "I love pizza!"

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    #  'meta': {'model': 'voyage-3.5', 'total_tokens': 4}}
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "voyage-3.5",
        "voyage-3.5-lite",
        "voyage-3-large",
        "voyage-code-3",
        "voyage-finance-2",
        "voyage-law-2",
        "voyage-multilingual-2",
    ]
    """A non-exhaustive list of embed models supported by this component.
    See https://docs.voyageai.com/docs/embeddings for the full list."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        model: str = "voyage-3.5",
        prefix: str = "",
        suffix: str = "",
        input_type: str | None = "query",
        truncation: bool = True,
        output_dimension: int | None = None,
        output_dtype: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Create a VoyageTextEmbedder component.

        :param api_key: The Voyage API key. It can be explicitly provided or automatically read from the
            environment variable `VOYAGE_API_KEY` (recommended).
        :param model: The name of the Voyage model to use.
            Check the list of available models on [Voyage documentation](https://docs.voyageai.com/docs/embeddings).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param input_type: The type of input text. Can be `"query"`, `"document"`, or `None`.
            Voyage prepends a prompt tailored to the input type to improve retrieval quality.
        :param truncation: Whether to truncate the input texts that exceed the model's context length.
            If `False`, an error is raised when an input exceeds the context length.
        :param output_dimension: The number of dimensions of the output embeddings.
            Only supported by a subset of models; defaults to the model's native dimension.
        :param output_dtype: The data type of the output embeddings. Can be `"float"`, `"int8"`, `"uint8"`,
            `"binary"`, or `"ubinary"`. Defaults to `"float"`.
        :param timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.input_type = input_type
        self.truncation = truncation
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.timeout = timeout

        self._client = voyageai.Client(api_key=self.api_key.resolve_value(), timeout=self.timeout)
        self._async_client = voyageai.AsyncClient(api_key=self.api_key.resolve_value(), timeout=self.timeout)

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            prefix=self.prefix,
            suffix=self.suffix,
            input_type=self.input_type,
            truncation=self.truncation,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoyageTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _validate_input(self, text: str) -> None:
        if not isinstance(text, str):
            msg = (
                "VoyageTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the VoyageDocumentEmbedder."
            )
            raise TypeError(msg)

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Embed a string.

        :param text: The string to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: The embedding of the input string.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a string.
        """
        self._validate_input(text)
        text_to_embed = self.prefix + text + self.suffix

        result = self._client.embed(
            texts=[text_to_embed],
            model=self.model,
            input_type=self.input_type,
            truncation=self.truncation,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
        )

        meta = {"model": self.model, "total_tokens": result.total_tokens}
        return {"embedding": result.embeddings[0], "meta": meta}

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, Any]:
        """
        Asynchronously embed a string.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param text: The string to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: The embedding of the input string.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a string.
        """
        self._validate_input(text)
        text_to_embed = self.prefix + text + self.suffix

        result = await self._async_client.embed(
            texts=[text_to_embed],
            model=self.model,
            input_type=self.input_type,
            truncation=self.truncation,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
        )

        meta = {"model": self.model, "total_tokens": result.total_tokens}
        return {"embedding": result.embeddings[0], "meta": meta}
