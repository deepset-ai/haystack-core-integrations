# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from ._embed import embed_text, embed_text_async

DEFAULT_MODEL = "marengo3.0"


@component
class TwelveLabsTextEmbedder:
    """
    Embeds strings using TwelveLabs Marengo.

    Marengo embeds text, images, audio, and video into a single shared vector
    space, so embeddings from this component are directly comparable (cosine
    similarity) with image/video embeddings from the same model — enabling
    cross-modal retrieval. Use it to embed a query before searching a document
    store populated with Marengo embeddings.

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.twelvelabs import TwelveLabsTextEmbedder

    # Set the TWELVELABS_API_KEY environment variable
    text_embedder = TwelveLabsTextEmbedder()
    result = text_embedder.run(text="a cat playing piano")
    print(result["embedding"])
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),
        model: str = DEFAULT_MODEL,
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        """
        Create a TwelveLabsTextEmbedder.

        :param api_key: The TwelveLabs API key. Read from the `TWELVELABS_API_KEY`
            environment variable by default.
        :param model: The Marengo model name.
        :param prefix: A string to add to the beginning of the text before embedding.
        :param suffix: A string to add to the end of the text before embedding.
        """
        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix

    def _get_telemetry_data(self) -> dict[str, Any]:
        """Data sent to Posthog for usage analytics."""
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            prefix=self.prefix,
            suffix=self.suffix,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TwelveLabsTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Embed a single string.

        :param text: The string to embed.
        :returns: A dictionary with keys:
            - `embedding`: The embedding vector for the input string.
            - `meta`: Metadata about the request (the model used).
        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            msg = (
                "TwelveLabsTextEmbedder expects a string as input. To embed a list "
                "of Documents, use the TwelveLabsDocumentEmbedder."
            )
            raise TypeError(msg)
        text_to_embed = self.prefix + text + self.suffix
        embedding = embed_text(text_to_embed, self.model, self.api_key.resolve_value() or "")
        return {"embedding": embedding, "meta": {"model": self.model}}

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, Any]:
        """
        Asynchronously embed a single string.

        :param text: The string to embed.
        :returns: A dictionary with keys `embedding` and `meta`.
        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            msg = "TwelveLabsTextEmbedder expects a string as input."
            raise TypeError(msg)
        text_to_embed = self.prefix + text + self.suffix
        embedding = await embed_text_async(text_to_embed, self.model, self.api_key.resolve_value() or "")
        return {"embedding": embedding, "meta": {"model": self.model}}
