# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from ._embed import embed_text, embed_text_async

DEFAULT_MODEL = "marengo3.0"


@component
class TwelveLabsDocumentEmbedder:
    """
    Embeds the text content of Documents using TwelveLabs Marengo.

    Computes a Marengo embedding for each Document's `content` and stores it on
    `Document.embedding`. Because Marengo embeds text, images, audio, and video
    into one shared space, these embeddings support cross-modal retrieval.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.twelvelabs import TwelveLabsDocumentEmbedder

    # Set the TWELVELABS_API_KEY environment variable
    doc_embedder = TwelveLabsDocumentEmbedder()
    docs = [Document(content="a cat playing piano")]
    docs = doc_embedder.run(documents=docs)["documents"]
    print(docs[0].embedding)
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),  # noqa: B008
        model: str = DEFAULT_MODEL,
    ) -> None:
        """
        Create a TwelveLabsDocumentEmbedder.

        :param api_key: The TwelveLabs API key. Read from the `TWELVELABS_API_KEY`
            environment variable by default.
        :param model: The Marengo model name.
        """
        self.api_key = api_key
        self.model = model

    def _get_telemetry_data(self) -> dict[str, Any]:
        """Data sent to Posthog for usage analytics."""
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(self, api_key=self.api_key.to_dict(), model=self.model)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TwelveLabsDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _validate(self, documents: list[Document]) -> None:
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            msg = (
                "TwelveLabsDocumentEmbedder expects a list of Documents. To embed a "
                "string, use the TwelveLabsTextEmbedder."
            )
            raise TypeError(msg)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embed a list of Documents.

        :param documents: The Documents to embed (their `content` is embedded).
        :returns: A dictionary with keys:
            - `documents`: The input Documents, each with `embedding` populated.
            - `meta`: Metadata about the request (the model used).
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate(documents)
        key = self.api_key.resolve_value() or ""
        for doc in documents:
            doc.embedding = embed_text(doc.content or "", self.model, key)
        return {"documents": documents, "meta": {"model": self.model}}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Asynchronously embed a list of Documents.

        :param documents: The Documents to embed.
        :returns: A dictionary with keys `documents` and `meta`.
        :raises TypeError: If the input is not a list of Documents.
        """
        self._validate(documents)
        key = self.api_key.resolve_value() or ""
        for doc in documents:
            doc.embedding = await embed_text_async(doc.content or "", self.model, key)
        return {"documents": documents, "meta": {"model": self.model}}
