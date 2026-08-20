# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from ._media_embed import MODALITIES, detect_modality, embed_media, embed_media_async

DEFAULT_MODEL = "marengo3.0"


@component
class TwelveLabsMultimodalEmbedder:
    """
    Embeds an image, audio, or video into TwelveLabs Marengo's shared multimodal space.

    Marengo embeds text, images, audio, and video into a single vector space, so the vector
    produced here is directly comparable (cosine similarity) with text embeddings from
    `TwelveLabsTextEmbedder` and with other media embeddings from the same model — enabling
    cross-modal retrieval such as text-to-video or image-to-video search. Use this component
    to embed a single media query before searching a document store populated with Marengo
    embeddings.

    Images and audio are embedded synchronously; video uses the asynchronous TwelveLabs
    video-embedding task, which this component submits and then blocks on until it completes.

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.twelvelabs import TwelveLabsMultimodalEmbedder

    # Set the TWELVELABS_API_KEY environment variable
    embedder = TwelveLabsMultimodalEmbedder()
    result = embedder.run(source="https://example.com/cat.jpg")
    print(result["embedding"])
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),
        model: str = DEFAULT_MODEL,
    ) -> None:
        """
        Create a TwelveLabsMultimodalEmbedder.

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
    def from_dict(cls, data: dict[str, Any]) -> "TwelveLabsMultimodalEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _resolve_modality(self, source: str, modality: str | None) -> str:
        resolved = (modality or detect_modality(source)).lower()
        if resolved not in MODALITIES:
            msg = f"Unsupported modality {modality!r}. Expected one of {MODALITIES}."
            raise ValueError(msg)
        return resolved

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, source: str, modality: str | None = None) -> dict[str, Any]:
        """
        Embed a single media source.

        :param source: A publicly accessible URL or a local file path to an image, audio, or video file.
        :param modality: Optional explicit modality (`"image"`, `"audio"` or `"video"`). If omitted,
            it is inferred from the source's file extension / MIME type.
        :returns: A dictionary with keys:
            - `embedding`: The embedding vector for the media in Marengo's shared space.
            - `meta`: Metadata about the request (the model used and the resolved modality).
        :raises TypeError: If `source` is not a string.
        :raises ValueError: If the modality is unsupported or cannot be inferred.
        """
        if not isinstance(source, str):
            msg = "TwelveLabsMultimodalEmbedder expects a string source (a URL or a local file path)."
            raise TypeError(msg)
        resolved = self._resolve_modality(source, modality)
        embedding = embed_media(source, resolved, self.model, self.api_key.resolve_value() or "")
        return {"embedding": embedding, "meta": {"model": self.model, "modality": resolved}}

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, source: str, modality: str | None = None) -> dict[str, Any]:
        """
        Asynchronously embed a single media source.

        See `run` for parameter and return details.

        :param source: A publicly accessible URL or a local file path.
        :param modality: Optional explicit modality; inferred from the source when omitted.
        :returns: A dictionary with keys `embedding` and `meta`.
        :raises TypeError: If `source` is not a string.
        :raises ValueError: If the modality is unsupported or cannot be inferred.
        """
        if not isinstance(source, str):
            msg = "TwelveLabsMultimodalEmbedder expects a string source (a URL or a local file path)."
            raise TypeError(msg)
        resolved = self._resolve_modality(source, modality)
        embedding = await embed_media_async(source, resolved, self.model, self.api_key.resolve_value() or "")
        return {"embedding": embedding, "meta": {"model": self.model, "modality": resolved}}
