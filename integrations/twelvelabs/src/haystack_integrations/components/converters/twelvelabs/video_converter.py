# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.utils import Secret, deserialize_secrets_inplace

from twelvelabs import TwelveLabs, VideoContext_AssetId, VideoContext_Url

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "pegasus1.5"
DEFAULT_PROMPT = (
    "Describe this video in detail. Include what happens visually, what is said "
    "(transcribe the audio), any on-screen text, and the overall purpose, with "
    "[MM:SS] timestamps."
)
_POLL_INTERVAL = 3.0
_ASSET_TIMEOUT = 900.0
_ANALYZE_TIMEOUT = 1800.0
_MAX_DIRECT_UPLOAD_BYTES = 200 * 1024 * 1024


@component
class TwelveLabsVideoConverter:
    """
    Converts videos to Haystack Documents using TwelveLabs Pegasus.

    Pegasus is a video-language model that analyzes a video on the fly (its
    visuals **and** its own audio ASR) and returns text. Each source video
    becomes one Document whose content is Pegasus's analysis (e.g. a description
    plus a transcript) — no frame extraction or separate transcription step.

    Sources may be publicly accessible direct video URLs or local file paths
    (uploaded to TwelveLabs, up to 200 MB).

    ### Usage example

    ```python
    from haystack_integrations.components.converters.twelvelabs import TwelveLabsVideoConverter

    # Set the TWELVELABS_API_KEY environment variable
    converter = TwelveLabsVideoConverter()
    result = converter.run(sources=["https://example.com/clip.mp4"])
    print(result["documents"][0].content)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("TWELVELABS_API_KEY"),
        model: str = DEFAULT_MODEL,
        prompt: str = DEFAULT_PROMPT,
        temperature: float = 0.2,
        max_tokens: int = 16384,
    ) -> None:
        """
        Create a TwelveLabsVideoConverter.

        :param api_key: The TwelveLabs API key. Read from the `TWELVELABS_API_KEY`
            environment variable by default.
        :param model: The Pegasus model name (`pegasus1.5` or `pegasus1.2`).
        :param prompt: The analysis prompt sent to Pegasus for each video.
        :param temperature: Sampling temperature (0-1).
        :param max_tokens: Maximum output tokens per analysis.
        """
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

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
            prompt=self.prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TwelveLabsVideoConverter":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Convert videos to Documents with Pegasus.

        :param sources: Video sources — publicly accessible direct video URLs or
            local file paths.
        :param meta: Optional metadata to attach to the produced Documents. Either
            a single dict applied to all, or a list aligned with `sources`.
        :returns: A dictionary with key `documents`: the produced Documents.
        """
        meta_list = normalize_metadata(meta, sources_count=len(sources))
        key = self.api_key.resolve_value() or ""
        documents: list[Document] = []

        for source, extra_meta in zip(sources, meta_list, strict=True):
            try:
                text, analysis_id, asset_id = self._analyze_source(source, key)
            except Exception as exc:  # one bad source shouldn't fail the whole batch
                logger.warning("TwelveLabs could not analyze {source}: {error}", source=source, error=str(exc))
                continue
            doc_meta = {
                "source": str(source),
                "asset_id": asset_id,
                "analysis_id": analysis_id,
                "model": self.model,
                "provider": "twelvelabs",
                **extra_meta,
            }
            documents.append(Document(content=text, meta=doc_meta))

        return {"documents": documents}

    # -- TwelveLabs SDK helpers --------------------------------------------- #
    def _analyze_source(self, source: str, key: str) -> tuple[str, str | None, str | None]:
        """Analyze one video with Pegasus, returning (text, response_id, asset_id)."""
        client = TwelveLabs(api_key=key)
        asset_id: str | None = None
        if urlparse(source).scheme in ("http", "https"):
            video: VideoContext_Url | VideoContext_AssetId = VideoContext_Url(type="url", url=source)
        else:
            asset_id = self._upload_asset(client, source)
            video = VideoContext_AssetId(type="asset_id", asset_id=asset_id)

        response = client.analyze(
            model_name=self.model,
            video=video,
            prompt=self.prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_options={"timeout_in_seconds": int(_ANALYZE_TIMEOUT)},
        )
        text = (response.data or "").strip()
        if not text:
            msg = f"TwelveLabs analysis of {source} produced no text"
            raise RuntimeError(msg)
        return text, response.id, asset_id

    def _upload_asset(self, client: TwelveLabs, source: str) -> str:
        """Upload a local video file as a TwelveLabs asset and wait until it's ready."""
        path = Path(source).expanduser().resolve()
        if not path.exists():
            msg = f"Video file not found: {path}"
            raise FileNotFoundError(msg)
        if path.stat().st_size > _MAX_DIRECT_UPLOAD_BYTES:
            msg = f"{path.name} exceeds the 200 MB direct-upload limit; host it and pass a URL."
            raise ValueError(msg)

        asset = client.assets.create(method="direct", file=(path.name, path.read_bytes()))
        asset_id = asset.id
        if not asset_id:
            msg = "TwelveLabs asset upload returned no id"
            raise RuntimeError(msg)
        self._await_asset_ready(client, asset_id)
        return asset_id

    def _await_asset_ready(self, client: TwelveLabs, asset_id: str) -> None:
        deadline = time.monotonic() + _ASSET_TIMEOUT
        while True:
            status = str(client.assets.retrieve(asset_id=asset_id).status or "").lower()
            if status == "ready":
                return
            if status == "failed":
                msg = f"TwelveLabs asset {asset_id} processing failed"
                raise RuntimeError(msg)
            if time.monotonic() > deadline:
                msg = f"TwelveLabs asset {asset_id} not ready in time"
                raise TimeoutError(msg)
            time.sleep(_POLL_INTERVAL)
