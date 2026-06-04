# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import ComponentDevice

from funasr import AutoModel

logger = logging.getLogger(__name__)

_MIME_TO_SUFFIX: dict[str, str] = {
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/m4a": ".m4a",
    "audio/webm": ".webm",
    "audio/mp4": ".mp4",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


@component
class FunASRTranscriber:
    """
    Transcribes audio files to Documents using [FunASR](https://github.com/modelscope/FunASR).

    FunASR is an open-source speech recognition toolkit from Alibaba DAMO Academy.
    It supports 50+ languages, speaker diarization, and timestamp extraction, and runs
    entirely locally — no API key required.

    Models are downloaded from ModelScope on first use and cached in `~/.cache/modelscope`.

    **Usage Example:**

    ```python
    from haystack_integrations.components.converters.funasr import FunASRTranscriber

    transcriber = FunASRTranscriber()
    result = transcriber.run(sources=["speech.wav", "interview.mp3"])
    documents = result["documents"]
    ```

    **Speaker diarization and punctuation:**

    ```python
    from haystack.utils import ComponentDevice

    transcriber = FunASRTranscriber(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        device=ComponentDevice.from_str("cuda"),
    )
    ```

    **SenseVoice with inverse text normalisation:**

    ```python
    transcriber = FunASRTranscriber(
        model="iic/SenseVoiceSmall",
        generation_kwargs={"use_itn": True, "merge_vad": True, "language": "auto"},
    )
    ```
    """

    def __init__(
        self,
        *,
        model: str = "iic/SenseVoiceSmall",
        vad_model: str | None = "fsmn-vad",
        punc_model: str | None = "ct-punc",
        spk_model: str | None = None,
        device: ComponentDevice | None = None,
        batch_size_s: int = 300,
        store_full_path: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a FunASRTranscriber component.

        :param model: FunASR model name or local path. Defaults to `"iic/SenseVoiceSmall"`,
            a multilingual model supporting 50+ languages that is 5-10x faster than Whisper.
            Alternatives include `"paraformer-zh"` (Chinese) or `"paraformer-en"` (English).
            Browse available models at https://modelscope.github.io/FunASR/model-selection.html.
        :param vad_model: Voice activity detection model used to split long audio into segments.
            Set to `None` to process the audio as a single stream.
            Browse available VAD models at https://www.modelscope.cn/models.
        :param punc_model: Punctuation restoration model. Set to `None` to disable punctuation.
            Browse available punctuation models at https://www.modelscope.cn/models.
        :param spk_model: Speaker diarization model (e.g. `"cam++"`). When set, a `"speakers"`
            key is included in the Document metadata. Defaults to `None` (diarization disabled).
            Browse available speaker diarization models at https://www.modelscope.cn/models.
        :param device: The device to run inference on. If `None`, the default device is selected
            automatically. Use `ComponentDevice.from_str("cuda")` for GPU inference.
        :param batch_size_s: Batch size in seconds for VAD-segmented audio. Larger values
            improve throughput at the cost of memory.
        :param store_full_path: If `True`, store the full audio file path in Document metadata.
            If `False` (default), store only the file name.
        :param generation_kwargs: Extra keyword arguments forwarded to `AutoModel.generate()`.
            Use this for model-specific options such as `use_itn=True` or `merge_vad=True`
            for SenseVoice, or `hotword="..."` for contextual recognition.
        """
        self.model = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.spk_model = spk_model
        self.device = device
        self.batch_size_s = batch_size_s
        self.store_full_path = store_full_path
        self.generation_kwargs: dict[str, Any] = generation_kwargs or {}
        self._asr_model: Any = None

    def warm_up(self) -> None:
        """
        Load the FunASR model into memory.

        Models are downloaded from ModelScope on first call and cached locally.
        This method is idempotent — calling it multiple times is safe.
        """
        if self._asr_model is not None:
            return
        self._asr_model = AutoModel(
            model=self.model,
            vad_model=self.vad_model,
            punc_model=self.punc_model,
            spk_model=self.spk_model,
            device=ComponentDevice.resolve_device(self.device).to_torch_str(),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            vad_model=self.vad_model,
            punc_model=self.punc_model,
            spk_model=self.spk_model,
            device=self.device.to_dict() if self.device else None,
            batch_size_s=self.batch_size_s,
            store_full_path=self.store_full_path,
            generation_kwargs=self.generation_kwargs or None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunASRTranscriber":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if init_params.get("device"):
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Transcribe audio sources to Documents.

        :param sources: Audio file paths (`str` or `Path`) or `ByteStream` objects.
            Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC, and any format that
            FunASR's underlying audio backend (soundfile/ffmpeg) can decode.
        :param meta: Metadata to attach to the produced Documents. Pass a single dict
            to apply the same metadata to all Documents, or a list aligned with `sources`.
        :returns: Dictionary with key `"documents"` — one `Document` per source whose
            `content` holds the full transcript text.
        """
        if not sources:
            return {"documents": []}
        self.warm_up()

        meta_list = normalize_metadata(meta, sources_count=len(sources))
        documents: list[Document] = []

        for source, user_meta in zip(sources, meta_list, strict=True):
            try:
                doc = self._transcribe(source, user_meta)
                if doc is not None:
                    documents.append(doc)
            except Exception as e:
                logger.warning(
                    "Could not transcribe {source}. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )

        return {"documents": documents}

    def _transcribe(
        self,
        source: str | Path | ByteStream,
        user_meta: dict[str, Any],
    ) -> Document | None:
        """Transcribe a single audio source and return a Document, or None on empty result."""
        tmp_path: str | None = None

        if isinstance(source, ByteStream):
            suffix = _mime_to_suffix(source.mime_type)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(source.data)
                tmp_path = tmp.name
            audio_path: str = tmp_path
            raw_name: str = str(source.meta.get("file_path") or source.meta.get("file_name") or "audio")
            source_name = raw_name if self.store_full_path else Path(raw_name).name
        else:
            audio_path = str(source)
            source_name = str(source) if self.store_full_path else Path(source).name

        try:
            results: list[dict[str, Any]] = self._asr_model.generate(
                input=audio_path,
                batch_size_s=self.batch_size_s,
                **self.generation_kwargs,
            )

            if not results:
                logger.warning("FunASR returned no results for {source}.", source=source)
                return None

            text = " ".join(r.get("text", "") for r in results).strip()

            doc_meta: dict[str, Any] = {"file_path": source_name}

            timestamps = [ts for r in results for ts in (r.get("timestamp") or [])]
            if timestamps:
                doc_meta["timestamps"] = timestamps

            speakers = [sp for r in results for sp in (r.get("spk") or [])]
            if speakers:
                doc_meta["speakers"] = speakers

            doc_meta.update(user_meta)
            return Document(content=text, meta=doc_meta)
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)


def _mime_to_suffix(mime_type: str | None) -> str:
    """Return a file extension for the given MIME type, defaulting to `.wav`."""
    if mime_type:
        return _MIME_TO_SUFFIX.get(mime_type.lower(), ".wav")
    return ".wav"
