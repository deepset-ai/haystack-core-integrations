# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import re
from pathlib import Path
from typing import Any

from haystack import Document, component
from haystack.components.converters.utils import normalize_metadata
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream

_TAG_PATTERN = re.compile(r"<\|([^|]+)\|>")


def _load_automodel() -> type[Any]:
    try:
        funasr = importlib.import_module("funasr")
    except ImportError as e:
        msg = "FunASRTranscriber requires the 'funasr' package. Install it with 'pip install funasr-haystack'."
        raise ImportError(msg) from e

    return funasr.AutoModel


def _extract_text(result: dict[str, Any]) -> str:
    text = result.get("text", "")
    if isinstance(text, str):
        return text
    return str(text)


def _strip_sensevoice_tags(text: str) -> str:
    return _TAG_PATTERN.sub("", text).strip()


def _extract_sensevoice_tags(text: str) -> list[str]:
    return _TAG_PATTERN.findall(text)


@component
class FunASRTranscriber:
    """
    Transcribes audio files with FunASR.

    The component wraps the official FunASR `AutoModel` SDK and returns one Haystack
    `Document` per input source. The transcribed text is stored in `Document.content`;
    timestamps, speaker labels, emotion tags, and the raw FunASR result are stored in
    `Document.meta`.

    ### Usage example

    ```python
    from haystack_integrations.components.audio.funasr import FunASRTranscriber

    transcriber = FunASRTranscriber(device="cuda:0", speaker_diarization=True)
    result = transcriber.run(sources=["meeting.wav"])
    print(result["documents"][0].content)
    ```
    """

    def __init__(
        self,
        model: str = "paraformer-zh",
        device: str = "cpu",
        vad_model: str | None = "fsmn-vad",
        punc_model: str | None = "ct-punc",
        spk_model: str | None = None,
        speaker_diarization: bool = False,
        emotion_detection: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the FunASRTranscriber.

        :param model:
            FunASR model name or local model path.
        :param device:
            Device passed to FunASR, for example `"cpu"` or `"cuda:0"`.
        :param vad_model:
            Optional VAD model name or local path. Set to `None` to disable VAD.
        :param punc_model:
            Optional punctuation model name or local path. Set to `None` to disable punctuation.
        :param spk_model:
            Optional speaker model name or local path. If `speaker_diarization` is `True`
            and this is not set, `"cam++"` is used.
        :param speaker_diarization:
            Whether to request speaker diarization metadata from FunASR.
        :param emotion_detection:
            Whether to extract SenseVoice-style emotion and event tags from the transcription.
            Use a FunASR model that emits these tags, such as SenseVoice, for this option.
        :param model_kwargs:
            Additional keyword arguments passed to `funasr.AutoModel`.
        :param generation_kwargs:
            Default keyword arguments passed to `AutoModel.generate`.
        """
        if not device:
            msg = "The 'device' parameter cannot be empty."
            raise ValueError(msg)

        self.model = model
        self.device = device
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.spk_model = spk_model
        self.speaker_diarization = speaker_diarization
        self.emotion_detection = emotion_detection
        self.model_kwargs = model_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}

        self._model: Any | None = None

    def warm_up(self) -> None:
        """Load the FunASR model."""
        if self._model is not None:
            return

        resolved_spk_model = self.spk_model
        if self.speaker_diarization and resolved_spk_model is None:
            resolved_spk_model = "cam++"

        automodel = _load_automodel()
        init_kwargs: dict[str, Any] = {
            "model": self.model,
            "device": self.device,
            **self.model_kwargs,
        }
        if self.vad_model is not None:
            init_kwargs["vad_model"] = self.vad_model
        if self.punc_model is not None:
            init_kwargs["punc_model"] = self.punc_model
        if resolved_spk_model is not None:
            init_kwargs["spk_model"] = resolved_spk_model

        self._model = automodel(**init_kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model=self.model,
            device=self.device,
            vad_model=self.vad_model,
            punc_model=self.punc_model,
            spk_model=self.spk_model,
            speaker_diarization=self.speaker_diarization,
            emotion_detection=self.emotion_detection,
            model_kwargs=self.model_kwargs,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunASRTranscriber":
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Transcribe audio sources.

        :param sources:
            List of audio file paths or `ByteStream` objects.
        :param meta:
            Optional metadata to attach to the produced Documents. Can be a single dictionary
            applied to all Documents or a list aligned with `sources`.
        :param generation_kwargs:
            Additional keyword arguments passed to `AutoModel.generate` for this run.
        :returns:
            A dictionary with key `"documents"` containing transcribed Documents.
        :raises FileNotFoundError:
            If a path source does not exist.
        """
        self.warm_up()
        if self._model is None:
            msg = "The FunASR model was not loaded."
            raise RuntimeError(msg)

        meta_list = normalize_metadata(meta, sources_count=len(sources))
        run_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        if self.speaker_diarization:
            run_generation_kwargs.setdefault("sentence_timestamp", True)
            run_generation_kwargs.setdefault("return_spk_res", True)

        documents: list[Document] = []
        for source, user_meta in zip(sources, meta_list, strict=True):
            funasr_input, source_meta = self._prepare_source(source)
            result = self._model.generate(input=funasr_input, **run_generation_kwargs)
            documents.append(self._document_from_result(result=result, source_meta=source_meta, user_meta=user_meta))

        return {"documents": documents}

    def _prepare_source(self, source: str | Path | ByteStream) -> tuple[str | bytes, dict[str, Any]]:
        if isinstance(source, ByteStream):
            source_meta = dict(source.meta)
            return source.data, source_meta

        path = Path(source)
        if not path.exists():
            msg = f"Audio source does not exist: {path}"
            raise FileNotFoundError(msg)

        return str(path), {"file_path": str(path)}

    def _document_from_result(
        self,
        result: Any,
        source_meta: dict[str, Any],
        user_meta: dict[str, Any],
    ) -> Document:
        raw_result = result[0] if isinstance(result, list) and result else result
        if not isinstance(raw_result, dict):
            raw_result = {"result": raw_result}

        text = _extract_text(raw_result)
        meta: dict[str, Any] = {**source_meta, **user_meta, "raw_result": raw_result}

        for key in ("timestamp", "timestamps", "sentence_info"):
            if key in raw_result:
                meta[key] = raw_result[key]

        if self.emotion_detection:
            tags = _extract_sensevoice_tags(text)
            if tags:
                meta["funasr_tags"] = tags
                text = _strip_sensevoice_tags(text)

        return Document(content=text, meta=meta)
