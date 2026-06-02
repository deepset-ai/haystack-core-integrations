import sys
import types
from typing import ClassVar

import pytest
from haystack import Pipeline
from haystack.dataclasses import ByteStream

from haystack_integrations.components.audio.funasr import FunASRTranscriber


class MockAutoModel:
    init_kwargs: ClassVar[dict | None] = None
    generate_calls: ClassVar[list[dict]] = []
    results: ClassVar[list[dict]] = [{"text": "hello world", "timestamp": [[0, 100]], "sentence_info": [{"spk": 0}]}]

    def __init__(self, **kwargs):
        MockAutoModel.init_kwargs = kwargs

    def generate(self, **kwargs):
        MockAutoModel.generate_calls.append(kwargs)
        return MockAutoModel.results


@pytest.fixture(autouse=True)
def mock_funasr(monkeypatch):
    MockAutoModel.init_kwargs = None
    MockAutoModel.generate_calls = []
    MockAutoModel.results = [{"text": "hello world", "timestamp": [[0, 100]], "sentence_info": [{"spk": 0}]}]
    module = types.ModuleType("funasr")
    module.AutoModel = MockAutoModel
    monkeypatch.setitem(sys.modules, "funasr", module)


def test_init_passes_model_configuration():
    transcriber = FunASRTranscriber(
        model="custom-model",
        device="cuda:0",
        vad_model="vad",
        punc_model="punc",
        spk_model="speaker",
        model_kwargs={"disable_pbar": True},
    )
    transcriber.warm_up()

    assert MockAutoModel.init_kwargs == {
        "model": "custom-model",
        "device": "cuda:0",
        "disable_pbar": True,
        "vad_model": "vad",
        "punc_model": "punc",
        "spk_model": "speaker",
    }


def test_speaker_diarization_uses_default_speaker_model_and_generation_flags(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    transcriber = FunASRTranscriber(speaker_diarization=True)
    result = transcriber.run(sources=[audio_path])

    assert MockAutoModel.init_kwargs["spk_model"] == "cam++"
    assert MockAutoModel.generate_calls[0]["sentence_timestamp"] is True
    assert MockAutoModel.generate_calls[0]["return_spk_res"] is True
    assert result["documents"][0].meta["sentence_info"] == [{"spk": 0}]


def test_run_transcribes_path_with_timestamps(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    transcriber = FunASRTranscriber()
    result = transcriber.run(sources=[audio_path], meta={"language": "en"})

    document = result["documents"][0]
    assert document.content == "hello world"
    assert document.meta["file_path"] == str(audio_path)
    assert document.meta["language"] == "en"
    assert document.meta["timestamp"] == [[0, 100]]
    assert MockAutoModel.generate_calls[0]["input"] == str(audio_path)


def test_run_transcribes_bytestream():
    transcriber = FunASRTranscriber()
    result = transcriber.run(sources=[ByteStream(data=b"audio", meta={"file_path": "audio.wav"})])

    assert result["documents"][0].content == "hello world"
    assert MockAutoModel.generate_calls[0]["input"] == b"audio"


def test_emotion_detection_extracts_sensevoice_tags(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    MockAutoModel.results = [{"text": "<|en|><|HAPPY|><|Speech|>Hello"}]

    transcriber = FunASRTranscriber(model="iic/SenseVoiceSmall", emotion_detection=True)
    result = transcriber.run(sources=[audio_path])

    document = result["documents"][0]
    assert document.content == "Hello"
    assert document.meta["funasr_tags"] == ["en", "HAPPY", "Speech"]


def test_missing_source_raises_file_not_found():
    transcriber = FunASRTranscriber()

    with pytest.raises(FileNotFoundError):
        transcriber.run(sources=["missing.wav"])


def test_empty_device_raises_value_error():
    with pytest.raises(ValueError):
        FunASRTranscriber(device="")


def test_to_dict_from_dict_roundtrip():
    transcriber = FunASRTranscriber(
        model="custom",
        device="cuda:0",
        speaker_diarization=True,
        emotion_detection=True,
        generation_kwargs={"batch_size_s": 300},
    )

    restored = FunASRTranscriber.from_dict(transcriber.to_dict())

    assert MockAutoModel.init_kwargs is None
    assert restored.model == "custom"
    assert restored.device == "cuda:0"
    assert restored.speaker_diarization is True
    assert restored.emotion_detection is True
    assert restored.generation_kwargs == {"batch_size_s": 300}


def test_pipeline_serialization_roundtrip():
    pipeline = Pipeline()
    pipeline.add_component("transcriber", FunASRTranscriber())

    restored = Pipeline.from_dict(pipeline.to_dict())

    assert isinstance(restored.get_component("transcriber"), FunASRTranscriber)
    assert MockAutoModel.init_kwargs is None


def test_warm_up_is_idempotent():
    transcriber = FunASRTranscriber()

    transcriber.warm_up()
    transcriber.warm_up()

    assert MockAutoModel.init_kwargs == {
        "model": "paraformer-zh",
        "device": "cpu",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
    }
