# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import urllib.request
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ByteStream
from haystack.utils import ComponentDevice

from haystack_integrations.components.converters.funasr import FunASRTranscriber


def _make_transcriber(**kwargs) -> FunASRTranscriber:
    return FunASRTranscriber(**kwargs)


def _inject_mock_model(transcriber: FunASRTranscriber, results: list[dict]) -> MagicMock:
    mock_model = MagicMock()
    mock_model.generate.return_value = results
    transcriber._asr_model = mock_model
    return mock_model


class TestFunASRTranscriberInit:
    def test_defaults(self):
        t = _make_transcriber()
        assert t.model == "iic/SenseVoiceSmall"
        assert t.vad_model == "fsmn-vad"
        assert t.punc_model == "ct-punc"
        assert t.spk_model is None
        assert t.device is None
        assert t.batch_size_s == 300
        assert t.store_full_path is False
        assert t.generation_kwargs == {}
        assert t._asr_model is None

    def test_custom_params(self):
        t = _make_transcriber(
            model="paraformer-zh",
            vad_model=None,
            punc_model=None,
            spk_model="cam++",
            device=ComponentDevice.from_str("cuda"),
            batch_size_s=60,
            store_full_path=True,
            generation_kwargs={"use_itn": True},
        )
        assert t.model == "paraformer-zh"
        assert t.vad_model is None
        assert t.punc_model is None
        assert t.spk_model == "cam++"
        assert t.device == ComponentDevice.from_str("cuda")
        assert t.batch_size_s == 60
        assert t.store_full_path is True
        assert t.generation_kwargs == {"use_itn": True}


class TestFunASRTranscriberSerialization:
    def test_to_dict(self):
        t = _make_transcriber()
        d = t.to_dict()
        assert d["type"].endswith("FunASRTranscriber")
        p = d["init_parameters"]
        assert p["model"] == "iic/SenseVoiceSmall"
        assert p["vad_model"] == "fsmn-vad"
        assert p["punc_model"] == "ct-punc"
        assert p["spk_model"] is None
        assert p["device"] is None
        assert p["batch_size_s"] == 300
        assert p["store_full_path"] is False
        assert p["generation_kwargs"] is None

    def test_to_dict_with_generation_kwargs(self):
        t = _make_transcriber(generation_kwargs={"use_itn": True, "merge_vad": True})
        d = t.to_dict()
        assert d["init_parameters"]["generation_kwargs"] == {"use_itn": True, "merge_vad": True}

    def test_to_dict_with_device(self):
        t = _make_transcriber(device=ComponentDevice.from_str("cuda"))
        d = t.to_dict()
        assert d["init_parameters"]["device"] == {"type": "single", "device": "cuda"}

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.converters.funasr.transcriber.FunASRTranscriber",
            "init_parameters": {
                "model": "paraformer-zh",
                "vad_model": "fsmn-vad",
                "punc_model": "ct-punc",
                "spk_model": "cam++",
                "device": {"type": "single", "device": "cuda"},
                "batch_size_s": 60,
                "store_full_path": False,
                "generation_kwargs": {"use_itn": True},
            },
        }
        t = FunASRTranscriber.from_dict(data)
        assert t.model == "paraformer-zh"
        assert t.spk_model == "cam++"
        assert t.device == ComponentDevice.from_str("cuda")
        assert t.generation_kwargs == {"use_itn": True}

    def test_from_dict_no_device(self):
        data = {
            "type": "haystack_integrations.components.converters.funasr.transcriber.FunASRTranscriber",
            "init_parameters": {
                "model": "iic/SenseVoiceSmall",
                "vad_model": "fsmn-vad",
                "punc_model": "ct-punc",
                "spk_model": None,
                "device": None,
                "batch_size_s": 300,
                "store_full_path": False,
                "generation_kwargs": None,
            },
        }
        t = FunASRTranscriber.from_dict(data)
        assert t.device is None

    def test_to_from_dict_roundtrip(self):
        t = _make_transcriber(
            model="paraformer-zh",
            spk_model="cam++",
            batch_size_s=60,
            device=ComponentDevice.from_str("cpu"),
        )
        t2 = FunASRTranscriber.from_dict(t.to_dict())
        assert t.model == t2.model
        assert t.spk_model == t2.spk_model
        assert t.batch_size_s == t2.batch_size_s
        assert t.device == t2.device


class TestFunASRTranscriberWarmUp:
    def test_warm_up_loads_model(self):
        t = _make_transcriber()
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("haystack_integrations.components.converters.funasr.transcriber.AutoModel", mock_cls):
            t.warm_up()

        assert t._asr_model is mock_instance
        mock_cls.assert_called_once_with(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model=None,
            device="cpu",
        )

    def test_warm_up_idempotent(self):
        t = _make_transcriber()
        existing_model = MagicMock()
        t._asr_model = existing_model
        t.warm_up()
        assert t._asr_model is existing_model


class TestFunASRTranscriberRun:
    def test_run_empty_sources(self):
        t = _make_transcriber()
        result = t.run(sources=[])
        assert result == {"documents": []}

    def test_run_single_file(self, tmp_path):
        t = _make_transcriber()
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")
        _inject_mock_model(t, [{"text": "Hello world"}])

        result = t.run(sources=[str(audio_file)])
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Hello world"
        assert result["documents"][0].meta["file_path"] == "test.wav"

    def test_run_multiple_files(self, tmp_path):
        t = _make_transcriber()
        f1, f2 = tmp_path / "a.wav", tmp_path / "b.wav"
        f1.write_bytes(b"x")
        f2.write_bytes(b"x")
        mock = _inject_mock_model(t, [{"text": "one"}])
        mock.generate.side_effect = [
            [{"text": "one"}],
            [{"text": "two"}],
        ]

        result = t.run(sources=[str(f1), str(f2)])
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "one"
        assert result["documents"][1].content == "two"

    def test_run_merges_vad_segments(self, tmp_path):
        t = _make_transcriber()
        audio = tmp_path / "long.wav"
        audio.write_bytes(b"x")
        _inject_mock_model(t, [{"text": "Hello"}, {"text": "world"}])

        result = t.run(sources=[str(audio)])
        assert result["documents"][0].content == "Hello world"

    def test_run_with_metadata(self, tmp_path):
        t = _make_transcriber()
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"x")
        _inject_mock_model(t, [{"text": "Hi"}])

        result = t.run(sources=[str(audio)], meta={"topic": "greeting"})
        assert result["documents"][0].meta["topic"] == "greeting"

    def test_run_with_metadata_list(self, tmp_path):
        t = _make_transcriber()
        f1, f2 = tmp_path / "a.wav", tmp_path / "b.wav"
        f1.write_bytes(b"x")
        f2.write_bytes(b"x")
        mock = _inject_mock_model(t, [{"text": "one"}])
        mock.generate.side_effect = [[{"text": "one"}], [{"text": "two"}]]

        result = t.run(sources=[str(f1), str(f2)], meta=[{"idx": 0}, {"idx": 1}])
        assert result["documents"][0].meta["idx"] == 0
        assert result["documents"][1].meta["idx"] == 1

    def test_run_store_full_path(self, tmp_path):
        t = _make_transcriber(store_full_path=True)
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"x")
        _inject_mock_model(t, [{"text": "Hi"}])

        result = t.run(sources=[str(audio)])
        assert result["documents"][0].meta["file_path"] == str(audio)

    def test_run_with_timestamps(self, tmp_path):
        t = _make_transcriber()
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"x")
        _inject_mock_model(t, [{"text": "Hi", "timestamp": [[0, 500], [500, 1000]]}])

        result = t.run(sources=[str(audio)])
        assert result["documents"][0].meta["timestamps"] == [[0, 500], [500, 1000]]

    def test_run_with_speakers(self, tmp_path):
        t = _make_transcriber(spk_model="cam++")
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"x")
        _inject_mock_model(t, [{"text": "Hi", "spk": [0, 1]}])

        result = t.run(sources=[str(audio)])
        assert result["documents"][0].meta["speakers"] == [0, 1]

    def test_run_no_timestamps_or_speakers_when_absent(self, tmp_path):
        t = _make_transcriber()
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"x")
        _inject_mock_model(t, [{"text": "Hi"}])

        result = t.run(sources=[str(audio)])
        assert "timestamps" not in result["documents"][0].meta
        assert "speakers" not in result["documents"][0].meta

    def test_run_empty_result_skipped(self, tmp_path):
        t = _make_transcriber()
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"x")
        _inject_mock_model(t, [])

        result = t.run(sources=[str(audio)])
        assert result["documents"] == []

    def test_run_error_skips_source(self, tmp_path):
        t = _make_transcriber()
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"x")
        mock = _inject_mock_model(t, [])
        mock.generate.side_effect = RuntimeError("model error")

        result = t.run(sources=[str(audio)])
        assert result["documents"] == []

    def test_run_passes_generation_kwargs(self, tmp_path):
        t = _make_transcriber(generation_kwargs={"use_itn": True, "language": "en"})
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"x")
        mock = _inject_mock_model(t, [{"text": "Hello"}])

        t.run(sources=[str(audio)])
        call_kwargs = mock.generate.call_args
        assert call_kwargs.kwargs.get("use_itn") is True
        assert call_kwargs.kwargs.get("language") == "en"
        assert call_kwargs.kwargs.get("batch_size_s") == 300


class TestFunASRTranscriberByteStream:
    def test_run_bytestream(self):
        t = _make_transcriber()
        bs = ByteStream(data=b"fake audio", mime_type="audio/wav", meta={"file_name": "clip.wav"})
        _inject_mock_model(t, [{"text": "Hello from bytes"}])

        result = t.run(sources=[bs])
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Hello from bytes"
        assert result["documents"][0].meta["file_path"] == "clip.wav"

    def test_run_bytestream_unknown_mime(self):
        t = _make_transcriber()
        bs = ByteStream(data=b"fake", mime_type=None)
        mock = _inject_mock_model(t, [{"text": "text"}])

        t.run(sources=[bs])
        mock.generate.assert_called_once()

    def test_run_bytestream_cleans_up_temp_file(self):
        t = _make_transcriber()
        bs = ByteStream(data=b"x", mime_type="audio/wav")
        _inject_mock_model(t, [{"text": "hi"}])

        created_paths: list[str] = []
        original_ntf = __import__("tempfile").NamedTemporaryFile

        def tracking_ntf(**kwargs):
            f = original_ntf(**kwargs)
            created_paths.append(f.name)
            return f

        with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            t.run(sources=[bs])

        for p in created_paths:
            assert not os.path.exists(p), f"Temp file {p} was not cleaned up"

    def test_run_bytestream_cleans_up_on_error(self):
        t = _make_transcriber()
        bs = ByteStream(data=b"x", mime_type="audio/wav")
        mock = _inject_mock_model(t, [])
        mock.generate.side_effect = RuntimeError("boom")

        created_paths: list[str] = []
        original_ntf = __import__("tempfile").NamedTemporaryFile

        def tracking_ntf(**kwargs):
            f = original_ntf(**kwargs)
            created_paths.append(f.name)
            return f

        with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            t.run(sources=[bs])

        for p in created_paths:
            assert not os.path.exists(p), f"Temp file {p} leaked after error"


@pytest.mark.integration
class TestFunASRTranscriberIntegration:
    def test_transcribe_wav(self, tmp_path):
        audio_url = "https://raw.githubusercontent.com/deepset-ai/haystack/main/test/test_files/audio/answer.wav"
        audio_path = tmp_path / "answer.wav"
        urllib.request.urlretrieve(audio_url, audio_path)  # noqa: S310

        t = FunASRTranscriber(model="iic/SenseVoiceSmall", vad_model=None, punc_model=None)
        t.warm_up()
        result = t.run(sources=[str(audio_path)])
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0].content, str)
        assert len(result["documents"][0].content) > 0
