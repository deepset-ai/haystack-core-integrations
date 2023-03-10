# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf
from haystack.schema import Span, Answer, Document
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import ffmpeg

from text2speech import AnswerToSpeech, DocumentToSpeech
from text2speech.utils import TextToSpeech



SAMPLES_PATH = Path(__file__).parent / "samples"

class WhisperHelper:
    def __init__(self, model):
        self._processor = WhisperProcessor.from_pretrained(model)
        self._model = WhisperForConditionalGeneration.from_pretrained(model)
        self._model.config.forced_decoder_ids = None

    def transcribe(self, media_file: str):
        output, _ = (
            ffmpeg.input(media_file)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        data = np.frombuffer(output, np.int16).flatten().astype(np.float32) / 32768.0

        features = self._processor(data, sampling_rate=16000, return_tensors="pt").input_features
        tokens = self._model.generate(features)

        return self._processor.batch_decode(tokens, skip_special_tokens=True)


@pytest.fixture(scope="session", autouse=True)
def whisper_helper():
    return WhisperHelper("openai/whisper-medium")

@pytest.mark.integration
class TestTextToSpeech:
    def test_text_to_speech_audio_data(self, tmp_path, whisper_helper: WhisperHelper):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 4535, "always_fix_seed": True},
        )

        audio_data = text2speech.text_to_audio_data(text="answer")

        sf.write(
            data=audio_data,
            file=str(tmp_path / "audio1.wav"),
            format="wav",
            subtype="PCM_16",
            samplerate=text2speech.model.fs,
        )

        expedtec_doc = whisper_helper.transcribe(str(SAMPLES_PATH / "audio" / "answer.wav"))
        generated_doc = whisper_helper.transcribe(str(tmp_path / "audio1.wav"))

        assert expedtec_doc == generated_doc

    def test_text_to_speech_audio_file(self, tmp_path):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        expected_audio_data, _ = sf.read(SAMPLES_PATH / "answer.wav")
        audio_file = text2speech.text_to_audio_file(
            text="answer", generated_audio_dir=tmp_path / "test_audio"
        )
        assert os.path.exists(audio_file)
        assert np.allclose(expected_audio_data, sf.read(audio_file)[0], atol=0.001)

    def test_text_to_speech_compress_audio(self, tmp_path):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        expected_audio_file = SAMPLES_PATH / "answer.wav"
        audio_file = text2speech.text_to_audio_file(
            text="answer",
            generated_audio_dir=tmp_path / "test_audio",
            audio_format="mp3",
        )
        assert os.path.exists(audio_file)
        assert audio_file.suffix == ".mp3"
        # FIXME find a way to make sure the compressed audio is similar enough to the wav version.
        # At a manual inspection, the code seems to be working well.

    def test_text_to_speech_naming_function(self, tmp_path):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        expected_audio_file = SAMPLES_PATH / "answer.wav"
        audio_file = text2speech.text_to_audio_file(
            text="answer",
            generated_audio_dir=tmp_path / "test_audio",
            audio_naming_function=lambda text: text,
        )
        assert os.path.exists(audio_file)
        assert audio_file.name == expected_audio_file.name
        assert np.allclose(
            sf.read(expected_audio_file)[0], sf.read(audio_file)[0], atol=0.001
        )


@pytest.mark.integration
class TestAnswerToSpeech:
    def test_answer_to_speech(self, tmp_path):
        text_answer = Answer(
            answer="answer",
            type="extractive",
            context="the context for this answer is here",
            offsets_in_document=[Span(31, 37)],
            offsets_in_context=[Span(21, 27)],
            meta={"some_meta": "some_value"},
        )
        expected_audio_answer = SAMPLES_PATH / "answer.wav"
        expected_audio_context = (
            SAMPLES_PATH / "the context for this answer is here.wav"
        )

        answer2speech = AnswerToSpeech(
            generated_audio_dir=tmp_path / "test_audio",
            audio_params={"audio_naming_function": lambda text: text},
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        results, _ = answer2speech.run(answers=[text_answer])

        audio_answer: Answer = results["answers"][0]
        assert isinstance(audio_answer, Answer)
        assert audio_answer.meta["audio"]["answer"]["path"] == expected_audio_answer
        assert audio_answer["audio"]["context"]["path"] == expected_audio_context
        assert audio_answer.answer == "answer"
        assert audio_answer.context == "the context for this answer is here"
        assert audio_answer.offsets_in_document == [Span(31, 37)]
        assert audio_answer.offsets_in_context == [Span(21, 27)]
        assert audio_answer.meta["some_meta"] == "some_value"
        assert audio_answer.meta["audio"]["answer"]["path"] == "wav"
        assert audio_answer.meta["audio"]["context"]["path"] == "wav"

        assert np.allclose(
            sf.read(audio_answer.answer_audio)[0],
            sf.read(expected_audio_answer)[0],
            atol=0.001,
        )
        assert np.allclose(
            sf.read(audio_answer.context_audio)[0],
            sf.read(expected_audio_context)[0],
            atol=0.001,
        )


@pytest.mark.integration
class TestDocumentToSpeech:
    def test_document_to_speech(self, tmp_path):
        text_doc = Document(
            content="this is the content of the document",
            content_type="text",
            meta={"name": "test_document.txt"},
        )
        expected_audio_content = (
            SAMPLES_PATH / "this is the content of the document.wav"
        )

        doc2speech = DocumentToSpeech(
            generated_audio_dir=tmp_path / "test_audio",
            audio_params={"audio_naming_function": lambda text: text},
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        results, _ = doc2speech.run(documents=[text_doc])

        audio_doc: Document = results["documents"][0]
        assert isinstance(audio_doc, Document)
        assert audio_doc.content_type == "text"
        assert audio_doc.meta["audio"]["content"]["path"] == expected_audio_content
        assert audio_doc.content == "this is the content of the document"
        assert audio_doc.meta["name"] == "test_document.txt"
        assert audio_doc.meta["audio"]["content"]["format"] == "wav"

        assert np.allclose(
            sf.read(audio_doc.content_audio)[0],
            sf.read(expected_audio_content)[0],
            atol=0.001,
        )
