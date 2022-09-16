# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from text2speech_nodes import SpeechDocument, SpeechAnswer


SAMPLES_PATH = Path(__file__).parent / "samples"


@pytest.mark.unit
class TestSpeechSchema:

    def test_serialize_speech_document(self):
        speech_doc = SpeechDocument(
            id=12345,
            content_type="audio",
            content="this is the content of the document",
            content_audio=SAMPLES_PATH / "audio" / "this is the content of the document.wav",
            meta={"some": "meta"},
        )
        speech_doc_dict = speech_doc.to_dict()

        assert speech_doc_dict["content"] == "this is the content of the document"
        assert speech_doc_dict["content_audio"] == str(
            (SAMPLES_PATH / "audio" / "this is the content of the document.wav").absolute()
        )


    def test_deserialize_speech_document(self):
        speech_doc = SpeechDocument(
            id=12345,
            content_type="audio",
            content="this is the content of the document",
            content_audio=SAMPLES_PATH / "audio" / "this is the content of the document.wav",
            meta={"some": "meta"},
        )
        assert speech_doc == SpeechDocument.from_dict(speech_doc.to_dict())


    def test_serialize_speech_answer(self):
        speech_answer = SpeechAnswer(
            answer="answer",
            answer_audio=SAMPLES_PATH / "audio" / "answer.wav",
            context="the context for this answer is here",
            context_audio=SAMPLES_PATH / "audio" / "the context for this answer is here.wav",
        )
        speech_answer_dict = speech_answer.to_dict()

        assert speech_answer_dict["answer"] == "answer"
        assert speech_answer_dict["answer_audio"] == str((SAMPLES_PATH / "audio" / "answer.wav").absolute())
        assert speech_answer_dict["context"] == "the context for this answer is here"
        assert speech_answer_dict["context_audio"] == str(
            (SAMPLES_PATH / "audio" / "the context for this answer is here.wav").absolute()
        )


    def test_deserialize_speech_answer(self):
        speech_answer = SpeechAnswer(
            answer="answer",
            answer_audio=SAMPLES_PATH / "audio" / "answer.wav",
            context="the context for this answer is here",
            context_audio=SAMPLES_PATH / "audio" / "the context for this answer is here.wav",
        )
        assert speech_answer == SpeechAnswer.from_dict(speech_answer.to_dict())
