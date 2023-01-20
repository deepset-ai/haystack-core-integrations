# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from haystack import Document

from speech2text.transcriber import WhisperTranscriber
from speech2text.errors import SpeechToTextNodeError


SAMPLES_PATH = Path(__file__).parent / "samples"


@pytest.fixture(scope="module")
def transcriber():
    return WhisperTranscriber()


@pytest.mark.integration
def test_transcribe(transcriber):
    assert (
        transcriber.transcribe(SAMPLES_PATH / "this is the content of the document.wav")["text"].lower().strip()
        == "this is the content of the document."
    )


@pytest.mark.integration
def test_run(transcriber):
    results, _ = transcriber.run(file_paths=[SAMPLES_PATH / "this is the content of the document.wav"])
    assert len(results["documents"]) == 1

    document = results["documents"][0]
    assert "this is the content of the document." == document.content.lower().strip()
    assert "text" == document.content_type
    assert "audio" in document.meta.keys()
    assert "path" in document.meta["audio"].keys()
