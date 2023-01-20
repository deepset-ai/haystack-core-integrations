# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from speech2text.transcriber import WhisperTranscriber
from speech2text.errors import SpeechToTextNodeError


SAMPLES_PATH = Path(__file__).parent / "samples"


@pytest.mark.integration
def test_transcribe():
    transcriber = WhisperTranscriber()
    assert (
        transcriber.transcribe(SAMPLES_PATH / "this is the content of the document.wav").lower()
        == "this is the content of the document"
    )
