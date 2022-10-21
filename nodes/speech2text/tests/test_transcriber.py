# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from speech2text.transcriber import Wav2VecTranscriber
from speech2text.errors import SpeechToTextNodeError


SAMPLES_PATH = Path(__file__).parent / "samples"


@pytest.mark.integration
class TestTranscriber:

    def test_transcribe(self):
        transcriber = Wav2VecTranscriber()
        assert transcriber.transcribe(SAMPLES_PATH / "this is the content of the document.wav").lower() == "this is the content of the document"


    def test_transcribe_supported_formats(self):
        transcriber = Wav2VecTranscriber()
        transcriber.transcribe(SAMPLES_PATH / "answer.wav")
        with pytest.raises(SpeechToTextNodeError):
            transcriber.transcribe(SAMPLES_PATH / "answer.mp3")