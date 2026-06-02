import os

import pytest

from haystack_integrations.components.audio.funasr import FunASRTranscriber


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("FUNASR_TEST_AUDIO_PATH"), reason="FUNASR_TEST_AUDIO_PATH is not set")
def test_transcribe_real_audio():
    transcriber = FunASRTranscriber()

    result = transcriber.run(sources=[os.environ["FUNASR_TEST_AUDIO_PATH"]])

    assert result["documents"]
    assert result["documents"][0].content
