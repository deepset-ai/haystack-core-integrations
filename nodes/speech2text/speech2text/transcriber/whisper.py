# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Dict, Any

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal  # type: ignore

import logging
from pathlib import Path

try:
    import whisper
except ImportError as e:
    raise ImportError("Could not import Whisper. Run 'pip install git+https://github.com/openai/whisper.git'") from e

from speech2text.transcriber.base import BaseSpeechTranscriber


logger = logging.getLogger(__name__)


class WhisperTranscriber(BaseSpeechTranscriber):
    """
    Converts audio containing speech into its trascription using Whisper.
    """

    def __init__(self, model_size: Literal["tiny", "base", "small", "medium", "large"] = "base"):  # type: ignore
        super().__init__()
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_file: Union[Path, str], sample_rate=16000) -> Dict[str, Any]:
        return self.model.transcribe(str(audio_file))
